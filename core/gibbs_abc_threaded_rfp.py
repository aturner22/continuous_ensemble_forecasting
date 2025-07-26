import os
import psutil
import numpy as np
import torch
from typing import Any
from tqdm import tqdm
from core.evaluation import (
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)

INITIAL_ALPHA_RANGE = (0.05, 1.5)
PROPOSAL_SCALE = 0.05
MIN_ALPHA = 1e-4
ADAPT_EVERY = 5
ADAPT_FACTOR = 0.85
EPS_ENERGY = 1e-12
CHECKPOINT_FILE = "gibbs_checkpoint_step.npz"


class DynamicBatchManager:
    def __init__(self, device, initial_batch_size: int = 64, min_batch_size: int = 1):
        self.device = device
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_successful_batch_size = initial_batch_size
        self.memory_history = []

    def get_memory_stats(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            free = total - allocated
            return allocated, reserved, total, free
        return 0, 0, psutil.virtual_memory().total, psutil.virtual_memory().available

    def try_batch_size(self, batch_size: int, test_fn):
        try:
            allocated_before, _, _, _ = self.get_memory_stats()
            result = test_fn(batch_size)
            allocated_after, _, _, _ = self.get_memory_stats()
            memory_used = allocated_after - allocated_before
            self.memory_history.append((batch_size, memory_used))
            self.max_successful_batch_size = max(self.max_successful_batch_size, batch_size)
            return result, True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return None, False
            raise e

    def find_optimal_batch_size(self, test_fn, target_batch_size: int | None = None):
        if target_batch_size is None:
            target_batch_size = self.current_batch_size
        low = self.min_batch_size
        high = min(target_batch_size * 2, 1024)
        best_batch_size = self.min_batch_size
        print(f"[DynamicBatch] Finding optimal batch size between {low} and {high}")
        while low <= high:
            mid = (low + high) // 2
            print(f"[DynamicBatch] Testing batch size: {mid}")
            _, success = self.try_batch_size(mid, test_fn)
            if success:
                best_batch_size = mid
                low = mid + 1
                print(f"[DynamicBatch] ✓ Batch size {mid} succeeded")
            else:
                high = mid - 1
                print(f"[DynamicBatch] ✗ Batch size {mid} failed (OOM)")
        self.current_batch_size = best_batch_size
        print(f"[DynamicBatch] Optimal batch size found: {best_batch_size}")
        return best_batch_size

    def predict_safe_batch_size(self, target_samples: int):
        if not self.memory_history:
            return self.current_batch_size
        recent_history = self.memory_history[-5:]
        if recent_history:
            avg_memory_per_sample = sum(mem / batch for batch, mem in recent_history) / len(recent_history)
            _, _, total_memory, free_memory = self.get_memory_stats()
            usable_memory = min(free_memory, total_memory * 0.8)
            predicted_batch_size = int(usable_memory / (avg_memory_per_sample * target_samples))
            predicted_batch_size = max(self.min_batch_size, min(predicted_batch_size, self.max_successful_batch_size))
            print(
                f"[DynamicBatch] Predicted safe batch size: {predicted_batch_size} "
                f"(free: {free_memory/1e9:.1f}GB, avg per sample: {avg_memory_per_sample/1e6:.1f}MB)"
            )
            return predicted_batch_size
        return self.current_batch_size


def compute_safe_batch_size(
    ensemble_size: int,
    num_variables: int,
    spatial_height: int,
    spatial_width: int,
    num_outputs: int,
    dtype_bytes: int = 4,
    model_overhead: float = 2.0,
    safety_divisor: float = 10.0,
    available_bytes: int | None = None,
) -> int:
    if available_bytes is None:
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            available_bytes = total - max(reserved, allocated)
        else:
            available_bytes = psutil.virtual_memory().available
    spatial = spatial_height * spatial_width
    tensor_bytes = (
        (num_variables + 2) * spatial
        + num_outputs * spatial
        + num_variables * ensemble_size * spatial * 3
    ) * dtype_bytes * model_overhead
    initial_estimate = max(1, int(available_bytes // (tensor_bytes * safety_divisor * 4)))
    return min(initial_estimate, 16)


def generate_joint_rfp(
    reference_tensor: torch.Tensor,
    alpha_matrix: torch.Tensor,
    batch_size: int,
    ensemble_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    P = alpha_matrix.shape[0]
    T, V = reference_tensor.shape[0], reference_tensor.shape[1]
    idx1 = torch.randint(0, T, (batch_size, ensemble_size), device=device, generator=generator)
    idx2 = torch.randint(0, T, (batch_size, ensemble_size), device=device, generator=generator)
    diff = reference_tensor[idx1] - reference_tensor[idx2]
    energy = torch.sqrt(diff.pow(2).mean(dim=(2, 3, 4), keepdim=True).clamp_min(EPS_ENERGY))
    diff_norm = diff / energy
    perturb = alpha_matrix.reshape(P, 1, 1, V, 1, 1) * diff_norm.unsqueeze(0)
    return perturb


def memory_efficient_crps(ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    K = ensemble.shape[0]
    absolute_error = torch.abs(ensemble - target.unsqueeze(0)).mean(dim=0)

    ensemble_flat = ensemble.reshape(K, -1)
    spatial_total = ensemble_flat.shape[1]
    pairwise_upper_sum = torch.zeros(spatial_total, device=ensemble.device, dtype=ensemble.dtype)

    chunk_size = min(K, 32)
    for i in range(0, K, chunk_size):
        end_i = min(i + chunk_size, K)
        chunk_i = ensemble_flat[i:end_i]
        a = end_i - i

        # within-block (i,i): sum over all ordered pairs then halve
        diffs_same = torch.abs(chunk_i.unsqueeze(1) - chunk_i.unsqueeze(0))  # [a,a,S]
        block_sum_all_pairs = diffs_same.sum(dim=(0, 1))
        pairwise_upper_sum += 0.5 * block_sum_all_pairs

        # cross-block (i,j), j>i
        j_start = end_i
        for j in range(j_start, K, chunk_size):
            end_j = min(j + chunk_size, K)
            chunk_j = ensemble_flat[j:end_j]
            diffs_cross = torch.abs(chunk_i.unsqueeze(1) - chunk_j.unsqueeze(0))  # [a,b,S]
            pairwise_upper_sum += diffs_cross.sum(dim=(0, 1))

    pairwise_mean = (2.0 * pairwise_upper_sum) / (K * K)
    pairwise_mean = pairwise_mean.view_as(target)
    return absolute_error - 0.5 * pairwise_mean


def compute_crps_for_proposal(
    ensemble_output: torch.Tensor,
    target: torch.Tensor,
    num_variables: int,
) -> float:
    crps_values = []
    for j in range(num_variables):
        crps_pj = memory_efficient_crps(
            ensemble_output[:, :, j].contiguous(),
            target[:, j].contiguous(),
        ).mean()
        crps_values.append(crps_pj)
    return torch.stack(crps_values).mean().item()


def batched_forward_proposals(
    *,
    model: torch.nn.Module,
    previous_fields: torch.Tensor,
    current_fields: torch.Tensor,
    time_normalised: torch.Tensor,
    reference_tensor: torch.Tensor,
    alpha_matrix: torch.Tensor,
    ensemble_size: int,
    device: torch.device,
    buffers: dict,
    batch_manager: DynamicBatchManager,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[float]]:
    N, C, H, W = previous_fields.shape
    V = current_fields.shape[1]
    P = alpha_matrix.shape[0]

    current_slice = previous_fields[:, :V]
    past_slice = previous_fields[:, V : 2 * V]
    static_slice = previous_fields[:, -2:]

    perturb = generate_joint_rfp(
        reference_tensor,
        alpha_matrix,
        batch_size=N,
        ensemble_size=ensemble_size,
        device=device,
        generator=generator,
    )

    curr_base = current_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    past_base = past_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    stat_base = static_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)

    def test_batch_size(test_batch_size: int):
        test_n = min(N, 2)
        test_input = torch.cat([curr_base[:test_n], past_base[:test_n], stat_base[:test_n]], dim=2)
        test_input = test_input.view(test_n * ensemble_size, C, H, W)
        test_time = time_normalised[:test_n].view(-1, 1).expand(-1, ensemble_size).reshape(-1, 1)
        step = test_batch_size * ensemble_size
        with torch.no_grad():
            test_end = min(step, test_n * ensemble_size)
            _ = model(test_input[:test_end], test_time[:test_end])
        return test_batch_size

    if not hasattr(batch_manager, "_calibrated_for_this_config"):
        print(f"[DynamicBatch] Calibrating for N={N}, ensemble_size={ensemble_size}")
        initial_guess = batch_manager.predict_safe_batch_size(ensemble_size)
        optimal_batch_size = batch_manager.find_optimal_batch_size(test_batch_size, initial_guess)
        batch_manager._calibrated_for_this_config = True
    else:
        optimal_batch_size = batch_manager.current_batch_size

    joint_scores: list[float] = []
    best_proposal_output: torch.Tensor | None = None
    best_score = float("inf")

    for p in range(P):
        buffers["curr"][:N].copy_(curr_base)
        buffers["curr"][:N].add_(perturb[p])
        buffers["past"][:N].copy_(past_base)
        buffers["stat"][:N].copy_(stat_base)

        full_input = torch.cat([buffers["curr"][:N], buffers["past"][:N], buffers["stat"][:N]], dim=2)
        full_input = full_input.view(N * ensemble_size, C, H, W)
        full_time = time_normalised.view(-1, 1).expand(-1, ensemble_size).reshape(-1, 1)

        step = optimal_batch_size * ensemble_size
        out_chunks = []
        start = 0
        while start < N * ensemble_size:
            end = min(start + step, N * ensemble_size)
            try:
                with torch.no_grad():
                    y = model(full_input[start:end], full_time[start:end])
                out_chunks.append(y)
                start = end
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    optimal_batch_size = max(1, optimal_batch_size // 2)
                    step = optimal_batch_size * ensemble_size
                    batch_manager.current_batch_size = optimal_batch_size
                    print(f"[DynamicBatch] OOM encountered, reducing batch size to {optimal_batch_size}")
                    end = min(start + step, N * ensemble_size)
                    with torch.no_grad():
                        y = model(full_input[start:end], full_time[start:end])
                    out_chunks.append(y)
                    start = end
                else:
                    raise e

        y_full = torch.cat(out_chunks, dim=0).view(N, ensemble_size, V, H, W)
        proposal_output = y_full.permute(1, 0, 2, 3, 4)

        joint_score = compute_crps_for_proposal(proposal_output, current_fields, V)
        joint_scores.append(joint_score)

        if joint_score < best_score:
            best_score = joint_score
            if best_proposal_output is not None:
                del best_proposal_output
                torch.cuda.empty_cache()
            best_proposal_output = proposal_output.clone()

        del proposal_output, y_full, out_chunks
        torch.cuda.empty_cache()

    return best_proposal_output, joint_scores


def run_gibbs_abc_rfp(
    *,
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ensemble_size: int,
    n_steps: int,
    n_proposals: int,
    num_variables: int,
    variable_names: list[str],
    max_horizon: int,
    reference_mmap: np.memmap,
    result_directory: str,
    log_diagnostics: bool = True,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ref_full = torch.from_numpy(np.array(reference_mmap, copy=True)).to(device)

    prev_all = torch.cat([b[0] for b in batches], dim=0).to(device)
    curr_all = torch.cat([b[1] for b in batches], dim=0).to(device)
    time_all = torch.cat([b[2] for b in batches], dim=0).to(device)
    N, C, H, W = prev_all.shape
    V = curr_all.shape[1]

    buffers = {
        "curr": torch.empty((N, ensemble_size, V, H, W), device=device),
        "past": torch.empty((N, ensemble_size, V, H, W), device=device),
        "stat": torch.empty((N, ensemble_size, 2, H, W), device=device),
    }

    initial_batch_size = compute_safe_batch_size(
        ensemble_size=ensemble_size,
        num_variables=C,
        spatial_height=H,
        spatial_width=W,
        num_outputs=V,
    )
    batch_manager = DynamicBatchManager(device, initial_batch_size=initial_batch_size)
    print(f"[DynamicBatch] Initial conservative batch size: {initial_batch_size}")

    with torch.no_grad():
        _ = model(prev_all[:1], time_all[:1])

    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_crps = np.zeros((n_steps, num_variables), dtype=np.float32)
    step_mean_crps = np.zeros(n_steps, dtype=np.float32)

    rank_histograms = [[] for _ in range(num_variables)]
    ensemble_spread_records = [[] for _ in range(num_variables)]
    mean_absolute_error_records = [[] for _ in range(num_variables)]

    current_alpha = np.random.uniform(*INITIAL_ALPHA_RANGE, size=(num_variables, 1))
    proposal_std = np.full((num_variables, 1), PROPOSAL_SCALE, dtype=np.float32)

    rng = np.random.default_rng()
    torch_gen = torch.Generator(device=device)

    ckpt_path = os.path.join(result_directory, CHECKPOINT_FILE)
    start_step = 0
    if os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        print(f"[checkpoint] Resuming from step {ck['step']+1}")
        posterior_samples[: ck["step"] + 1] = ck["posterior_samples"]
        posterior_crps[: ck["step"] + 1] = ck["posterior_crps"]
        step_mean_crps[: ck["step"] + 1] = ck["step_mean_crps"]
        current_alpha = ck["last_alpha"]
        start_step = int(ck["step"]) + 1
        del ck

    for s in tqdm(range(start_step, n_steps), desc="Gibbs", position=0):
        print(f"\n[Gibbs step {s+1}/{n_steps}]")
        if s and (s % ADAPT_EVERY == 0):
            proposal_std *= ADAPT_FACTOR
            print(f"[adapt] proposal σ -> {proposal_std.mean():.3f}")

        for v in range(num_variables):
            proposals_v = np.clip(
                rng.normal(loc=current_alpha[v], scale=proposal_std[v], size=(n_proposals, 1)),
                MIN_ALPHA,
                None,
            )
            alpha_mat = np.repeat(current_alpha.squeeze(-1)[None, :], n_proposals, axis=0)
            alpha_mat[:, v] = proposals_v.squeeze(-1)
            alpha_tensor = torch.tensor(alpha_mat, device=device, dtype=torch.float32)

            allocated_before, _, total_mem, _ = batch_manager.get_memory_stats()
            print(
                f" {variable_names[v]:5s} | Mem: {allocated_before/1e9:.1f}/{total_mem/1e9:.1f}GB, "
                f"Batch: {batch_manager.current_batch_size}"
            )

            torch_gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
            best_ensemble, joint_scores = batched_forward_proposals(
                model=model,
                previous_fields=prev_all,
                current_fields=curr_all,
                time_normalised=time_all,
                reference_tensor=ref_full,
                alpha_matrix=alpha_tensor,
                ensemble_size=ensemble_size,
                device=device,
                buffers=buffers,
                batch_manager=batch_manager,
                generator=torch_gen,
            )

            joint_scores = np.array(joint_scores)
            best_idx = int(joint_scores.argmin())
            current_alpha[v] = proposals_v[best_idx]
            posterior_samples[s, v] = current_alpha[v]
            posterior_crps[s, v] = joint_scores[best_idx]

            print(f" {variable_names[v]:5s} α*={current_alpha[v,0]:.3f}  jointCRPS={joint_scores[best_idx]: .4f}")

            if log_diagnostics:
                for j in range(num_variables):
                    if j == v:
                        spread_val = compute_ensemble_spread(best_ensemble[:, :, j].cpu())
                        mae_val = compute_mean_absolute_error(best_ensemble[:, :, j].cpu(), curr_all[:, j].cpu())
                        ensemble_spread_records[j].append(spread_val)
                        mean_absolute_error_records[j].append(mae_val)
                        ranks = compute_rank_histogram(best_ensemble[:, :, j], curr_all[:, j], ensemble_size)
                        rank_histograms[j].extend(ranks.tolist())

            del best_ensemble
            torch.cuda.empty_cache()

        step_mean_crps[s] = posterior_crps[s].mean()
        print(f"⇒ mean joint CRPS (all vars) = {step_mean_crps[s]:.4f}")

        np.savez_compressed(
            ckpt_path,
            step=s,
            posterior_samples=posterior_samples,
            posterior_crps=posterior_crps,
            step_mean_crps=step_mean_crps,
            last_alpha=current_alpha,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {
        "posterior_samples": posterior_samples,
        "posterior_crps": posterior_crps,
        "posterior_mean": posterior_samples.mean(axis=0),
        "posterior_variance": posterior_samples.var(axis=0),
        "rank_histograms": rank_histograms,
        "ensemble_mae": np.array(mean_absolute_error_records, dtype=np.float32),
        "ensemble_spread": np.array(ensemble_spread_records, dtype=np.float32),
        "step_mean_crps": step_mean_crps,
    }
