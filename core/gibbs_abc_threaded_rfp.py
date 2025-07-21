import torch
import numpy as np
import psutil
import os
from typing import Any
from tqdm import tqdm
from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)

PROPOSAL_SCALE = 0.05


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

    spatial_elements = spatial_height * spatial_width
    tensor_bytes = (
        (num_variables + 2) * spatial_elements +
        num_outputs * spatial_elements +
        num_variables * ensemble_size * spatial_elements * 3
    ) * dtype_bytes * model_overhead

    return max(1, int(available_bytes // (tensor_bytes * safety_divisor)))


def generate_batched_ensemble_from_tensor(
    *,
    model: torch.nn.Module,
    previous_fields: torch.Tensor,
    current_fields: torch.Tensor,
    time_normalised: torch.Tensor,
    reference_tensor: torch.Tensor,
    variable_index: int,
    alpha: float,
    ensemble_size: int,
    device: torch.device,
    buffer: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, C, H, W = previous_fields.shape
    T = reference_tensor.shape[0]

    variable_fields = previous_fields[:, :-2]
    static_fields = previous_fields[:, -2:]
    base_tensor = variable_fields[:, variable_index]

    idx1 = torch.randint(0, T, (N, ensemble_size), device=device)
    idx2 = torch.randint(0, T, (N, ensemble_size), device=device)

    delta = reference_tensor[idx1, variable_index] - reference_tensor[idx2, variable_index]
    perturbations = alpha * delta
    perturbed = base_tensor.unsqueeze(1) + perturbations

    buffer['var'][:N, :ensemble_size].copy_(
        variable_fields.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    )
    buffer['var'][:N, :ensemble_size, variable_index].copy_(perturbed)
    buffer['stat'][:N, :ensemble_size].copy_(
        static_fields.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    )

    full_input_tensor = torch.cat(
        [buffer['var'][:N, :ensemble_size], buffer['stat'][:N, :ensemble_size]], dim=2
    ).reshape(N * ensemble_size, C, H, W)
    full_time_tensor = time_normalised.unsqueeze(1).expand(-1, ensemble_size).reshape(-1, 1)

    outer_bs = compute_safe_batch_size(
        ensemble_size=ensemble_size,
        num_variables=C,
        spatial_height=H,
        spatial_width=W,
        num_outputs=current_fields.shape[1]
    )

    output_list = []
    for start in range(0, N, outer_bs):
        end = min(start + outer_bs, N)
        with torch.no_grad():
            output = model(
                full_input_tensor[start * ensemble_size:end * ensemble_size],
                full_time_tensor[start * ensemble_size:end * ensemble_size]
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        output_list.append(output)

    output_tensor = torch.cat(output_list, dim=0).view(N, ensemble_size, -1, H, W)
    return output_tensor.permute(1, 0, 2, 3, 4).contiguous(), current_fields[:, variable_index]


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
    log_diagnostics: bool = False,
) -> dict[str, Any]:
    checkpoint_path = os.path.join(result_directory, "gibbs_checkpoint_step.npz")
    resume_from_step = 0

    device = next(model.parameters()).device
    reference_tensor = torch.from_numpy(np.array(reference_mmap)).to(device)

    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_crps = np.zeros((n_steps, num_variables), dtype=np.float32)
    rank_histograms = [[] for _ in range(num_variables)]
    ensemble_spread_records = [[] for _ in range(num_variables)]
    mean_absolute_error_records = [[] for _ in range(num_variables)]
    step_mean_crps = np.zeros(n_steps, dtype=np.float32)
    current_parameter_matrix = np.random.uniform(1.0, 5.0, size=(num_variables, 1))

    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        posterior_samples[:ckpt["step"]+1] = ckpt["posterior_samples"]
        posterior_crps[:ckpt["step"]+1] = ckpt["posterior_crps"]
        step_mean_crps[:ckpt["step"]+1] = ckpt["step_mean_crps"]
        current_parameter_matrix = ckpt["last_params"]
        resume_from_step = int(ckpt["step"]) + 1
        del ckpt

    full_previous = torch.cat([b[0] for b in batches], dim=0).to(device)
    full_current = torch.cat([b[1] for b in batches], dim=0).to(device)
    full_time = torch.cat([b[2] for b in batches], dim=0).to(device)

    N, C, H, W = full_previous.shape
    V = full_current.shape[1]

    buffer = {
        'var': torch.empty((N, ensemble_size, C - 2, H, W), device=device),
        'stat': torch.empty((N, ensemble_size, 2, H, W), device=device),
    }
    chunk_size = compute_safe_batch_size(
        ensemble_size=ensemble_size,
        num_variables=C,
        spatial_height=H,
        spatial_width=W,
        num_outputs=V,
    )
    print("[Device Warm-up]...")
    dummy_input = torch.zeros((1, C, H, W), device=device)
    dummy_time = full_time[0:1]
    _ = model(dummy_input, dummy_time)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    for step_index in tqdm(range(resume_from_step, n_steps), desc="Gibbs Sampling Steps"):
        print(f"\n[Gibbs Step {step_index + 1}/{n_steps}]")
        for variable_index in range(num_variables):
            proposal_matrix = np.clip(
                np.random.normal(
                    loc=current_parameter_matrix[variable_index],
                    scale=PROPOSAL_SCALE,
                    size=(n_proposals, 1)
                ), 1e-4, None
            )

            best_crps_value = float("inf")
            best_parameter_vector = current_parameter_matrix[variable_index].copy()
            best_members = None
            best_targets = None
            best_ranks = None

            for proposal_index in range(n_proposals):
                alpha = proposal_matrix[proposal_index][0]
                print(f"    Proposal {proposal_index + 1}/{n_proposals} (alpha = {alpha:.4f})")

                crps_values = []
                ranks_agg = []
                members_list = []
                targets_list = []

                for batch_start in range(0, N, chunk_size):
                    batch_end = min(batch_start + chunk_size, N)

                    ensemble_tensor, target_tensor = generate_batched_ensemble_from_tensor(
                        model=model,
                        previous_fields=full_previous[batch_start:batch_end],
                        current_fields=full_current[batch_start:batch_end],
                        time_normalised=full_time[batch_start:batch_end],
                        reference_tensor=reference_tensor,
                        variable_index=variable_index,
                        alpha=alpha,
                        ensemble_size=ensemble_size,
                        device=device,
                        buffer=buffer,
                    )

                    crps = continuous_ranked_probability_score(
                        ensemble_tensor[:, :, variable_index], target_tensor
                    )
                    ranks = compute_rank_histogram(
                        ensemble_tensor[:, :, variable_index], target_tensor, ensemble_size
                    )

                    crps_values.append(crps.cpu())
                    ranks_agg.extend(ranks)

                    if log_diagnostics:
                        members_list.append(ensemble_tensor[:, :, variable_index].cpu())
                        targets_list.append(target_tensor.cpu())

                crps_mean = torch.cat(crps_values, dim=0).mean().item()
                print(f"      Mean CRPS: {crps_mean:.6f}")

                if crps_mean < best_crps_value:
                    best_crps_value = crps_mean
                    best_parameter_vector = proposal_matrix[proposal_index]
                    best_ranks = ranks_agg
                    if log_diagnostics:
                        best_members = torch.cat(members_list, dim=1)
                        best_targets = torch.cat(targets_list, dim=0)

            posterior_samples[step_index, variable_index] = best_parameter_vector
            posterior_crps[step_index, variable_index] = best_crps_value
            rank_histograms[variable_index].extend(best_ranks)

            if log_diagnostics and best_members is not None:
                spread_value = compute_ensemble_spread(best_members)
                error_value = compute_mean_absolute_error(best_members, best_targets)
                ensemble_spread_records[variable_index].append(spread_value)
                mean_absolute_error_records[variable_index].append(error_value)

        step_mean_crps[step_index] = posterior_crps[step_index].mean()
        print(f"Completed step {step_index + 1}: mean CRPS = {step_mean_crps[step_index]:.6f}")
        np.savez_compressed(
            checkpoint_path,
            step=step_index,
            posterior_samples=posterior_samples,
            posterior_crps=posterior_crps,
            step_mean_crps=step_mean_crps,
            last_params=current_parameter_matrix
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

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
