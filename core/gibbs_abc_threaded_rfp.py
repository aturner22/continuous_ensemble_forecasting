import torch
import numpy as np
import psutil
import gc
from typing import Any
from tqdm import tqdm
from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)

PROPOSAL_SCALE = 0.05

def estimate_safe_chunk_size(
    num_input_channels: int,
    num_output_channels: int,
    height: int,
    width: int,
    available_memory_bytes: int = None,
    dtype_size_bytes: int = 4,
    model_overhead_factor: float = 3.0,
    safety_factor: float = 32.0,
) -> int:
    if available_memory_bytes is None:
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            available_memory_bytes = total - max(reserved, allocated)
        else:
            virtual_mem = psutil.virtual_memory()
            available_memory_bytes = virtual_mem.available
    spatial_size = height * width
    bytes_per_sample = (
        (num_input_channels + num_output_channels)
        * spatial_size * dtype_size_bytes * model_overhead_factor
    )
    return max(8, int(available_memory_bytes // (bytes_per_sample * safety_factor)))

def run_gibbs_abc_rfp(
    *,
    model: Any,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ensemble_size: int,
    n_steps: int,
    n_proposals: int,
    num_variables: int,
    variable_names: list[str],
    max_horizon: int,
    reference_tensor: torch.Tensor,
    log_diagnostics: bool = False,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    reference_tensor = reference_tensor.to(device)
    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_crps = np.zeros((n_steps, num_variables), dtype=np.float32)
    rank_histograms = [[] for _ in range(num_variables)]
    ensemble_spread_records = [[] for _ in range(num_variables)]
    mean_absolute_error_records = [[] for _ in range(num_variables)]
    step_mean_crps = np.zeros(n_steps, dtype=np.float32)

    full_previous = torch.cat([b[0] for b in batches], dim=0).to(device)
    full_current = torch.cat([b[1] for b in batches], dim=0).to(device)
    full_time = torch.cat([b[2] for b in batches], dim=0).to(device)

    C = full_previous.shape[1]
    V = full_current.shape[1]
    H, W = full_previous.shape[-2:]
    chunk_size = estimate_safe_chunk_size(C, V, H, W)
    print(f"[Chunked Forward Pass] Using chunk size = {chunk_size}")

    def generate_batched_ensemble(
        model: Any,
        previous_fields: torch.Tensor,
        current_fields: torch.Tensor,
        time_normalised: torch.Tensor,
        reference_tensor: torch.Tensor,
        variable_index: int,
        alpha: float,
        ensemble_size: int,
        device: torch.device,
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

        variable_fields = variable_fields.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1).clone()
        variable_fields[:, :, variable_index] = perturbed
        static_fields = static_fields.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
        model_inputs = torch.cat([variable_fields, static_fields], dim=2).reshape(N * ensemble_size, C, H, W)
        time_inputs = time_normalised.unsqueeze(1).expand(-1, ensemble_size).reshape(-1, 1)

        outputs = []
        for start in range(0, model_inputs.size(0), chunk_size):
            end = min(start + chunk_size, model_inputs.size(0))
            with torch.no_grad():
                out = model(model_inputs[start:end], time_inputs[start:end]).detach()
            outputs.append(out)

        output = torch.cat(outputs, dim=0)
        output = output.view(N, ensemble_size, -1, H, W).permute(1, 0, 2, 3, 4).contiguous()
        return output, current_fields[:, variable_index]

    current_parameter_matrix = np.random.uniform(low=1.0, high=5.0, size=(num_variables, 1))

    print("[Device Warm-up]...")
    dummy_input = torch.zeros((1, *batches[0][0].shape[1:]), device=device)
    dummy_time = batches[0][2][None].to(device)
    _ = model(dummy_input, dummy_time)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for step_index in tqdm(range(n_steps), desc="Gibbs Sampling Steps"):
        print(f"\n[Gibbs Step {step_index + 1}/{n_steps}]")

        for variable_index in range(num_variables):
            variable_name = variable_names[variable_index]
            print(f"  Optimizing variable: {variable_name} [{variable_index + 1}/{num_variables}]")

            proposal_matrix = np.random.normal(
                loc=current_parameter_matrix[variable_index],
                scale=PROPOSAL_SCALE,
                size=(n_proposals, 1),
            )

            best_crps_value = float("inf")
            best_parameter_vector = current_parameter_matrix[variable_index].copy()
            best_members = None
            best_targets = None
            best_ranks = None

            for proposal_index in range(n_proposals):
                alpha = proposal_matrix[proposal_index][0]
                print(f"    Proposal {proposal_index + 1}/{n_proposals} (alpha = {alpha:.4f})")

                ensemble_tensor, target_tensor = generate_batched_ensemble(
                    model=model,
                    previous_fields=full_previous,
                    current_fields=full_current,
                    time_normalised=full_time,
                    reference_tensor=reference_tensor,
                    variable_index=variable_index,
                    alpha=alpha,
                    ensemble_size=ensemble_size,
                    device=device,
                )

                crps = continuous_ranked_probability_score(
                    ensemble_tensor[:, :, variable_index], target_tensor
                )
                ranks = compute_rank_histogram(
                    ensemble_tensor[:, :, variable_index], target_tensor, ensemble_size
                )

                crps_mean = crps.mean().item()
                print(f"      Mean CRPS: {crps_mean:.6f}")

                if crps_mean < best_crps_value:
                    best_crps_value = crps_mean
                    best_parameter_vector = proposal_matrix[proposal_index]
                    best_ranks = ranks
                    if log_diagnostics:
                        best_members = ensemble_tensor[:, :, variable_index].clone()
                        best_targets = target_tensor.clone()

                del ensemble_tensor, target_tensor
                gc.collect()

            posterior_samples[step_index, variable_index] = best_parameter_vector
            posterior_crps[step_index, variable_index] = best_crps_value
            rank_histograms[variable_index].extend(best_ranks)

            if log_diagnostics and best_members is not None:
                spread_value = compute_ensemble_spread(best_members)
                error_value = compute_mean_absolute_error(best_members, best_targets)
                ensemble_spread_records[variable_index].append(spread_value)
                mean_absolute_error_records[variable_index].append(error_value)

            del best_members, best_targets, best_ranks, proposal_matrix
            gc.collect()

        step_mean_crps[step_index] = posterior_crps[step_index].mean()
        print(f"Completed step {step_index + 1}: mean CRPS = {step_mean_crps[step_index]:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
