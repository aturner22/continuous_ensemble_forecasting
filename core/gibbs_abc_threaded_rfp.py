import torch
import numpy as np
from typing import Any
from tqdm import tqdm
from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)
import gc


def generate_batched_ensemble(
    *,
    model: torch.nn.Module,
    variable_fields: torch.Tensor,
    static_fields: torch.Tensor,
    base_tensor: torch.Tensor,
    time_normalised: torch.Tensor,
    reference_tensor: torch.Tensor,
    variable_index: int,
    alpha: float,
    ensemble_size: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = variable_fields.shape[0]
    tensor_shape = base_tensor.shape[1:]  # For broadcasting spatial dims if present

    idx_pairs = np.random.randint(0, reference_tensor.shape[0], size=(ensemble_size, 2))
    deltas = reference_tensor[idx_pairs[:, 0], variable_index] - reference_tensor[idx_pairs[:, 1], variable_index]
    deltas = torch.tensor(deltas, dtype=base_tensor.dtype, device=device).view(ensemble_size, *tensor_shape)
    perturbations = alpha * deltas

    # Expand base tensor to ensemble shape
    base_tensor_expanded = base_tensor.unsqueeze(0).expand(ensemble_size, -1, *tensor_shape)
    perturbed_tensor = base_tensor_expanded + perturbations

    # Repeat other inputs
    if variable_fields.dim() == 3:
        repeated_variable_fields = variable_fields.unsqueeze(0).repeat(ensemble_size, 1, 1)
        repeated_static_fields = static_fields.unsqueeze(0).repeat(ensemble_size, 1, 1)
    elif variable_fields.dim() == 4:
        repeated_variable_fields = variable_fields.unsqueeze(0).repeat(ensemble_size, 1, 1, 1)
        repeated_static_fields = static_fields.unsqueeze(0).repeat(ensemble_size, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported tensor shape for variable_fields: {variable_fields.shape}")

    # Replace variable index
    index_tensor = torch.tensor([variable_index], device=device)
    if variable_fields.dim() == 3:
        repeated_variable_fields[:, :, variable_index] = perturbed_tensor
    else:
        repeated_variable_fields.index_copy_(2, index_tensor, perturbed_tensor.transpose(1, 2))

    # Concatenate input and evaluate
    input_tensor = torch.cat([repeated_variable_fields, repeated_static_fields], dim=2 if variable_fields.dim() == 3 else 1)

    with torch.no_grad():
        if time_normalised.dim() == 2:
            repeated_time = time_normalised.unsqueeze(0).repeat(ensemble_size, 1)
        else:
            repeated_time = time_normalised.unsqueeze(0).repeat(ensemble_size, 1, 1)

        output_tensor = model(input_tensor, repeated_time).detach()

    return output_tensor  # shape: [E, B, V, H, W] or [E, B, V, D]

def compute_crps_and_ranks(
    ensemble_tensor: torch.Tensor,
    ground_truth: torch.Tensor,
    variable_index: int,
    ensemble_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    crps = continuous_ranked_probability_score(
        ensemble_tensor[:, :, variable_index],
        ground_truth[:, variable_index],
    )
    ranks = compute_rank_histogram(
        ensemble_tensor[:, :, variable_index],
        ground_truth[:, variable_index],
        ensemble_size,
    )
    return crps, ranks


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
    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_crps = np.zeros((n_steps, num_variables), dtype=np.float32)
    rank_histograms = [[] for _ in range(num_variables)]
    ensemble_spread_records = [[] for _ in range(num_variables)]
    mean_absolute_error_records = [[] for _ in range(num_variables)]
    step_mean_crps = np.zeros(n_steps, dtype=np.float32)

    device = next(model.parameters()).device
    current_parameter_matrix = np.random.uniform(low=1.0, high=5.0, size=(num_variables, 1))

    for step_index in tqdm(range(n_steps), desc="Gibbs Sampling Steps"):
        print(f"\n[Gibbs Step {step_index + 1}/{n_steps}]")
        for variable_index in range(num_variables):
            variable_name = variable_names[variable_index]
            print(f"  Optimizing variable: {variable_name} [{variable_index + 1}/{num_variables}]")

            proposal_matrix = np.abs(np.random.normal(
                loc=current_parameter_matrix[variable_index],
                scale=0.5,
                size=(n_proposals, 1),
            ))

            best_crps_value = float("inf")
            best_parameter_vector = current_parameter_matrix[variable_index].copy()
            best_members = None
            best_targets = None
            best_ranks = None

            for proposal_index in range(n_proposals):
                alpha = proposal_matrix[proposal_index][0]
                print(f"    Proposal {proposal_index + 1}/{n_proposals} (alpha = {alpha:.4f})")

                crps_values = []
                rank_arrays = []
                members_buffer = [] if log_diagnostics else None
                targets_buffer = [] if log_diagnostics else None

                for batch_idx, (previous_fields, current_fields, time_normalised) in enumerate(batches):
                    variable_fields = previous_fields[:, :-2].to(device)
                    static_fields = previous_fields[:, -2:].to(device)
                    base_tensor = variable_fields[:, variable_index]

                    ensemble_tensor = generate_batched_ensemble(
                        model=model,
                        variable_fields=variable_fields,
                        static_fields=static_fields,
                        base_tensor=base_tensor,
                        time_normalised=time_normalised,
                        reference_tensor=reference_tensor,
                        variable_index=variable_index,
                        alpha=alpha,
                        ensemble_size=ensemble_size,
                        device=device,
                    )

                    crps, ranks = compute_crps_and_ranks(
                        ensemble_tensor,
                        current_fields.to(device),
                        variable_index,
                        ensemble_size
                    )

                    crps_values.append(crps)
                    rank_arrays.append(ranks)

                    if log_diagnostics:
                        members_buffer.append(ensemble_tensor[:, :, variable_index].clone())
                        targets_buffer.append(current_fields[:, variable_index].clone())

                    del ensemble_tensor, variable_fields, static_fields, base_tensor
                    gc.collect()

                crps_mean = torch.mean(torch.stack(crps_values)).item()
                print(f"      Mean CRPS: {crps_mean:.6f}")

                if crps_mean < best_crps_value:
                    best_crps_value = crps_mean
                    best_parameter_vector = proposal_matrix[proposal_index]
                    best_members = members_buffer
                    best_targets = targets_buffer
                    best_ranks = np.concatenate(rank_arrays)

                del crps_values, rank_arrays, members_buffer, targets_buffer
                gc.collect()

            posterior_samples[step_index, variable_index] = best_parameter_vector
            posterior_crps[step_index, variable_index] = best_crps_value
            rank_histograms[variable_index].extend(best_ranks)

            if log_diagnostics and best_members is not None:
                spread_value = np.mean([
                    compute_ensemble_spread(m) for m in best_members
                ])
                error_value = np.mean([
                    compute_mean_absolute_error(m, t)
                    for m, t in zip(best_members, best_targets)
                ])
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
