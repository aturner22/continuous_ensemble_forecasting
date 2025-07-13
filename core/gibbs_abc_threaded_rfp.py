import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)
from core.parallel_utils import parallel_map
import gc
import os

def generate_ensemble_member(
    model: Any,
    variable_fields: torch.Tensor,
    static_fields: torch.Tensor,
    base_tensor: torch.Tensor,
    time_normalised: torch.Tensor,
    reference_tensor: torch.Tensor,
    variable_index: int,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    idx1, idx2 = np.random.choice(reference_tensor.shape[0], size=2, replace=False)
    delta_field = reference_tensor[idx1, variable_index] - reference_tensor[idx2, variable_index]
    perturbation = alpha * delta_field.to(device)

    perturbed_tensor = base_tensor + perturbation
    input_tensor = torch.cat([
        variable_fields.clone().index_copy(1, torch.tensor([variable_index], device=device), perturbed_tensor.unsqueeze(1)),
        static_fields
    ], dim=1)

    with torch.no_grad():
        return model(input_tensor, time_normalised).detach()


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

    for step_index in range(n_steps):
        for variable_index in range(num_variables):
            proposal_matrix = np.abs(np.random.normal(
                loc=current_parameter_matrix[variable_index],
                scale=[0.5],
                size=(n_proposals, 1),
            ))

            best_crps_value = float("inf")
            best_parameter_vector = current_parameter_matrix[variable_index].copy()
            best_members = None
            best_targets = None
            best_ranks = None

            for proposal_index in range(n_proposals):
                alpha = proposal_matrix[proposal_index][0]

                crps_values = []
                rank_arrays = []
                members_buffer = [] if log_diagnostics else None
                targets_buffer = [] if log_diagnostics else None

                for previous_fields, current_fields, time_normalised in batches:
                    variable_fields = previous_fields[:, :-2].to(device)
                    static_fields = previous_fields[:, -2:].to(device)
                    base_tensor = variable_fields[:, variable_index]

                    args_list = [
                        (
                            model, variable_fields, static_fields,
                            base_tensor, time_normalised,
                            reference_tensor, variable_index,
                            alpha, device
                        ) for _ in range(ensemble_size)
                    ]

                    ensemble_members = parallel_map(lambda args: generate_ensemble_member(*args), args_list)
                    ensemble_tensor = torch.stack(ensemble_members, dim=0)

                    crps, ranks = compute_crps_and_ranks(
                        ensemble_tensor, current_fields.to(device), variable_index, ensemble_size
                    )

                    crps_values.append(crps)
                    rank_arrays.append(ranks)

                    if log_diagnostics:
                        members_buffer.append(ensemble_tensor[:, :, variable_index].clone())
                        targets_buffer.append(current_fields[:, variable_index].clone())

                    del ensemble_members, ensemble_tensor, variable_fields, static_fields, base_tensor
                    gc.collect()

                crps_mean = torch.mean(torch.stack(crps_values)).item()

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
        print(f"Time-averaged CRPS after Gibbs step {step_index + 1:02d}: {step_mean_crps[step_index]:.6f}")

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
