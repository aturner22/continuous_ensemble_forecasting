import os
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)
from core.parallel_utils import N_WORKERS

global_shared_model = None
global_shared_batches = None
global_reference_era5_tensor = None


def set_shared_objects(model, batches, reference_tensor):
    global global_shared_model, global_shared_batches, global_reference_era5_tensor
    global_shared_model = model
    global_shared_batches = batches
    global_reference_era5_tensor = reference_tensor


def evaluate_single_proposal_rfp(
    proposal_index: int,
    proposal_matrix: np.ndarray,
    variable_index: int,
    ensemble_size: int,
) -> tuple[float, np.ndarray, list[torch.Tensor], list[torch.Tensor], np.ndarray]:
    alpha = proposal_matrix[proposal_index][0]
    num_reference_samples = global_reference_era5_tensor.shape[0]

    batch_crps_list = []
    rank_array_list = []
    member_tensor_list = []
    target_tensor_list = []

    for previous_fields, current_fields, time_normalised in global_shared_batches:
        variable_subset = previous_fields[:, :-2]
        static_subset = previous_fields[:, -2:]
        base_field = variable_subset[:, variable_index]

        member_accumulator = []
        for _ in range(ensemble_size):
            τ1, τ2 = np.random.choice(num_reference_samples, size=2, replace=False)
            reference_diff = global_reference_era5_tensor[τ1, variable_index] - global_reference_era5_tensor[τ2, variable_index]
            perturbation = alpha * reference_diff.to(base_field.device)
            perturbed_field = base_field + perturbation

            modified_variable_subset = variable_subset.clone()
            modified_variable_subset[:, variable_index] = perturbed_field
            model_input_tensor = torch.cat([modified_variable_subset, static_subset], dim=1)
            model_output_tensor = global_shared_model(model_input_tensor, time_normalised)
            member_accumulator.append(model_output_tensor)

        ensemble_tensor = torch.stack(member_accumulator, dim=0)

        batch_crps_list.append(
            continuous_ranked_probability_score(
                ensemble_tensor[:, :, variable_index],
                current_fields[:, variable_index],
            )
        )
        rank_array_list.append(
            compute_rank_histogram(
                ensemble_tensor[:, :, variable_index],
                current_fields[:, variable_index],
                ensemble_size,
            )
        )
        member_tensor_list.append(ensemble_tensor[:, :, variable_index])
        target_tensor_list.append(current_fields[:, variable_index])

    aggregate_crps = torch.mean(torch.stack(batch_crps_list)).item()
    return (
        aggregate_crps,
        proposal_matrix[proposal_index],
        member_tensor_list,
        target_tensor_list,
        np.concatenate(rank_array_list),
    )


def run_gibbs_abc_rfp(
    *,
    model,
    batches,
    ensemble_size: int,
    n_steps: int,
    n_proposals: int,
    num_variables: int,
    variable_names,
    max_horizon: int,
    reference_tensor: torch.Tensor,
):
    set_shared_objects(model, batches, reference_tensor)

    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_crps = np.zeros((n_steps, num_variables), dtype=np.float32)

    rank_histograms: list[list[float]] = [[] for _ in range(num_variables)]
    ensemble_spread_records: list[list[float]] = [[] for _ in range(num_variables)]
    mean_absolute_error_records: list[list[float]] = [[] for _ in range(num_variables)]
    step_mean_crps = np.zeros(n_steps, dtype=np.float32)

    current_parameter_matrix = np.random.uniform(low=1.0, high=5.0, size=(num_variables, 1))

    with torch.inference_mode():
        for step_index in tqdm(range(n_steps), desc="Gibbs iteration"):
            for variable_index in range(num_variables):
                proposal_matrix = np.abs(np.random.normal(
                    loc=current_parameter_matrix[variable_index],
                    scale=[0.5],
                    size=(n_proposals, 1),
                ))

                futures = []
                with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                    for proposal_index in range(n_proposals):
                        futures.append(
                            executor.submit(
                                evaluate_single_proposal_rfp,
                                proposal_index,
                                proposal_matrix,
                                variable_index,
                                ensemble_size,
                            )
                        )

                    best_crps_value = float("inf")
                    best_parameter_vector = current_parameter_matrix[variable_index].copy()
                    best_members_for_diagnostics = None
                    best_targets_for_diagnostics = None
                    best_rank_values = None

                    for future in as_completed(futures):
                        (
                            aggregate_crps,
                            candidate_vector,
                            member_list,
                            target_list,
                            rank_vector,
                        ) = future.result()

                        if aggregate_crps < best_crps_value:
                            best_crps_value = aggregate_crps
                            best_parameter_vector = candidate_vector
                            best_members_for_diagnostics = member_list
                            best_targets_for_diagnostics = target_list
                            best_rank_values = rank_vector

                current_parameter_matrix[variable_index] = best_parameter_vector
                posterior_samples[step_index, variable_index] = best_parameter_vector
                posterior_crps[step_index, variable_index] = best_crps_value

                rank_histograms[variable_index].extend(best_rank_values)
                ensemble_spread_records[variable_index].append(
                    np.mean([compute_ensemble_spread(m) for m in best_members_for_diagnostics])
                )
                mean_absolute_error_records[variable_index].append(
                    np.mean([
                        compute_mean_absolute_error(m, t)
                        for m, t in zip(best_members_for_diagnostics, best_targets_for_diagnostics)
                    ])
                )

            step_mean_crps[step_index] = posterior_crps[step_index].mean()
            print(
                f"Time-averaged CRPS after Gibbs step {step_index + 1:02d}: "
                f"{step_mean_crps[step_index]:.6f}  "
                f"(evaluated with N_WORKERS={N_WORKERS})"
            )

    posterior_mean = posterior_samples.mean(axis=0)
    posterior_variance = posterior_samples.var(axis=0)

    return {
        "posterior_samples": posterior_samples,
        "posterior_crps": posterior_crps,
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance,
        "rank_histograms": rank_histograms,
        "ensemble_mae": np.array(mean_absolute_error_records, dtype=np.float32),
        "ensemble_spread": np.array(ensemble_spread_records, dtype=np.float32),
        "step_mean_crps": step_mean_crps,
    }
