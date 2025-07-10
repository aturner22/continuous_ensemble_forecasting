# core/gibbs_abc.py
# Gibbs–ABC engine with embarrassingly-parallel proposal evaluation.
# Cleaned for multiprocessing: all worker targets are top-level functions.

import numpy as np
import torch
from tqdm import tqdm
from functools import partial

from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)
from core.parallel_utils import parallel_map, N_WORKERS


# ────────────────────────────────────────────────────────────────────────────────
# Picklable, top-level worker ----------------------------------------------------

def evaluate_single_proposal(                        # noqa: C901
    proposal_index: int,
    proposal_matrix: np.ndarray,
    variable_index: int,
    ensemble_size: int,
    batches,
    model,
) -> tuple[float, np.ndarray, list[torch.Tensor], list[torch.Tensor], np.ndarray]:
    """
    CRPS-based fitness evaluation for one proposal vector.

    Returns
    -------
    aggregate_crps : float
    proposal_vector : np.ndarray
    member_list : list[Tensor]
    target_list : list[Tensor]
    rank_vector : np.ndarray
    """
    alpha_bias, beta_bias, alpha_scale, beta_scale = proposal_matrix[proposal_index]

    batch_crps_list: list[torch.Tensor] = []
    rank_array_list: list[np.ndarray] = []
    member_tensor_list: list[torch.Tensor] = []
    target_tensor_list: list[torch.Tensor] = []

    for previous_fields, current_fields, time_normalised in batches:
        variable_subset = previous_fields[:, :-2]
        static_subset = previous_fields[:, -2:]
        base_field = variable_subset[:, variable_index]

        multiplicative_scale = torch.exp(alpha_scale + beta_scale * base_field)
        additive_bias = alpha_bias + beta_bias * base_field

        member_accumulator = []
        for _ in range(ensemble_size):
            perturbation = torch.randn_like(base_field)
            perturbed_field = base_field + (perturbation + additive_bias) * multiplicative_scale
            modified_variable_subset = variable_subset.clone()
            modified_variable_subset[:, variable_index] = perturbed_field
            model_input_tensor = torch.cat([modified_variable_subset, static_subset], dim=1)
            model_output_tensor = model(model_input_tensor, time_normalised)
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


# ────────────────────────────────────────────────────────────────────────────────
# Main Gibbs–ABC routine --------------------------------------------------------

def run_gibbs_abc(
    *,
    model,
    batches,
    ensemble_size: int,
    n_steps: int,
    n_proposals: int,
    num_variables: int,
    parameter_dim: int,
    variable_names,
    max_horizon: int,  # retained for signature consistency
):
    """
    Gibbs–ABC inference with CRPS summary statistic.

    Parallelism: proposal vectors are evaluated concurrently across
    `N_WORKERS` processes or threads as configured in `parallel_utils.py`.
    """

    posterior_samples = np.zeros((n_steps, num_variables, parameter_dim), dtype=np.float32)
    posterior_crps = np.zeros((n_steps, num_variables), dtype=np.float32)

    rank_histograms: list[list[float]] = [[] for _ in range(num_variables)]
    ensemble_spread_records: list[list[float]] = [[] for _ in range(num_variables)]
    mean_absolute_error_records: list[list[float]] = [[] for _ in range(num_variables)]
    step_mean_crps = np.zeros(n_steps, dtype=np.float32)

    current_parameter_matrix = np.random.uniform(
        low=[-0.2, -0.1, -1.0, -1.0],
        high=[0.2, 0.1, 1.0, 1.0],
        size=(num_variables, parameter_dim),
    )

    # ────────────────────────────────────────────────────────────────────────────
    with torch.inference_mode():
        for step_index in tqdm(range(n_steps), desc="Gibbs iteration"):
            for variable_index in range(num_variables):

                proposal_matrix = np.random.normal(
                    loc=current_parameter_matrix[variable_index],
                    scale=[0.05] * parameter_dim,
                    size=(n_proposals, parameter_dim),
                )

                best_crps_value = float("inf")
                best_parameter_vector = current_parameter_matrix[variable_index].copy()
                best_members_for_diagnostics = None
                best_targets_for_diagnostics = None
                best_rank_values = None

                # ------------------------------------------------------------------
                # Parallel evaluation of all proposals
                # ------------------------------------------------------------------
                evaluator = partial(
                    evaluate_single_proposal,
                    proposal_matrix=proposal_matrix,
                    variable_index=variable_index,
                    ensemble_size=ensemble_size,
                    batches=batches,
                    model=model,
                )

                for (
                    aggregate_crps,
                    candidate_vector,
                    member_list,
                    target_list,
                    rank_vector,
                ) in parallel_map(evaluator, range(n_proposals)):

                    if aggregate_crps < best_crps_value:
                        best_crps_value = aggregate_crps
                        best_parameter_vector = candidate_vector
                        best_members_for_diagnostics = member_list
                        best_targets_for_diagnostics = target_list
                        best_rank_values = rank_vector

                # ------------------------------------------------------------------
                # Persist optimum for this variable
                # ------------------------------------------------------------------
                current_parameter_matrix[variable_index] = best_parameter_vector
                posterior_samples[step_index, variable_index] = best_parameter_vector
                posterior_crps[step_index, variable_index] = best_crps_value

                rank_histograms[variable_index].extend(best_rank_values)
                ensemble_spread_records[variable_index].append(
                    np.mean([compute_ensemble_spread(m) for m in best_members_for_diagnostics])
                )
                mean_absolute_error_records[variable_index].append(
                    np.mean(
                        [
                            compute_mean_absolute_error(m, t)
                            for m, t in zip(best_members_for_diagnostics, best_targets_for_diagnostics)
                        ]
                    )
                )

            # ----------------------------------------------------------------------
            # Global performance metric for this Gibbs step
            # ----------------------------------------------------------------------
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
