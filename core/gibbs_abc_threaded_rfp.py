import torch
import numpy as np
import gc
import os
from typing import Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)


N_WORKERS = int(os.getenv("N_WORKERS", "16"))

def generate_batched_ensemble(
    model: Any,
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
    B, C, H, W = variable_fields.shape
    idx_pairs = np.random.randint(0, reference_tensor.shape[0], size=(ensemble_size, 2))

    delta_fields = reference_tensor[idx_pairs[:, 0], variable_index] - reference_tensor[idx_pairs[:, 1], variable_index]  # [E, H, W]
    perturbations = alpha * delta_fields.to(device)  # [E, H, W]

    base_tensor = base_tensor.squeeze(1)  # Ensure shape [B, H, W]

    # Ensemble: [E, B, H, W]
    perturbed_fields = base_tensor[None, :, :, :].expand(ensemble_size, -1, -1, -1) + perturbations[:, None, :, :]  # [E, B, H, W]

    # Insert into variable field
    repeated_variable_fields = variable_fields.unsqueeze(0).repeat(ensemble_size, 1, 1, 1, 1).contiguous()
    repeated_variable_fields[:, :, variable_index] = perturbed_fields  # Replace channel

    repeated_static_fields = static_fields.unsqueeze(0).repeat(ensemble_size, 1, 1, 1, 1)
    model_inputs = torch.cat([repeated_variable_fields, repeated_static_fields], dim=2)  # [E, B, C+2, H, W]

    time_repeated = time_normalised.unsqueeze(0).repeat(ensemble_size, 1)  # [E, T]
    time_repeated = time_repeated.view(ensemble_size, 1, -1).expand(-1, B, -1).reshape(ensemble_size * B, -1)

    # Merge ensemble and batch dimensions
    model_inputs = model_inputs.view(ensemble_size * B, model_inputs.size(2), H, W)

    with torch.no_grad():
        output = model(model_inputs, time_repeated).detach()  # [E*B, C, H, W]
    
    return output.view(ensemble_size, B, -1, H, W).to(device)  # [E, B, C, H, W]



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
    reference_tensor = reference_tensor.to(device)
    current_parameter_matrix = np.random.uniform(low=1.0, high=5.0, size=(num_variables, 1))

    print(f"[CUDA Warm-up]...")
    _ = model(torch.zeros((1, *batches[0][0].shape[1:]), device=device), batches[0][2][None].to(device))
    torch.cuda.synchronize()

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

                def evaluate_batch(batch):
                    try:
                        previous_fields, current_fields, time_normalised = batch
                        variable_fields = previous_fields[:, :-2].to(device)
                        static_fields = previous_fields[:, -2:].to(device)
                        base_tensor = variable_fields[:, variable_index]

                        ensemble_tensor = generate_batched_ensemble(
                            model, variable_fields, static_fields, base_tensor,
                            time_normalised.to(device), reference_tensor,
                            variable_index, alpha, ensemble_size, device
                        )

                        crps = continuous_ranked_probability_score(
                            ensemble_tensor[:, :, variable_index],
                            current_fields[:, variable_index].to(device)
                        )
                        ranks = compute_rank_histogram(
                            ensemble_tensor[:, :, variable_index],
                            current_fields[:, variable_index].to(device),
                            ensemble_size
                        )

                        if log_diagnostics:
                            return crps, ranks, ensemble_tensor[:, :, variable_index].clone(), current_fields[:, variable_index].clone()
                        return crps, ranks, None, None
                    except Exception as e:
                        print(f"[Thread Error] {e}")
                        raise

                with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                    results = list(pool.map(evaluate_batch, batches))

                for result in results:
                    crps, ranks, member, target = result
                    crps_values.append(crps)
                    rank_arrays.append(ranks)
                    if log_diagnostics:
                        members_buffer.append(member)
                        targets_buffer.append(target)

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
                spread_value = np.mean([compute_ensemble_spread(m) for m in best_members])
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
