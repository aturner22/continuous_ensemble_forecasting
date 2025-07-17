import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from prepare_model import prepare_model_and_loader

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SAMPLE_SIZE: int = 10
ENSEMBLE_SIZE: int = 5
N_GIBBS_STEPS: int = 30
N_PROPOSALS_PER_VARIABLE: int = 5
VARIABLE_NAMES = ["z500", "t850", "t2m", "u10", "v10"]
NUM_VARIABLES: int = len(VARIABLE_NAMES)
NUM_STATIC_FIELDS: int = 2
MAX_HORIZON: int = 240
PARAMETER_DIMENSION: int = 4
PARAMETER_LABELS = ["alpha_bias", "beta_bias", "alpha_scale", "beta_scale"]

DATA_DIRECTORY = Path("./data")
MODEL_DIRECTORY = Path("./models")
RESULT_DIRECTORY = Path("./results/calibration_abc_crps")
RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def continuous_ranked_probability_score(
    ensemble_tensor: torch.Tensor,  # shape (M, H, W)
    target_field: torch.Tensor      # shape (H, W)
) -> torch.Tensor:
    """Strictly proper integral score: CRPS(F, y).
    Averaged over the spatial grid. Complexity O(M^2)."""

    absolute_error_term = torch.mean(torch.abs(ensemble_tensor - target_field.unsqueeze(0)))

    pairwise_differences = torch.abs(
        ensemble_tensor.unsqueeze(0) - ensemble_tensor.unsqueeze(1)
    )
    second_moment_term = 0.5 * torch.mean(pairwise_differences)

    return absolute_error_term - second_moment_term


def compute_rank_histogram(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    ranks = (predictions < targets.unsqueeze(0)).sum(dim=0).flatten().cpu().numpy()
    return ranks / ENSEMBLE_SIZE

def compute_mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    mean_prediction = predictions.mean(dim=0)
    return torch.mean(torch.abs(mean_prediction - targets)).item()

def compute_ensemble_spread(predictions: torch.Tensor) -> float:
    return torch.std(predictions, dim=0).mean().item()



loader, model, latitude, longitude, _ = prepare_model_and_loader(
    device=device,
    data_directory=str(DATA_DIRECTORY),
    result_directory=str(RESULT_DIRECTORY),
    model_directory=str(MODEL_DIRECTORY),
    variable_names=VARIABLE_NAMES,
    num_variables=NUM_VARIABLES,
    num_static_fields=NUM_STATIC_FIELDS,
    max_horizon=MAX_HORIZON,
    random_subset_size=SAMPLE_SIZE,
    random_subset_seed=777,
)
model.eval()

# -------------------------
#  7. Storage tensors
# -------------------------

posterior_samples = np.zeros((N_GIBBS_STEPS, NUM_VARIABLES, PARAMETER_DIMENSION))
posterior_crps = np.zeros((N_GIBBS_STEPS, NUM_VARIABLES))

rank_histograms = [[] for _ in range(NUM_VARIABLES)]
mean_absolute_error_records = [[] for _ in range(NUM_VARIABLES)]
ensemble_spread_records = [[] for _ in range(NUM_VARIABLES)]

# -------------------------
#  8. Helper to materialise all batches in memory for one Gibbs step
# -------------------------

def materialise_batches():
    batches = []
    for previous_fields, current_fields, valid_time in loader:
        previous_fields = previous_fields.to(device)
        current_fields = current_fields.view(-1, NUM_VARIABLES, len(latitude), len(longitude)).to(device)
        time_normalised = torch.tensor([valid_time[0]], dtype=torch.float32, device=device) / MAX_HORIZON
        batches.append((previous_fields, current_fields, time_normalised))
    return batches

# -------------------------
#  9. Plotting routines
# -------------------------

def produce_trace_and_histogram_plots(sample_tensor: np.ndarray, output_directory: Path):
    for parameter_index in range(PARAMETER_DIMENSION):
        # Trace plot
        plt.figure(figsize=(10, 6))
        for variable_index in range(NUM_VARIABLES):
            plt.plot(sample_tensor[:, variable_index, parameter_index], label=VARIABLE_NAMES[variable_index])
        plt.title(f"Trace – {PARAMETER_LABELS[parameter_index]}")
        plt.xlabel("Gibbs iteration")
        plt.ylabel(PARAMETER_LABELS[parameter_index])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_directory / f"trace_{PARAMETER_LABELS[parameter_index]}.png")
        plt.close()

        # Posterior histogram
        plt.figure(figsize=(12, 6))
        for variable_index in range(NUM_VARIABLES):
            plt.hist(
                sample_tensor[:, variable_index, parameter_index],
                bins=30,
                alpha=0.6,
                label=VARIABLE_NAMES[variable_index],
                density=True,
            )
        plt.title(f"Posterior – {PARAMETER_LABELS[parameter_index]}")
        plt.xlabel(PARAMETER_LABELS[parameter_index])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_directory / f"hist_{PARAMETER_LABELS[parameter_index]}.png")
        plt.close()


def produce_rank_histograms(histogram_list, output_directory: Path):
    for variable_index in range(NUM_VARIABLES):
        plt.figure(figsize=(8, 4))
        plt.hist(
            histogram_list[variable_index],
            bins=ENSEMBLE_SIZE + 1,
            range=(0, 1),
            density=True,
            edgecolor="black",
        )
        plt.title(f"Rank histogram – {VARIABLE_NAMES[variable_index]}")
        plt.xlabel("Normalised rank")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(output_directory / f"rank_hist_{VARIABLE_NAMES[variable_index]}.png")
        plt.close()

# -------------------------
# 10. Gibbs–ABC loop
# -------------------------

print("Commencing Gibbs–ABC with CRPS summary statistic…")

with torch.inference_mode():
    # Initial parameter matrix: NUM_VARIABLES × PARAMETER_DIMENSION
    current_parameter_matrix = np.random.uniform(
        low=[-0.2, -0.1, -1.0, -1.0],
        high=[0.2, 0.1, 1.0, 1.0],
        size=(NUM_VARIABLES, PARAMETER_DIMENSION),
    )

    for gibbs_iteration_index in tqdm(range(N_GIBBS_STEPS), desc="Gibbs iteration"):
        print(f"\nGibbs iteration {gibbs_iteration_index + 1} / {N_GIBBS_STEPS}")

        # Materialise batches once per Gibbs iteration
        cached_batches = materialise_batches()

        for variable_index in range(NUM_VARIABLES):
            print(f"  Variable {VARIABLE_NAMES[variable_index]}…")

            proposal_matrix = np.random.normal(
                loc=current_parameter_matrix[variable_index],
                scale=[0.05] * PARAMETER_DIMENSION,
                size=(N_PROPOSALS_PER_VARIABLE, PARAMETER_DIMENSION),
            )

            best_crps_value = float("inf")
            best_parameter_vector = current_parameter_matrix[variable_index].copy()

            best_members_for_diagnostics = None
            best_targets_for_diagnostics = None
            best_rank_values = None

            # — iterate over candidate proposals
            for proposal_index in range(N_PROPOSALS_PER_VARIABLE):
                alpha_bias, beta_bias, alpha_scale, beta_scale = proposal_matrix[proposal_index]

                crps_values_per_batch = []
                batch_rank_arrays = []
                batch_member_tensors = []
                batch_target_tensors = []

                # propagate through all cached batches
                for previous_fields, current_fields, time_normalised in cached_batches:
                    variable_subset = previous_fields[:, : -NUM_STATIC_FIELDS, :, :]
                    static_subset = previous_fields[:, -NUM_STATIC_FIELDS :, :, :]
                    base_field = variable_subset[:, variable_index, :, :]

                    # ensemble generation
                    member_tensors = []
                    multiplicative_scale = torch.exp(alpha_scale + beta_scale * base_field)
                    additive_bias = alpha_bias + beta_bias * base_field

                    for _ in range(ENSEMBLE_SIZE):
                        perturbation = torch.randn_like(base_field)
                        perturbed_field = base_field + (perturbation + additive_bias) * multiplicative_scale
                        perturbed_variable_subset = variable_subset.clone()
                        perturbed_variable_subset[:, variable_index, :, :] = perturbed_field
                        full_input_tensor = torch.cat([perturbed_variable_subset, static_subset], dim=1)
                        model_output = model(full_input_tensor, time_normalised)
                        member_tensors.append(model_output)

                    ensemble_stack = torch.stack(member_tensors, dim=0)

                    # CRPS summary statistic
                    crps_values_per_batch.append(
                        continuous_ranked_probability_score(
                            ensemble_stack[:, :, variable_index],
                            current_fields[:, variable_index, :, :],
                        )
                    )

                    # Diagnostics
                    batch_rank_arrays.append(
                        compute_rank_histogram(
                            ensemble_stack[:, :, variable_index],
                            current_fields[:, variable_index, :, :],
                        )
                    )
                    batch_member_tensors.append(ensemble_stack[:, :, variable_index])
                    batch_target_tensors.append(current_fields[:, variable_index, :, :])

                aggregate_crps = torch.mean(torch.stack(crps_values_per_batch)).item()

                if aggregate_crps < best_crps_value:
                    best_crps_value = aggregate_crps
                    best_parameter_vector = proposal_matrix[proposal_index]
                    best_members_for_diagnostics = batch_member_tensors
                    best_targets_for_diagnostics = batch_target_tensors
                    best_rank_values = np.concatenate(batch_rank_arrays)

            # persist per-variable optimum
            current_parameter_matrix[variable_index] = best_parameter_vector
            posterior_samples[gibbs_iteration_index, variable_index] = best_parameter_vector
            posterior_crps[gibbs_iteration_index, variable_index] = best_crps_value

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

# -------------------------
# 11. Persist artefacts
# -------------------------

np.save(RESULT_DIRECTORY / "posterior_samples.npy", posterior_samples)
np.save(RESULT_DIRECTORY / "posterior_crps.npy", posterior_crps)

posterior_mean = posterior_samples.mean(axis=0)
posterior_variance = posterior_samples.var(axis=0)
np.save(RESULT_DIRECTORY / "posterior_mean.npy", posterior_mean)
np.save(RESULT_DIRECTORY / "posterior_variance.npy", posterior_variance)

produce_trace_and_histogram_plots(posterior_samples, RESULT_DIRECTORY)
produce_rank_histograms(rank_histograms, RESULT_DIRECTORY)

np.save(RESULT_DIRECTORY / "ensemble_mae.npy", np.array(mean_absolute_error_records))
np.save(RESULT_DIRECTORY / "ensemble_spread.npy", np.array(ensemble_spread_records))

# -------------------------
# 12. Final diagnostics
# -------------------------

print("\nPosterior parameter moments:")
print("---------------------------")
for variable_index in range(NUM_VARIABLES):
    print(f"{VARIABLE_NAMES[variable_index]}:")
    for parameter_index in range(PARAMETER_DIMENSION):
        mean_estimate = posterior_mean[variable_index, parameter_index]
        variance_estimate = posterior_variance[variable_index, parameter_index]
        print(
            f"  {PARAMETER_LABELS[parameter_index]}: μ = {mean_estimate:+.4f}, σ² = {variance_estimate:.4e}"
        )

print("\nInference procedure complete.")
