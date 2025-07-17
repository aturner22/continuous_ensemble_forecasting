import torch
import numpy as np
import random
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from prepare_model import prepare_model_and_loader

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
SAMPLE_SIZE = 20
N_GIBBS_STEPS = 5
N_CANDIDATES = 100

data_directory = './data'
result_directory = './results/global_gibbs_abc_fixed_variance'
model_directory = './models'

VARIABLE_NAMES = ['z500', 't850', 't2m', 'u10', 'v10']
NUM_VARIABLES = len(VARIABLE_NAMES)
NUM_STATIC_FIELDS = 2
MAX_HORIZON = 240
param_dim = 2
param_labels = ['alpha_bias', 'beta_bias']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model and data loader...")
loader, model, lat, lon, result_path = prepare_model_and_loader(
    device=device,
    data_directory=data_directory,
    result_directory=result_directory,
    model_directory=model_directory,
    variable_names=VARIABLE_NAMES,
    num_variables=NUM_VARIABLES,
    num_static_fields=NUM_STATIC_FIELDS,
    max_horizon=MAX_HORIZON,
    random_subset_size=SAMPLE_SIZE,
    random_subset_seed=42
)
model.eval()

result_path = Path(result_path)
result_path.mkdir(parents=True, exist_ok=True)

parameter_samples = np.zeros((N_GIBBS_STEPS, NUM_VARIABLES, param_dim))
final_mae = np.zeros((N_GIBBS_STEPS, NUM_VARIABLES))

parameter_matrix = np.random.uniform(
    low=[-0.2, -0.1],
    high=[0.2, 0.1],
    size=(NUM_VARIABLES, param_dim)
)

def load_batches_from_loader():
    batches = []
    for previous, current, time_labels in loader:
        previous = previous.to(device)
        current = current.view(-1, NUM_VARIABLES, len(lat), len(lon)).to(device)
        time_tensor = torch.tensor([time_labels[0]], dtype=torch.float32).to(device) / MAX_HORIZON
        batches.append((previous, current, time_tensor))
    return batches

print("Starting Gibbs-ABC Inference...")

with torch.inference_mode():
    for t in tqdm(range(N_GIBBS_STEPS), desc='Gibbs Sampling'):
        cached_batches = load_batches_from_loader()

        for v in tqdm(range(NUM_VARIABLES), desc='Variables', leave=False):
            # Compute baseline MAE with unperturbed input
            baseline_maes = []
            for previous, current, time_tensor in cached_batches:
                output_baseline = model(previous, time_tensor)
                baseline_maes.append(torch.mean(torch.abs(output_baseline - current)).item())
            baseline_mae = np.mean(baseline_maes)

            candidate_params = np.random.normal(
                loc=parameter_matrix[v],
                scale=[0.02] * param_dim,
                size=(N_CANDIDATES, param_dim)
            )

            best_candidate = parameter_matrix[v]
            best_mae = baseline_mae

            for c_idx, (alpha_b, beta_b) in enumerate(candidate_params):
                maes = []
                for previous, current, time_tensor in cached_batches:
                    var_channels = previous[:, :-NUM_STATIC_FIELDS, :, :]
                    static_channels = previous[:, -NUM_STATIC_FIELDS:, :, :]
                    base_field = var_channels[:, v, :, :]

                    bias_field = alpha_b + beta_b * base_field
                    noise = torch.randn_like(base_field)
                    perturbed_field = base_field + (noise + bias_field)

                    perturbed_input = var_channels.clone()
                    perturbed_input[:, v, :, :] = perturbed_field
                    full_input = torch.cat([perturbed_input, static_channels], dim=1)

                    output = model(full_input, time_tensor)
                    mae = torch.mean(torch.abs(output - current)).item()
                    maes.append(mae)

                mean_mae = np.mean(maes)

                if mean_mae < best_mae:
                    best_mae = mean_mae
                    best_candidate = np.array([alpha_b, beta_b])

            parameter_matrix[v] = best_candidate
            parameter_samples[t, v] = best_candidate
            final_mae[t, v] = best_mae

# Save posterior samples and MAEs
np.save(result_path / "parameter_samples.npy", parameter_samples)
np.save(result_path / "final_mae.npy", final_mae)

posterior_mean = np.mean(parameter_samples, axis=0)
posterior_var = np.var(parameter_samples, axis=0)

np.save(result_path / "posterior_mean.npy", posterior_mean)
np.save(result_path / "posterior_var.npy", posterior_var)

# Logging
print("\nPosterior Diagonal Gaussian Estimates per Variable")
print("--------------------------------------------------")
for v in range(NUM_VARIABLES):
    μ = posterior_mean[v]
    σ2 = posterior_var[v]
    print(f"{VARIABLE_NAMES[v]}:")
    for i in range(param_dim):
        print(f"   {param_labels[i]}: μ = {μ[i]:+.4f}, σ² = {σ2[i]:.4e}")

# Trace plots
print("\nPlotting results...")
for v in range(NUM_VARIABLES):
    plt.figure(figsize=(12, 6))
    for i in range(param_dim):
        plt.plot(parameter_samples[:, v, i], label=param_labels[i])
    plt.title(f"Posterior Trace – {VARIABLE_NAMES[v]}")
    plt.xlabel("Gibbs Iteration")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path / f"trace_{VARIABLE_NAMES[v]}.png")
    plt.close()

# KDE plots
for i in range(param_dim):
    plt.figure(figsize=(14, 6))
    for v in range(NUM_VARIABLES):
        sns.kdeplot(parameter_samples[:, v, i], label=VARIABLE_NAMES[v])
    plt.title(f"Posterior KDE – {param_labels[i]}")
    plt.xlabel("Parameter Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path / f"kde_{param_labels[i]}.png")
    plt.close()

print("Inference complete.")
