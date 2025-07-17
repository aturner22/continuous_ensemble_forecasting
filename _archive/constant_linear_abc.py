from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import random
import time
from pathlib import Path
from prepare_model import prepare_model_and_loader

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
SAMPLE_SIZE = 60
N_GIBBS_STEPS = 100
N_CANDIDATES = 40

data_directory = './data'
result_directory = './results/global_gibbs_abc'
model_directory = './models'

VARIABLE_NAMES = ['z500', 't850', 't2m', 'u10', 'v10']
NUM_VARIABLES = len(VARIABLE_NAMES)
NUM_STATIC_FIELDS = 2
MAX_HORIZON = 240
param_dim = 3
param_labels = ['alpha_bias', 'beta_bias', 'alpha_scale', 'beta_scale']

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
    low=[-0.2, -0.1, 0.01, 0.0],
    high=[0.2, 0.1, 0.5, 0.2],
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

print("Starting inference...")

with torch.inference_mode():
    for t in tqdm(range(N_GIBBS_STEPS), desc='Gibbs Sampling'):
        cached_batches = load_batches_from_loader()

        for v in tqdm(range(NUM_VARIABLES), desc='Variables', leave=False):
            candidate_params = np.random.normal(
                loc=parameter_matrix[v],
                scale=[0.02] * param_dim,
                size=(N_CANDIDATES, param_dim)
            )

            best_candidate = parameter_matrix[v]
            best_mae = float('inf')

            for c_idx, (alpha_b, beta_b, alpha_s, beta_s) in enumerate(candidate_params):
                maes = []

                for previous, current, time_tensor in cached_batches:
                    var_channels = previous[:, :-NUM_STATIC_FIELDS, :, :]
                    static_channels = previous[:, -NUM_STATIC_FIELDS:, :, :]
                    base_field = var_channels[:, v, :, :]

                    bias_field = alpha_b + beta_b * base_field
                    scale_field = alpha_s + beta_s * base_field
                    noise = torch.randn_like(base_field)
                    perturbed_field = base_field + (noise + bias_field) * scale_field

                    perturbed_input = var_channels.clone()
                    perturbed_input[:, v, :, :] = perturbed_field
                    full_input = torch.cat([perturbed_input, static_channels], dim=1)

                    t0 = time.time()
                    output = model(full_input, time_tensor)

                    mae = torch.mean(torch.abs(output - current)).item()
                    maes.append(mae)

                mean_mae = np.mean(maes)


                if mean_mae < best_mae:
                    best_mae = mean_mae
                    best_candidate = np.array([alpha_b, beta_b, alpha_s, beta_s])

            parameter_matrix[v] = best_candidate
            parameter_samples[t, v] = best_candidate
            final_mae[t, v] = best_mae

np.save(result_path / "parameter_samples.npy", parameter_samples)
np.save(result_path / "final_mae.npy", final_mae)

posterior_mean = np.mean(parameter_samples, axis=0)
posterior_var = np.var(parameter_samples, axis=0)

np.save(result_path / "posterior_mean.npy", posterior_mean)
np.save(result_path / "posterior_var.npy", posterior_var)

print("\nPosterior Diagonal Gaussian Estimates per Variable")
print("--------------------------------------------------")
for v in range(NUM_VARIABLES):
    μ = posterior_mean[v]
    σ2 = posterior_var[v]
    print(f"{VARIABLE_NAMES[v]}:")
    for i in range(param_dim):
        print(f"   {param_labels[i]}: μ = {μ[i]:+.4f}, σ² = {σ2[i]:.4e}")

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