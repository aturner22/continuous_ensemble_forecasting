from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from pathlib import Path
from prepare_model import prepare_model_and_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_directory = './data'
result_directory = './results/gibbs_abc'
model_directory = './models'
VARIABLE_NAMES = ['z500', 't850', 't2m', 'u10', 'v10']
NUM_VARIABLES, NUM_STATIC_FIELDS = 5, 2
MAX_HORIZON = 240

n_gibbs_steps = 20
n_candidates = 30

param_labels = ['alpha_bias', 'beta_bias', 'alpha_scale', 'beta_scale']
param_dim = 4

loader, model, lat, lon, result_path = prepare_model_and_loader(
    device,
    data_directory,
    result_directory,
    model_directory,
    VARIABLE_NAMES,
    NUM_VARIABLES,
    NUM_STATIC_FIELDS,
    MAX_HORIZON
)
model.eval()

result_path = Path(result_path)
result_path.mkdir(parents=True, exist_ok=True)

temporal_stats = {
    'raw_parameters': [],
    'raw_mae': []
}

for idx, (previous, current, time_labels) in tqdm(enumerate(loader), total=len(loader)):
    previous = previous.to(device)
    current = current.view(-1, NUM_VARIABLES, len(lat), len(lon)).to(device)

    var_channels = previous[:, :-NUM_STATIC_FIELDS, :, :]
    static_channels = previous[:, -NUM_STATIC_FIELDS:, :, :]

    parameter_matrix = np.random.uniform(
        low=[-0.2, -0.1, 0.01, 0.0],
        high=[0.2, 0.1, 0.5, 0.2],
        size=(NUM_VARIABLES, param_dim)
    )

    accepted_parameters = []
    accepted_mae = []

    time_tensor = torch.tensor([time_labels[0]], dtype=torch.float32).to(device) / MAX_HORIZON

    for _ in range(n_gibbs_steps):
        for var_idx in range(NUM_VARIABLES):
            base_field = var_channels[:, var_idx, :, :]

            candidate_params = np.random.normal(
                loc=parameter_matrix[var_idx],
                scale=[0.05, 0.05, 0.05, 0.05],
                size=(n_candidates, param_dim)
            )

            best_mae = float("inf")
            best_candidate = parameter_matrix[var_idx].copy()

            for alpha_b, beta_b, alpha_s, beta_s in candidate_params:
                bias_field = alpha_b + beta_b * base_field
                scale_field = alpha_s + beta_s * base_field

                noise = torch.randn_like(base_field)
                perturbed_field = base_field + (noise + bias_field) * scale_field

                perturbed_input = var_channels.clone()
                perturbed_input[:, var_idx, :, :] = perturbed_field

                full_input = torch.cat([perturbed_input, static_channels], dim=1)
                output = model(full_input, time_tensor)
                mae = torch.mean(torch.abs(output - current)).item()

                if mae < best_mae:
                    best_mae = mae
                    best_candidate = np.array([alpha_b, beta_b, alpha_s, beta_s])

            parameter_matrix[var_idx] = best_candidate

        accepted_parameters.append(parameter_matrix.copy())
        accepted_mae.append(best_mae)

    accepted_parameters = np.array(accepted_parameters)
    temporal_stats['raw_parameters'].append(accepted_parameters)
    temporal_stats['raw_mae'].append(np.array(accepted_mae))

    sample_save_path = result_path / f"accepted_parameters_t{idx:05d}.npy"
    np.save(sample_save_path, accepted_parameters)

def epanechnikov_kernel(u):
    return np.maximum(0, 1 - u**2)

def regression_adjustment(params, maes):
    T, V, D = params.shape
    adjusted = np.zeros_like(params)

    for v in range(V):
        for d in range(D):
            X = maes.reshape(-1, 1)
            y = params[:, v, d]
            h = np.std(X) * (len(X) ** (-1/5))
            weights = epanechnikov_kernel((X - np.median(X)) / h).flatten()

            reg = LinearRegression()
            reg.fit(X, y, sample_weight=weights)
            y_adj = y - reg.coef_[0] * (X.flatten() - np.median(X))
            adjusted[:, v, d] = y_adj

    return adjusted

raw_params = np.concatenate(temporal_stats['raw_parameters'], axis=0)
raw_mae = np.concatenate(temporal_stats['raw_mae'], axis=0)
adjusted_params = regression_adjustment(raw_params, raw_mae)

print("\nABC Posterior Diagnostics After Regression Adjustment")
print("-----------------------------------------------------")
print(f"Adjusted MAE: mean = {raw_mae.mean():.4f}, std = {raw_mae.std():.4f}")
print(f"MAE autocorrelation (lag-1): {np.corrcoef(raw_mae[:-1], raw_mae[1:])[0, 1]:.4f}")

for v in range(NUM_VARIABLES):
    for p in range(param_dim):
        samples = adjusted_params[:, v, p]
        entropy = stats.entropy(np.histogram(samples, bins=50, density=True)[0] + 1e-12)
        cv = np.std(samples) / (np.mean(samples) + 1e-8)
        span = np.percentile(samples, 95) - np.percentile(samples, 5)
        print(f"{VARIABLE_NAMES[v]} – {param_labels[p]}: Entropy = {entropy:.3f}, CV = {cv:.3f}, 90% span = {span:.3f}")

mean_over_time = np.mean(adjusted_params.reshape(-1, NUM_VARIABLES, param_dim), axis=0)
for p in range(param_dim):
    plt.figure(figsize=(15, 6))
    for v in range(NUM_VARIABLES):
        plt.plot([mean_over_time[v, p]] * 100, label=f'{param_labels[p]}: {VARIABLE_NAMES[v]}')
    plt.title(f"Posterior Mean ({param_labels[p]}) After Regression Adjustment")
    plt.xlabel("Sample Index")
    plt.ylabel(param_labels[p])
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path / f"posterior_mean_adjusted_{param_labels[p]}.png")
    plt.close()

for p in range(param_dim):
    plt.figure(figsize=(14, 6))
    for v in range(NUM_VARIABLES):
        sns.kdeplot(adjusted_params[:, v, p], label=f'{VARIABLE_NAMES[v]}')
    plt.title(f"Adjusted Posterior Distribution: {param_labels[p]}")
    plt.xlabel(param_labels[p])
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path / f"posterior_adjusted_distribution_{param_labels[p]}.png")
    plt.close()

for p in range(param_dim):
    plt.figure(figsize=(10, 6))
    data = [adjusted_params[:, v, p] for v in range(NUM_VARIABLES)]
    plt.boxplot(data, labels=VARIABLE_NAMES)
    plt.title(f"Adjusted Posterior Boxplot: {param_labels[p]}")
    plt.ylabel(param_labels[p])
    plt.tight_layout()
    plt.savefig(result_path / f"posterior_adjusted_boxplot_{param_labels[p]}.png")
    plt.close()
