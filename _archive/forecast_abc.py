import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm
import pandas as pd
import datetime
import gc
import zarr
import seaborn as sns
import json
import argparse

from utils import *
from archive.loss import *

data_directory = './data'
result_directory = './results'
model_directory = './models'

variable_names = ['z500', 't850', 't2m', 'u10', 'v10']
num_variables, num_static_fields = 5, 2
max_horizon = 240

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Run model with configuration from JSON file.')
parser.add_argument('config_path', type=str, help='Path to JSON configuration file.')
args = parser.parse_args()

def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config
config_path = args.config_path
config = load_config(config_path)

# Load config
name        =   config['name']
spacing     =   config['spacing']
t_direct =      config['t_direct']
t_max =         config['t_max']
batch_size =    config['batch_size']
t_min =         t_direct
t_iter =        config['t_iter']
n_ens =         config['n_ens']
model_path =    config['model']

print(name, flush=True)
print("[t_direct, t_iter, t_max]", [t_direct, t_iter, t_max],  flush=True)
print("n_ens:", n_ens,  flush=True)

# Copy config
result_path = Path(f'{result_directory}/{name}')
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, result_path / "config.json")

# Load normalization factors
with open(f'{data_directory}/norm_factors.json', 'r') as f:
    statistics = json.load(f)
mean_data = torch.tensor([stats["mean"] for (key, stats) in statistics.items() if key in variable_names])
std_data = torch.tensor([stats["std"] for (key, stats) in statistics.items() if key in variable_names])
norm_factors = np.stack([mean_data, std_data], axis=0)
mean_data = mean_data.to(device)
std_data = std_data.to(device)
def renormalize(x, mean_ar=mean_data, std_ar=std_data):
    x = x * std_ar[None, :, None, None] + mean_ar[None, :, None, None]
    return x

# Get the number of samples, training and validation samples
ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
n_samples, n_train, n_val = len(ti), sum(ti.year <= 2015), sum((ti.year >= 2016) & (ti.year <= 2017))

# Load the latitudes and longitudes
lat, lon = np.load(f'{data_directory}/latlon_1979-2018_5.625deg.npz').values()

# Load config of trained model
train_config_path = f'{model_directory}/{model_path}/config.json'
config = load_config(train_config_path)

# Constants and configurations loaded from JSON
filters     = config['filters']
max_trained_lead_time = config['t_max']
conditioning_times = config['conditioning_times']
delta_t = config['delta_t']
model_choice = config['model']

if t_iter > max_trained_lead_time:
    print(f"The iterative lead time {t_iter} is larger than the maximum trained lead time {max_trained_lead_time}")
if t_direct < delta_t:
    print(f"The direct lead time {t_direct} is smaller than the trained dt {delta_t}")

kwargs = {
            'dataset_path':     f'{data_directory}/z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy',
            'sample_counts':    (n_samples, n_train, n_val),
            'dimensions':       (num_variables, len(lat), len(lon)),
            'max_horizon':      max_horizon,
            'norm_factors':     norm_factors,
            'device':           device,
            'spacing':          spacing,
            'dtype':            'float32',
            'conditioning_times':    conditioning_times,
            'lead_time_range':  [t_min, t_max, t_direct],
            'static_data_path': f'{data_directory}/orog_lsm_1979-2018_5.625deg.npy',
            'random_lead_time': 0,
            }

input_times = (1 + len(conditioning_times))*num_variables + num_static_fields

if 'deterministic' in model_choice:
    if n_ens > 1:
        raise ValueError("Deterministic model can not be used with n_ens > 1. Use n_ens = 1 for deterministic models.")
    deterministic = True    
    input_times = (len(conditioning_times))*num_variables + num_static_fields
    model = DetPrecond(filters=filters, img_channels=input_times, out_channels=num_variables, img_resolution = 64)
    print("Using deterministic model", flush=True)
else:
    raise ValueError(f"Model choice {model_choice} not recognized.")

model.load_state_dict(torch.load(f'{model_directory}/{model_path}/best_model.pth', map_location=device))
model.to(device)

print(f"Loaded model {model_path}, {model_choice}",  flush=True)
print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)

forecasting_times = t_min + t_direct * np.arange(0, 1 + (t_max-t_min)//t_direct)
dataset = ERA5Dataset(lead_time=forecasting_times, dataset_mode='test', **kwargs)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print(f"Datset contains {len(dataset)} samples",  flush=True)
print(f"We do {len(loader)} batches",  flush=True)

model.eval()

# --- ABC-Gibbs for noise hyperparameters ---
# Get num_vars from the data
for previous, current, time_labels in loader:
    var_channels = previous[:, :-num_static_fields, :, :]
    num_vars = var_channels.shape[1]
    break

n_gibbs_steps = 1000
epsilon = 0.4

perturbation_scales = np.random.uniform(0.01, 0.5, size=num_vars)
perturbation_biases = np.random.uniform(-0.2, 0.2, size=num_vars)

accepted_params = []
accepted_mae = []

for gibbs_iter in tqdm(range(n_gibbs_steps)):
    for var_idx in range(num_vars):
        # Propose new scale and bias for this variable
        prop_scale = np.random.exponential(scale=0.1)  # Exponential prior, adjust scale as needed
        prop_bias = np.random.normal(loc=0.0, scale=0.1)  # Normal prior, adjust mean and std as needed

        # Build scale and bias arrays for all variables
        test_scales = perturbation_scales.copy()
        test_biases = perturbation_biases.copy()
        test_scales[var_idx] = prop_scale
        test_biases[var_idx] = prop_bias

        mae_list = []
        for previous, current, time_labels in loader:
            with torch.no_grad():
                previous = previous.to(device)
                current = current.view(-1, num_variables, len(lat), len(lon)).to(device)

                # Only perturb variable channels, not static fields
                var_channels = previous[:, :-num_static_fields, :, :]
                static_channels = previous[:, -num_static_fields:, :, :]

                # Apply independent noise per variable channel
                noise = torch.randn_like(var_channels)
                scale = torch.tensor(test_scales, device=device).view(1, -1, 1, 1)
                bias = torch.tensor(test_biases, device=device).view(1, -1, 1, 1)
                perturbed_var_channels = var_channels + (noise + bias) * scale

                previous_perturbed = torch.cat([perturbed_var_channels, static_channels], dim=1)

                # Prepare time_labels as before
                time_labels_tensor = time_labels[:, 0].to(device) / max_horizon

                output = model(previous_perturbed, time_labels_tensor)
                mae = torch.mean(torch.abs(output - current)).item()
                mae_list.append(mae)

        mean_mae = np.mean(mae_list)

        # ABC-Gibbs accept/reject
        if mean_mae < epsilon:
            perturbation_scales[var_idx] = prop_scale
            perturbation_biases[var_idx] = prop_bias

    # Store current state after each sweep
    accepted_params.append((perturbation_scales.copy(), perturbation_biases.copy()))
    accepted_mae.append(mean_mae)

# Save results
np.savez(f"{result_path}/abc_gibbs_params.npz",
         scales=np.array([p[0] for p in accepted_params]),
         biases=np.array([p[1] for p in accepted_params]),
         mae=accepted_mae)
print(f"ABC-Gibbs complete. {len(accepted_params)} samples stored.")

# --- Plotting ABC-Gibbs results ---
abc_gibbs_results = np.load(f"{result_path}/abc_gibbs_params.npz", allow_pickle=True)
scales = abc_gibbs_results['scales']
biases = abc_gibbs_results['biases']
mae = abc_gibbs_results['mae']

# 1. Trace plots for scales and biases
plt.figure(figsize=(12, 6))
for var_idx in range(num_vars):
    plt.subplot(2, num_vars, var_idx + 1)
    plt.plot(scales[:, var_idx])
    plt.xlabel('Gibbs Iteration')
    plt.ylabel(f'Scale (Var {var_idx+1})')
    plt.subplot(2, num_vars, var_idx + 1 + num_vars)
    plt.plot(biases[:, var_idx])
    plt.xlabel('Gibbs Iteration')
    plt.ylabel(f'Bias (Var {var_idx+1})')
plt.tight_layout()
plt.show()

# 2. Marginal distributions of scales and biases
fig, axs = plt.subplots(2, num_vars, figsize=(12, 8))
for var_idx in range(num_vars):
    axs[0, var_idx].hist(scales[:, var_idx], bins=30, color='skyblue', edgecolor='k', alpha=0.7)
    axs[0, var_idx].set_xlabel(f'Scale (Var {var_idx+1})')
    axs[0, var_idx].set_ylabel('Count')
    axs[1, var_idx].hist(biases[:, var_idx], bins=30, color='salmon', edgecolor='k', alpha=0.7)
    axs[1, var_idx].set_xlabel(f'Bias (Var {var_idx+1})')
    axs[1, var_idx].set_ylabel('Count')
plt.tight_layout()
plt.show()

# 3. Pair plots for selected variables
selected_vars = [0, 1]  # Select first two variables for pair plot
scales_selected = scales[:, selected_vars]
biases_selected = biases[:, selected_vars]

# Pair plot for scales
sns.pairplot(pd.DataFrame(scales_selected, columns=[f'Scale Var {i+1}' for i in selected_vars]), diag_kind='kde')
plt.suptitle('Pair Plot of Scales for Selected Variables')
plt.show()

# Pair plot for biases
sns.pairplot(pd.DataFrame(biases_selected, columns=[f'Bias Var {i+1}' for i in selected_vars]), diag_kind='kde')
plt.suptitle('Pair Plot of Biases for Selected Variables')
plt.show()

# --- Summary statistics for ABC-Gibbs ---

# Posterior means and variances for scales and biases
posterior_mean_scales = np.mean(scales, axis=0)
posterior_var_scales = np.var(scales, axis=0)
posterior_mean_biases = np.mean(biases, axis=0)
posterior_var_biases = np.var(biases, axis=0)

# MAE statistics
mean_mae = np.mean(mae)
var_mae = np.var(mae)
min_mae = np.min(mae)
max_mae = np.max(mae)

print("\nABC-Gibbs Summary Metrics:")
print("Posterior mean (scales):", posterior_mean_scales)
print("Posterior variance (scales):", posterior_var_scales)
print("Posterior mean (biases):", posterior_mean_biases)
print("Posterior variance (biases):", posterior_var_biases)
print(f"MAE: mean={mean_mae:.4f}, var={var_mae:.4f}, min={min_mae:.4f}, max={max_mae:.4f}")

# Store summary metrics to a JSON file
summary_metrics = {
    "posterior_mean_scales": posterior_mean_scales.tolist(),
    "posterior_var_scales": posterior_var_scales.tolist(),
    "posterior_mean_biases": posterior_mean_biases.tolist(),
    "posterior_var_biases": posterior_var_biases.tolist(),
    "mae_mean": float(mean_mae),
    "mae_var": float(var_mae),
    "mae_min": float(min_mae),
    "mae_max": float(max_mae)
}
with open(result_path / "abc_gibbs_summary_metrics.json", "w") as f:
    json.dump(summary_metrics, f, indent=2)
print(f"Summary metrics saved to {result_path / 'abc_gibbs_summary_metrics.json'}")

# Regression post-processing (Beaumont adjustment) for each variable
from sklearn.linear_model import LinearRegression

scales_reg_adj = np.zeros_like(scales)
biases_reg_adj = np.zeros_like(biases)

for var_idx in range(num_vars):
    # Log-linear regression for scale (to ensure positivity)
    reg_scale = LinearRegression().fit(mae.reshape(-1, 1), np.log(scales[:, var_idx]))
    # Standard linear regression for bias
    reg_bias = LinearRegression().fit(mae.reshape(-1, 1), biases[:, var_idx])
    # Predict at ideal MAE (0)
    scales_reg_adj[:, var_idx] = np.exp(reg_scale.predict(np.zeros((len(mae), 1))))
    biases_reg_adj[:, var_idx] = reg_bias.predict(np.zeros((len(mae), 1)))

# Optionally, use the regression-adjusted samples for summary statistics:
posterior_mean_scales_reg = np.mean(scales_reg_adj, axis=0)
posterior_var_scales_reg = np.var(scales_reg_adj, axis=0)
posterior_mean_biases_reg = np.mean(biases_reg_adj, axis=0)
posterior_var_biases_reg = np.var(biases_reg_adj, axis=0)

print("\nRegression-adjusted Posterior Means (scales):", posterior_mean_scales_reg)
print("Regression-adjusted Posterior Vars (scales):", posterior_var_scales_reg)
print("Regression-adjusted Posterior Means (biases):", posterior_mean_biases_reg)
print("Regression-adjusted Posterior Vars (biases):", posterior_var_biases_reg)

# Save regression-adjusted summary metrics
summary_metrics_reg = {
    "posterior_mean_scales_reg": posterior_mean_scales_reg.tolist(),
    "posterior_var_scales_reg": posterior_var_scales_reg.tolist(),
    "posterior_mean_biases_reg": posterior_mean_biases_reg.tolist(),
    "posterior_var_biases_reg": posterior_var_biases_reg.tolist()
}
with open(result_path / "abc_gibbs_regression_adjusted_summary_metrics.json", "w") as f:
    json.dump(summary_metrics_reg, f, indent=2)
print(f"Regression-adjusted summary metrics saved to {result_path / 'abc_gibbs_regression_adjusted_summary_metrics.json'}")

# Optionally, plot regression-adjusted vs original
fig, axs = plt.subplots(2, num_vars, figsize=(12, 8))
for var_idx in range(num_vars):
    axs[0, var_idx].hist(scales[:, var_idx], bins=30, color='skyblue', edgecolor='k', alpha=0.5, label='Original')
    axs[0, var_idx].hist(scales_reg_adj[:, var_idx], bins=30, color='red', edgecolor='k', alpha=0.5, label='Reg. Adj.')
    axs[0, var_idx].set_xlabel(f'Scale (Var {var_idx+1})')
    axs[0, var_idx].set_ylabel('Count')
    axs[1, var_idx].hist(biases[:, var_idx], bins=30, color='salmon', edgecolor='k', alpha=0.5, label='Original')
    axs[1, var_idx].hist(biases_reg_adj[:, var_idx], bins=30, color='purple', edgecolor='k', alpha=0.5, label='Reg. Adj.')
    axs[1, var_idx].set_xlabel(f'Bias (Var {var_idx+1})')
    axs[1, var_idx].set_ylabel('Count')
    axs[0, var_idx].legend()
    axs[1, var_idx].legend()
plt.tight_layout()
plt.show()
