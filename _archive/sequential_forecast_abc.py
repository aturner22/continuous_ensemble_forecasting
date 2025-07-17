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
        return json.load(file)

config = load_config(args.config_path)
name = config['name']
spacing = config['spacing']
t_direct = config['t_direct']
t_max = config['t_max']
batch_size = 1
t_min = t_direct
t_iter = config['t_iter']
n_ens = config['n_ens']
model_path = config['model']

result_path = Path(f'{result_directory}/{name}')
result_path.mkdir(parents=True, exist_ok=True)
shutil.copy(args.config_path, result_path / "config.json")

with open(f'{data_directory}/norm_factors.json', 'r') as f:
    statistics = json.load(f)

mean_data = torch.tensor([statistics[k]["mean"] for k in variable_names]).to(device)
std_data = torch.tensor([statistics[k]["std"] for k in variable_names]).to(device)
norm_factors = np.stack([mean_data.cpu().numpy(), std_data.cpu().numpy()], axis=0)

def renormalize(x, mean_ar=mean_data, std_ar=std_data):
    return x * std_ar[None, :, None, None] + mean_ar[None, :, None, None]

ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
n_samples, n_train, n_val = len(ti), sum(ti.year <= 2015), sum((ti.year >= 2016) & (ti.year <= 2017))

lat, lon = np.load(f'{data_directory}/latlon_1979-2018_5.625deg.npz').values()

train_config_path = f'{model_directory}/{model_path}/config.json'
train_config = load_config(train_config_path)
filters = train_config['filters']
conditioning_times = train_config['conditioning_times']
delta_t = train_config['delta_t']
model_choice = train_config['model']

kwargs = {
    'dataset_path': f'{data_directory}/z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy',
    'sample_counts': (n_samples, n_train, n_val),
    'dimensions': (num_variables, len(lat), len(lon)),
    'max_horizon': max_horizon,
    'norm_factors': norm_factors,
    'device': device,
    'spacing': spacing,
    'dtype': 'float32',
    'conditioning_times': conditioning_times,
    'lead_time_range': [t_direct, t_direct, t_direct],
    'static_data_path': f'{data_directory}/orog_lsm_1979-2018_5.625deg.npy',
    'random_lead_time': 0,
}

input_times = (len(conditioning_times)) * num_variables + num_static_fields
if 'deterministic' in model_choice:
    if n_ens > 1:
        raise ValueError("Deterministic model cannot be used with n_ens > 1.")
    model = DetPrecond(filters=filters, img_channels=input_times, out_channels=num_variables, img_resolution=64)
else:
    raise ValueError(f"Model choice {model_choice} not recognized.")

model.load_state_dict(torch.load(f'{model_directory}/{model_path}/best_model.pth', map_location=device))
model.to(device)
model.eval()

dataset = ERA5Dataset(lead_time=[t_direct], dataset_mode='test', **kwargs)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

def summarize_and_print_posterior(scales, biases, mae):
    summary = {
        "posterior_mean_scales": np.mean(scales, axis=0).tolist(),
        "posterior_var_scales": np.var(scales, axis=0).tolist(),
        "posterior_mean_biases": np.mean(biases, axis=0).tolist(),
        "posterior_var_biases": np.var(biases, axis=0).tolist(),
        "mae_mean": float(np.mean(mae)),
        "mae_var": float(np.var(mae)),
        "mae_min": float(np.min(mae)),
        "mae_max": float(np.max(mae))
    }
    return summary

n_gibbs_steps = 1000
epsilon = 0.4

temporal_stats = {
    'mean_scales': [],
    'var_scales': [],
    'mean_biases': [],
    'var_biases': [],
    'mae': [],
}

for idx, (previous, current, time_labels) in tqdm(enumerate(loader), total=len(loader)):
    previous = previous.to(device)
    current = current.view(-1, num_variables, len(lat), len(lon)).to(device)

    var_channels = previous[:, :-num_static_fields, :, :]
    static_channels = previous[:, -num_static_fields:, :, :]
    num_vars = var_channels.shape[1]

    perturbation_scales = np.random.uniform(0.01, 0.5, size=num_vars)
    perturbation_biases = np.random.uniform(-0.2, 0.2, size=num_vars)
    accepted_params, accepted_mae = [], []

    for _ in range(n_gibbs_steps):
        for var_idx in range(num_vars):
            prop_scale = np.random.exponential(scale=0.1)
            prop_bias = np.random.normal(loc=0.0, scale=0.1)

            test_scales = perturbation_scales.copy()
            test_biases = perturbation_biases.copy()
            test_scales[var_idx] = prop_scale
            test_biases[var_idx] = prop_bias

            noise = torch.randn_like(var_channels)
            scale = torch.tensor(test_scales, device=device).view(1, -1, 1, 1)
            bias = torch.tensor(test_biases, device=device).view(1, -1, 1, 1)
            perturbed = var_channels + (noise + bias) * scale
            previous_perturbed = torch.cat([perturbed, static_channels], dim=1)

            time_labels_tensor = torch.tensor([time_labels[0]], dtype=torch.float32).to(device) / max_horizon

            output = model(previous_perturbed, time_labels_tensor)
            mae = torch.mean(torch.abs(output - current)).item()

            if mae < epsilon:
                perturbation_scales[var_idx] = prop_scale
                perturbation_biases[var_idx] = prop_bias

        accepted_params.append((perturbation_scales.copy(), perturbation_biases.copy()))
        accepted_mae.append(mae)

    scales = np.array([p[0] for p in accepted_params])
    biases = np.array([p[1] for p in accepted_params])
    mae = np.array(accepted_mae)

    np.savez(result_path / f"abc_gibbs_sample_{idx:05d}.npz", scales=scales, biases=biases, mae=mae)
    summary = summarize_and_print_posterior(scales, biases, mae)

    with open(result_path / f"abc_gibbs_sample_{idx:05d}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # print("\nPerformance and Posterior Metrics for Sample", idx)
    # print("Posterior mean (scales):", summary['posterior_mean_scales'])
    # print("Posterior variance (scales):", summary['posterior_var_scales'])
    # print("Posterior mean (biases):", summary['posterior_mean_biases'])
    # print("Posterior variance (biases):", summary['posterior_var_biases'])
    # print(f"MAE: mean={summary['mae_mean']:.4f}, var={summary['mae_var']:.4f}, min={summary['mae_min']:.4f}, max={summary['mae_max']:.4f}\n")

    temporal_stats['mean_scales'].append(summary['posterior_mean_scales'])
    temporal_stats['var_scales'].append(summary['posterior_var_scales'])
    temporal_stats['mean_biases'].append(summary['posterior_mean_biases'])
    temporal_stats['var_biases'].append(summary['posterior_var_biases'])
    temporal_stats['mae'].append(summary['mae_mean'])

# Plotting temporal behaviour
mean_scales = np.array(temporal_stats['mean_scales'])
mean_biases = np.array(temporal_stats['mean_biases'])
mae_seq = np.array(temporal_stats['mae'])

plt.figure(figsize=(15, 6))
for v in range(min(mean_scales.shape[1], len(variable_names))):

    plt.plot(mean_scales[:, v], label=f'Scale: {variable_names[v]}')
plt.title("Posterior Mean of Scales Over Time")
plt.xlabel("Sample Index")
plt.ylabel("Scale")
plt.legend()
plt.tight_layout()
plt.savefig(result_path / "posterior_mean_scales_over_time.png")
plt.show()

plt.figure(figsize=(15, 6))
for v in range(min(mean_scales.shape[1], len(variable_names))):

    plt.plot(mean_biases[:, v], label=f'Bias: {variable_names[v]}')
plt.title("Posterior Mean of Biases Over Time")
plt.xlabel("Sample Index")
plt.ylabel("Bias")
plt.legend()
plt.tight_layout()
plt.savefig(result_path / "posterior_mean_biases_over_time.png")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(mae_seq)
plt.title("MAE Over Time")
plt.xlabel("Sample Index")
plt.ylabel("Mean Absolute Error")
plt.tight_layout()
plt.savefig(result_path / "mae_over_time.png")
plt.show()
