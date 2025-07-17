import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm
import pandas as pd
import datetime
import json
import argparse
import statsmodels.api as sm

from utils import *
from archive.loss import *

# --- CONFIGURATION ---
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

# --- Load posterior means/variances from summary files ---
summary_files = sorted(result_path.glob("abc_gibbs_sample_*_summary.json"))

mean_scales = []
var_scales = []
mean_biases = []
var_biases = []

for f in summary_files:
    with open(f, "r") as jf:
        summary = json.load(jf)
        mean_scales.append(summary["posterior_mean_scales"])
        var_scales.append(summary["posterior_var_scales"])
        mean_biases.append(summary["posterior_mean_biases"])
        var_biases.append(summary["posterior_var_biases"])

mean_scales = np.array(mean_scales)  # shape: (n_times, n_vars)
var_scales = np.array(var_scales)
mean_biases = np.array(mean_biases)
var_biases = np.array(var_biases)

# --- Fit ARMA models to each variable's mean series ---
arma_models_scales = []
arma_models_biases = []

for v in range(mean_scales.shape[1]):
    # Fit ARMA(p,q) with order selection by AIC (try p,q in 0..3)
    best_aic = np.inf
    best_model = None
    for p in range(4):
        for q in range(4):
            try:
                arma_model = sm.tsa.ARIMA(mean_scales[:, v], order=(p, 0, q))
                result = arma_model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_model = result
            except Exception:
                continue
    arma_models_scales.append(best_model)

    best_aic = np.inf
    best_model = None
    for p in range(4):
        for q in range(4):
            try:
                arma_model = sm.tsa.ARIMA(mean_biases[:, v], order=(p, 0, q))
                result = arma_model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_model = result
            except Exception:
                continue
    arma_models_biases.append(best_model)

# --- Recalibrated simulation ---
n_sim = 200
recalib_mae = []
baseline_mae = []

for idx, (previous, current, time_labels) in tqdm(enumerate(loader), total=len(loader)):
    previous = previous.to(device)
    current = current.view(-1, num_variables, len(lat), len(lon)).to(device)
    var_channels = previous[:, :-num_static_fields, :, :]
    static_channels = previous[:, -num_static_fields:, :, :]

    # ARMA-predicted mean for this time step using in-sample one-step-ahead forecast
    pred_mean_scales = np.array([
        arma_models_scales[v].predict(start=idx, end=idx)[0] if arma_models_scales[v] is not None else mean_scales[idx, v]
        for v in range(num_variables)
    ])
    pred_mean_biases = np.array([
        arma_models_biases[v].predict(start=idx, end=idx)[0] if arma_models_biases[v] is not None else mean_biases[idx, v]
        for v in range(num_variables)
    ])

    # Use the mean of the posterior variance as the std for the noise (or use ARMA on variance if desired)
    pred_std_scales = np.sqrt(np.mean(var_scales, axis=0))
    pred_std_biases = np.sqrt(np.mean(var_biases, axis=0))

    # --- Recalibrated simulations ---
    sim_mae = []
    for sim in range(n_sim):
        noise = torch.randn_like(var_channels)
        n_time_steps = var_channels.shape[1] // num_variables
        scale = torch.tensor(np.ones(num_variables)*0.1, device=device).repeat(n_time_steps, 1).reshape(1, -1, 1, 1)
        bias = torch.tensor(np.zeros(num_variables), device=device).repeat(n_time_steps, 1).reshape(1, -1, 1, 1)
        perturbed = var_channels + (noise + bias) * scale
        previous_perturbed = torch.cat([perturbed, static_channels], dim=1)
        time_labels_tensor = torch.tensor([time_labels[0]], dtype=torch.float32).to(device) / max_horizon
        output = model(previous_perturbed, time_labels_tensor)
        mae = torch.mean(torch.abs(output - current)).item()
        sim_mae.append(mae)
    recalib_mae.append(sim_mae)

    # --- Baseline simulations (original noise) ---
    sim_mae_base = []
    for sim in range(n_sim):
        noise = torch.randn_like(var_channels)
        scale = torch.tensor(np.ones(num_variables)*0.1, device=device).view(1, -1, 1, 1)
        bias = torch.tensor(np.zeros(num_variables), device=device).view(1, -1, 1, 1)
        perturbed = var_channels + (noise + bias) * scale
        previous_perturbed = torch.cat([perturbed, static_channels], dim=1)
        output = model(previous_perturbed, time_labels_tensor)
        mae = torch.mean(torch.abs(output - current)).item()
        sim_mae_base.append(mae)
    baseline_mae.append(sim_mae_base)

# --- Save results ---
np.savez(result_path / "recalibrated_mae.npz", recalib_mae=np.array(recalib_mae), baseline_mae=np.array(baseline_mae))

# --- Plotting recalibrated vs baseline ---
recalib_mae = np.array(recalib_mae)
baseline_mae = np.array(baseline_mae)

plt.figure(figsize=(12, 5))
plt.plot(np.mean(recalib_mae, axis=1), label='Recalibrated Mean MAE')
plt.plot(np.mean(baseline_mae, axis=1), label='Baseline Mean MAE')
plt.fill_between(np.arange(len(recalib_mae)), np.percentile(recalib_mae, 5, axis=1), np.percentile(recalib_mae, 95, axis=1), alpha=0.2, color='blue')
plt.fill_between(np.arange(len(baseline_mae)), np.percentile(baseline_mae, 5, axis=1), np.percentile(baseline_mae, 95, axis=1), alpha=0.2, color='orange')
plt.xlabel("Time Index")
plt.ylabel("MAE")
plt.legend()
plt.title("Recalibrated vs Baseline MAE Over Time")
plt.tight_layout()
plt.savefig(result_path / "recalibrated_vs_baseline_mae.png")
plt.show()