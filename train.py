import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import json  
import argparse
import shutil
from tqdm import tqdm
import pandas as pd
import datetime
import os

from utils import *
from loss import *
from sampler import *

data_directory = './data'
result_directory = './models'

variable_names = ['geopotential_500', 'temperature_850', '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
num_variables, num_static_fields = 5, 2
max_horizon = 240 # Maximum time horizon for the model. Used for scaling time embedding and making sure we don't go outside dataset

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

def get_actual_dataset_shape(dataset_path):
    """Get the actual shape of the dataset file"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Get file size
    file_size = os.path.getsize(dataset_path)
    bytes_per_element = 4  # float32
    
    # Calculate total elements
    total_elements = file_size // bytes_per_element
    
    # Known dimensions
    num_vars = 5
    n_lat = 32
    n_lon = 64
    
    # Calculate n_samples
    n_samples = total_elements // (num_vars * n_lat * n_lon)
    
    print(f"Dataset file size: {file_size} bytes")
    print(f"Calculated dataset shape: ({n_samples}, {num_vars}, {n_lat}, {n_lon})")
    
    return n_samples, num_vars, n_lat, n_lon

# Load config
name            = config['name']
spacing         = config['spacing']
delta_t         = config['delta_t']
t_max           = config['t_max']
t_min           = delta_t
batch_size      = config['batch_size']

num_epochs      = config['num_epochs']
weight_decay    = config['weight_decay']
learning_rate   = config['learning_rate']

filters         = config['filters']
conditioning_times   = config['conditioning_times']
model_choice    = config['model']

# Copy config
result_path = Path(f'{result_directory}/{name}')
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, result_path / "config.json")

# Load precomputed standard deviations
residual_stds = []
for var_name in variable_names:
    std_values = torch.tensor(np.loadtxt(f'{data_directory}/residual_stds/WB_{var_name}.txt', delimiter=' ')[:, 1], dtype=torch.float32).to(device)
    residual_stds.append(std_values)
residual_stds = torch.stack([res_std for res_std in residual_stds], axis=1)[:t_max]

# Load normalization factors
with open(f'{data_directory}/norm_factors.json', 'r') as f:
    json_statistics = json.load(f)

statistics = json_statistics.copy()
for key in json_statistics.keys():
    for subkey in json_statistics[key]:
        statistics[key][subkey] = np.float32(json_statistics[key][subkey])

mean_data = torch.tensor([stats["mean"] for (key, stats) in statistics.items() if key in variable_names])
std_data = torch.tensor([stats["std"] for (key, stats) in statistics.items() if key in variable_names])
norm_factors = np.stack([mean_data, std_data], axis=0)
mean_data = mean_data.to(device)
std_data = std_data.to(device)
def renormalize(x, mean_ar=mean_data, std_ar=std_data):
    x = x * std_ar[None, :, None, None] + mean_ar[None, :, None, None]
    return x

# Get the actual dataset dimensions
dataset_path = f'{data_directory}/geopotential_500_temperature_850_2m_temperature_10m_u_component_of_wind_10m_v_component_of_wind1979-2018_5.625deg.npy'
n_samples, num_vars_check, n_lat, n_lon = get_actual_dataset_shape(dataset_path)

# Verify dimensions match expectations
if num_vars_check != num_variables:
    raise ValueError(f"Expected {num_variables} variables, but dataset has {num_vars_check}")

# Calculate split based on actual sample count
n_train = int(0.8 * n_samples)
n_val = int(0.1 * n_samples)
n_test = n_samples - n_train - n_val

print(f"Actual dataset info:")
print(f"Total samples: {n_samples}")
print(f"Training samples: {n_train}")
print(f"Validation samples: {n_val}")
print(f"Test samples: {n_test}")

# Adjust max_horizon based on actual dataset size
max_horizon = min(max_horizon, n_samples // 4)
print(f"Adjusted max_horizon: {max_horizon}")



# Load the latitudes and longitudes
lat_lon_file = f'{data_directory}/latlon_1979-2018_5.625deg.npz'
if os.path.exists(lat_lon_file):
    lat_lon_data = np.load(lat_lon_file)
    lat, lon = lat_lon_data['lat'], lat_lon_data['lon']
else:
    # Create dummy lat/lon if file doesn't exist
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    print("Warning: Using dummy lat/lon coordinates")


# Check if static data exists
static_data_path = f'{data_directory}/orog_lsm_1979-2018_5.625deg.npy'
if not os.path.exists(static_data_path):
    static_data_path = f'{data_directory}/orog_lsm1979-2018_5.625deg.npy'
    if not os.path.exists(static_data_path):
        print("Warning: Static data file not found, setting to None")
        static_data_path = None
        num_static_fields = 0

kwargs = {
            'dataset_path':     f'{data_directory}/geopotential_500_temperature_850_2m_temperature_10m_u_component_of_wind_10m_v_component_of_wind1979-2018_5.625deg.npy',
            'sample_counts':    (n_samples, n_train, n_val),
            'dimensions':       (num_variables, n_lat, n_lon),
            'max_horizon':      max_horizon, # For scaling the time embedding
            'norm_factors':     norm_factors,
            'device':           device,
            'spacing':          spacing,
            'dtype':            'float32',
            'conditioning_times':    conditioning_times,
            'lead_time_range':  [t_min, t_max, delta_t],
            'static_data_path': static_data_path,
            'random_lead_time': 1,
            }

# Define the batch samplers
update_t_per_batch = get_uniform_t_dist_fn(t_min=delta_t, t_max=t_max, delta_t=delta_t)
train_time_dataset = ERA5Dataset(lead_time=t_max, dataset_mode='train', **kwargs)
train_batch_sampler = DynamicKBatchSampler(train_time_dataset, batch_size=batch_size, drop_last=True, t_update_callback=update_t_per_batch, shuffle=True)
train_time_loader = DataLoader(train_time_dataset, batch_sampler=train_batch_sampler)
val_time_dataset = ERA5Dataset(lead_time=t_max, dataset_mode='val', **kwargs)
val_batch_sampler = DynamicKBatchSampler(val_time_dataset, batch_size=batch_size, drop_last=True, t_update_callback=update_t_per_batch, shuffle=True)
val_time_loader = DataLoader(val_time_dataset, batch_sampler=val_batch_sampler)

input_times = (1 + len(conditioning_times))*num_variables + num_static_fields

if 'autoregressive' in model_choice:
    time_emb = 0
elif 'continuous' in model_choice:
    time_emb = 1
else:
    raise ValueError(f"Model choice {model_choice} not recognized.")

# Define the model and loss function
model = EDMPrecond(filters=filters, img_channels=input_times, out_channels=num_variables, img_resolution = 64, time_emb=time_emb, 
                    sigma_data=1, sigma_min=0.02, sigma_max=88)

loss_fn = WGCLoss(lat, lon, device, precomputed_std=residual_stds)

print(name, flush=True)
print(model_choice, flush=True)
print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)
model.to(device)

print("Lead times", kwargs['lead_time_range'], flush=True)


optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=1000)

loss_values = []
val_loss_values = []
best_val_loss = float('inf')

# Setup for logging
log_file_path = result_path / f'training_log.csv'
with open(log_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss'])

# Training loop
for epoch in range(num_epochs):
    
    # Training phase
    model.train()
    total_train_loss = 0
    for previous, current, time_label in tqdm(train_time_loader):
        current = current.to(device)
        previous = previous.to(device)
        time_label = time_label.to(device)
        
        optimizer.zero_grad()   
        loss = loss_fn(model, current, previous, time_label/max_horizon)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()        
        warmup_scheduler.step()

    avg_train_loss = total_train_loss / len(train_time_loader)
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for previous, current, time_label in (val_time_loader):
            current = current.to(device)
            previous = previous.to(device)
            time_label = time_label.to(device)
            
            loss = loss_fn(model, current, previous, time_label/max_horizon)
            total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_time_loader)

    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), result_path/f'best_model.pth')
        
    scheduler.step()
    
    loss_values.append([avg_train_loss])
    val_loss_values.append(avg_val_loss)
    
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}', flush=True)

    torch.save(model.state_dict(), result_path/f'final_model.pth')

