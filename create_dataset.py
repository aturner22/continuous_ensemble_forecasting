import xarray as xr
import numpy as np
import os
import json
from tqdm import tqdm
import torch
import pandas as pd
import datetime
from utils import ERA5Dataset
from torch.utils.data import DataLoader


file_directory = "era5_data"
save_directory = "./data"
os.makedirs(save_directory, exist_ok=True)

field_names = {
    "orography": "orog",
    "lsm": "lsm",
}

var_names = {
    "geopotential_500": ("z500", "z"),
    "temperature_850": ("t850", "t"),
    "2m_temperature": ("t2m", "t2m"),
    "10m_u_component_of_wind": ("u10", "u10"),
    "10m_v_component_of_wind": ("v10", "v10"),
}

chunk_size = 1000

# Constants
var_name = "constants"
file_pattern = f"{file_directory}/{var_name}/{var_name}*.nc"
df = xr.open_mfdataset(file_pattern, combine='by_coords')

lat = df['lat'].values
lon = df['lon'].values
np.savez(f'{save_directory}/latlon_1979-2018_5.625deg.npz', lat=lat, lon=lon)

## Static fields
static_fields = []
save_name = ''
for field_name, var_name in field_names.items():
    data_array = df[field_name].values
    static_fields.append(data_array)
    save_name += var_name + '_'

np.save(f'{save_directory}/{save_name}1979-2018_5.625deg.npy', np.stack(static_fields, axis=0))

# Variables
file_prefix, names = list(var_names.items())[0]
var_name = names[0]
short_name = names[1]
file_pattern = f"{file_directory}/{file_prefix}/{file_prefix}*.nc"
ds = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'lat': 32, 'lon': 64})
combined_shape = (ds[short_name].shape[0], len(var_names), ds[short_name].shape[1], ds[short_name].shape[2])
print("Shape:", combined_shape)

save_name = '_'.join([var_name[0] for var_name in var_names.values()])
memmap_file_path = f'{save_directory}/{save_name}_1979-2018_5.625deg.npy'
memmap_array = np.memmap(memmap_file_path, dtype='float32', mode='w+', shape=combined_shape)

statistics = {}
mean_value = 0
std_value = 0

i = 0
for file_prefix, names in var_names.items():
    var_name = names[0]
    short_name = names[1]

    file_pattern = f"{file_directory}/{file_prefix}/{file_prefix}*.nc"
    print(f"Opening: {file_pattern}")
    
    # Open the dataset with dask for efficient memory handling
    ds = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'lat': 32, 'lon': 64})
    array = ds[short_name]
    
    # Loop through the chunks of the array and write each chunk to the memmap file
    # We will iterate through the time dimension (0th axis) chunk by chunk
    for j in tqdm(range(0, array.shape[0], chunk_size)):
        end_idx = min(j + chunk_size, array.shape[0])  # Ensure we don't go out of bounds
        
        # Convert the chunk to a NumPy array and assign it to the corresponding memmap slice
        chunk = array[j:end_idx, :, :].compute()  # Compute the chunk lazily
        memmap_array[j:end_idx, i, :, :] = chunk

        # Calculate the mean and std of the chunk
        mean_value += np.sum(chunk).values
        std_value += np.sum(chunk ** 2).values
    
    # Calculate the mean and std of the variable
    num_elements = array.size
    mean_value /= num_elements
    std_value = np.sqrt(std_value / num_elements - mean_value ** 2)
    statistics[var_name] = {"mean": float(mean_value),"std": float(std_value)}

    i += 1

print(f"Combined data saved as memory-mapped file: {memmap_file_path}")

# Save the statistics to a JSON file
json_file = f'{save_directory}/norm_factors.json'
with open(json_file, 'w') as f:
    json.dump(statistics, f, indent=4)

print(f"Normalization factors saved to {json_file}")

# Print out the mean and std for each variable
for var_name, stats in statistics.items():
    print(f"{var_name}: Mean = {stats['mean']}, Std = {stats['std']}")

## Calculate residual stds

variable_names = [k[0] for k in var_names.values()]

mean_data = torch.tensor([stats["mean"] for (key, stats) in statistics.items() if key in variable_names])
std_data = torch.tensor([stats["std"] for (key, stats) in statistics.items() if key in variable_names])
norm_factors = np.stack([mean_data, std_data], axis=0)

# Get the number of samples, training and validation samples
ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
n_samples, n_train, n_val = len(ti), sum(ti.year <= 2015), sum((ti.year >= 2016) & (ti.year <= 2017))

kwargs = {
            'dataset_path':     f'{save_directory}/{save_name}_1979-2018_5.625deg.npy',
            'sample_counts':    (n_samples, n_train, n_val),
            'dimensions':       (len(var_names), 32, 64),
            'max_horizon':      240, # For scaling the time embedding
            'norm_factors':     norm_factors,
            'device':           'cpu',
            'spacing':          1,
            'dtype':            'float32',
            'conditioning_times':    [0],
            'lead_time_range':  (1, 240, 1),
            'static_data_path': None,
            'random_lead_time': 0,
            }

stds_directory = f"{save_directory}/residual_stds"
os.makedirs(stds_directory, exist_ok=True)

def calculate_residual_mean_std(loader):
    mean_data_latent, std_data_latent, count = 0.0, 0.0, 0
    
    with torch.no_grad():
        for current, next, _ in loader:
            inputs = next - current
            count += inputs.size(0)
            
            mean_data_latent += torch.sum(inputs, dim=(0,2,3))
            std_data_latent += torch.sum(inputs ** 2, dim=(0,2,3))
            break # Calculating for a single batch is sufficient
            
        count = count * inputs[0, 0].cpu().detach().numpy().size
        mean_data_latent /= count
        std_data_latent = torch.sqrt(std_data_latent / count - mean_data_latent ** 2)
    
    return mean_data_latent, std_data_latent

lead_time = 1
max_lead_time = 240

bs = 10000
train_dataset = ERA5Dataset(lead_time=lead_time, dataset_mode='train', **kwargs)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

ts = np.arange(lead_time, max_lead_time + 1, 1)

stds_dict = {var_name: [] for var_name in variable_names}

for t in (ts):
    train_dataset.set_lead_time(t)
    
    mean_t, std_t = calculate_residual_mean_std(train_loader)
    print(t, std_t)
    for i, var_name in enumerate(stds_dict):
        stds_dict[var_name].append(std_t[i].item())

save_directory = "data/residual_stds"
os.makedirs(save_directory, exist_ok=True)

for var_name, stds in stds_dict.items():
    stds_content = "\n".join([f"{ts[i]} {std}" for i, std in enumerate(stds)])
    
    file_path = f"{stds_directory}/WB_{var_name}.txt"
    with open(file_path, "w") as file:
        file.write(stds_content)

    print(f"Standard deviations for {var_name} saved to {file_path}")
