import xarray as xr
import numpy as np
import json
from tqdm import tqdm
import torch
import pandas as pd
import datetime
from utils import ERA5Dataset
from torch.utils.data import DataLoader
import os

file_directory = "era5_data"
save_directory = "data"

# Create save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

field_names = {
    "orography": "orog",
    "lsm": "lsm",
}

var_names = {
    "geopotential_500": ("geopotential_500", "geopotential_500"),
    "temperature_850": ("temperature_850", "temperature_850"),
    "2m_temperature": ("2m_temperature", "2m_temperature"),
    "10m_u_component_of_wind": ("10m_u_component_of_wind", "10m_u_component_of_wind"),
    "10m_v_component_of_wind": ("10m_v_component_of_wind", "10m_v_component_of_wind"),
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

np.save(f'{save_directory}/{save_name}_1979-2018_5.625deg.npy', np.stack(static_fields, axis=0))
np.save(f'{save_directory}/{save_name}1979-2018_5.625deg.npy', np.stack(static_fields, axis=0))

# Variables - First determine the shape
file_prefix, names = list(var_names.items())[0]
var_name = names[0]
short_name = names[1]
file_pattern = f"{file_directory}/{file_prefix}/{file_prefix}*.nc"
ds = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'lat': 32, 'lon': 64})
combined_shape = (ds[short_name].shape[0], len(var_names), ds[short_name].shape[1], ds[short_name].shape[2])
print("Shape:", combined_shape)

# Create the memmap file
save_name = '_'.join([var_name[0] for var_name in var_names.values()])
memmap_file_path = f'{save_directory}/{save_name}1979-2018_5.625deg.npy'

print(f"Creating memmap file: {memmap_file_path}")
print(f"Shape: {combined_shape}")
print(f"Expected file size: {np.prod(combined_shape) * 4 / (1024**2):.2f} MB")

memmap_array = np.memmap(memmap_file_path, dtype='float32', mode='w+', shape=combined_shape)

# Initialize statistics dictionary properly
statistics = {}

# Process each variable
for i, (file_prefix, names) in enumerate(var_names.items()):
    var_name = names[0]
    short_name = names[1]
    
    print(f"\nProcessing variable {i+1}/{len(var_names)}: {var_name}")
    
    file_pattern = f"{file_directory}/{file_prefix}/{file_prefix}*.nc"
    print(f"Opening: {file_pattern}")
    
    # Open the dataset with dask for efficient memory handling
    ds = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'lat': 32, 'lon': 64})
    array = ds[short_name]
    
    # Initialize statistics for this variable
    sum_value = 0.0
    sum_squared_value = 0.0
    num_elements = 0
    
    # Process in chunks
    for j in tqdm(range(0, array.shape[0], chunk_size), desc=f"Processing {var_name}"):
        end_idx = min(j + chunk_size, array.shape[0])
        
        # Compute the chunk
        chunk = array[j:end_idx, :, :].compute()
        
        # Store in memmap
        memmap_array[j:end_idx, i, :, :] = chunk
        
        # Update statistics for this variable
        chunk_sum = np.sum(chunk)
        chunk_sum_squared = np.sum(chunk ** 2)
        chunk_elements = chunk.size
        
        sum_value += chunk_sum
        sum_squared_value += chunk_sum_squared
        num_elements += chunk_elements
    
    # Calculate final statistics for this variable
    mean_value = sum_value / num_elements
    std_value = np.sqrt(sum_squared_value / num_elements - mean_value ** 2)
    
    statistics[var_name] = {"mean": float(mean_value), "std": float(std_value)}
    print(f"{var_name}: Mean = {mean_value:.6f}, Std = {std_value:.6f}")

# Flush the memmap to ensure data is written
del memmap_array
print(f"\nCombined data saved as memory-mapped file: {memmap_file_path}")

# Verify the file was created correctly
if os.path.exists(memmap_file_path):
    file_size = os.path.getsize(memmap_file_path)
    expected_size = np.prod(combined_shape) * 4  # 4 bytes per float32
    print(f"File size: {file_size} bytes ({file_size / (1024**2):.2f} MB)")
    print(f"Expected size: {expected_size} bytes ({expected_size / (1024**2):.2f} MB)")
    print(f"Size match: {'✓' if file_size == expected_size else '✗'}")
else:
    print("ERROR: Memmap file was not created!")

# Save the statistics to a JSON file
json_statistics = {}
for var_name, stats in statistics.items():
    json_statistics[var_name] = {"mean": str(stats["mean"]), "std": str(stats["std"])}

json_file = f'{save_directory}/norm_factors.json'
with open(json_file, 'w') as f:
    json.dump(json_statistics, f, indent=4)

print(f"Normalization factors saved to {json_file}")

# Print out the mean and std for each variable
print("\nFinal statistics:")
for var_name, stats in statistics.items():
    print(f"{var_name}: Mean = {stats['mean']}, Std = {stats['std']}")

## Calculate residual stds
variable_names = [k[0] for k in var_names.values()]

mean_data = torch.tensor([stats["mean"] for var_name, stats in statistics.items() if var_name in variable_names])
std_data = torch.tensor([stats["std"] for var_name, stats in statistics.items() if var_name in variable_names])
norm_factors = np.stack([mean_data, std_data], axis=0)

# Get the actual number of samples from the dataset shape
actual_n_samples = combined_shape[0]  # Use the actual number of samples from the data

# For a small dataset, let's use a simple split: 80% train, 10% val, 10% test
n_train = int(0.8 * actual_n_samples)
n_val = int(0.1 * actual_n_samples)
n_test = actual_n_samples - n_train - n_val

print(f"\nDataset info:")
print(f"Total samples: {actual_n_samples}")
print(f"Training samples: {n_train}")
print(f"Validation samples: {n_val}")
print(f"Test samples: {n_test}")

kwargs = {
    'dataset_path':     f'{save_directory}/{save_name}1979-2018_5.625deg.npy',
    'sample_counts':    (actual_n_samples, n_train, n_val),
    'dimensions':       (len(var_names), 32, 64),
    'max_horizon':      min(240, actual_n_samples // 4), # Adjust max_horizon based on actual samples
    'norm_factors':     norm_factors,
    'device':           'cpu',
    'spacing':          1,
    'dtype':            'float32',
    'conditioning_times':    [0],
    'lead_time_range':  (1, min(240, actual_n_samples // 4), 1),
    'static_data_path': None,
    'random_lead_time': 0,
}

# Create residual stds directory
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

print("\nCalculating residual statistics...")

lead_time = 1
max_lead_time = min(20, actual_n_samples // 4)  # Reduce max_lead_time for small dataset
bs = min(32, n_train)  # Reduce batch size for small dataset

try:
    train_dataset = ERA5Dataset(lead_time=lead_time, dataset_mode='train', **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    ts = np.arange(lead_time, max_lead_time + 1, 1)
    stds_dict = {var_name: [] for var_name in variable_names}
    
    for t in tqdm(ts, desc="Computing residual stds"):
        train_dataset.set_lead_time(t)
        
        mean_t, std_t = calculate_residual_mean_std(train_loader)
        for i, var_name in enumerate(stds_dict):
            stds_dict[var_name].append(std_t[i].item())
    
    # Save residual stds
    for var_name, stds in stds_dict.items():
        stds_content = "\n".join([f"{ts[i]} {std}" for i, std in enumerate(stds)])
        
        file_path = f"{stds_directory}/WB_{var_name}.txt"
        with open(file_path, "w") as file:
            file.write(stds_content)
        
        print(f"Standard deviations for {var_name} saved to {file_path}")

except Exception as e:
    print(f"Error during residual statistics calculation: {e}")
    print("The main dataset creation was successful, but residual stats failed.")
    print("You can run the residual stats calculation separately later.")