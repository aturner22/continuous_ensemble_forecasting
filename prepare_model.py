import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import shutil
import pandas as pd
import datetime
import json
import argparse

from utils import ERA5Dataset
from diffusion_networks import DetPrecond


def prepare_model_and_loader(
    device,
    data_directory,
    result_directory,
    model_directory,
    variable_names,
    num_variables,
    num_static_fields,
    max_horizon,
    random_subset_size: int = None,
    random_subset_seed: int = 42,
):

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

    model_path = "deterministic-iterative-6h"

    result_path = Path(f'{result_directory}/{name}')
    result_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config_path, result_path / "config.json")

    with open(f'{data_directory}/norm_factors.json', 'r') as f:
        statistics = json.load(f)

    mean_data = torch.tensor([statistics[k]["mean"] for k in variable_names]).to(device)
    std_data = torch.tensor([statistics[k]["std"] for k in variable_names]).to(device)
    norm_factors = np.stack([mean_data.cpu().numpy(), std_data.cpu().numpy()], axis=0)

    ti = pd.date_range(datetime.datetime(1979, 1, 1, 0), datetime.datetime(2018, 12, 31, 23), freq='1h')
    n_samples = len(ti)

    print("Number of samples:", n_samples)
    print(f"Using every {spacing}'th sample for evaluation...")

    lat, lon = np.load(f'{data_directory}/latlon_1979-2018_5.625deg.npz').values()

    train_config_path = f'{model_directory}/{model_path}/config.json'
    train_config = load_config(train_config_path)
    filters = train_config['filters']
    conditioning_times = train_config['conditioning_times']

    kwargs = {
        'dataset_path': f'{data_directory}/z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy',
        'sample_counts': (n_samples, 0, 0),
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
    model = DetPrecond(filters=filters, img_channels=input_times, out_channels=num_variables, img_resolution=64)
    model.load_state_dict(torch.load(f'{model_directory}/{model_path}/best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    full_dataset = ERA5Dataset(lead_time=[t_direct], dataset_mode='test', **kwargs)

    if random_subset_size is not None:
        np.random.seed(random_subset_seed)
        subset_indices = np.random.choice(len(full_dataset), size=random_subset_size, replace=False)
        dataset = torch.utils.data.Subset(full_dataset, subset_indices)
        print(f"Using random subset of {random_subset_size} samples (seed={random_subset_seed})")
    else:
        dataset = full_dataset

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)


    return loader, model, lat, lon, result_path
