import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import multiprocessing
import os
import platform
import json

from utils import ERA5Dataset
from diffusion_networks import DetPrecond

from config import Config
import pandas as pd
import shutil
import datetime 


def load_model_and_test_data(
    config: Config,
    device: torch.device,
    random_subset_seed: int,
) -> tuple[DataLoader, torch.nn.Module, np.ndarray, np.ndarray, Path]:

    result_path = config.result_directory
    result_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.config_path, result_path / "config.json")

    with open(config.data_directory / "norm_factors.json", "r") as f:
        statistics = json.load(f)

    mean_data = torch.tensor([statistics[k]["mean"] for k in config.variable_names], dtype=torch.float32, device=device)
    std_data = torch.tensor([statistics[k]["std"] for k in config.variable_names], dtype=torch.float32, device=device)
    norm_factors = np.stack([mean_data.cpu().numpy(), std_data.cpu().numpy()], axis=0)

    ti = pd.date_range(datetime.datetime(1979, 1, 1, 0), datetime.datetime(2018, 12, 31, 23), freq='1h')
    n_samples = len(ti)

    print("Number of samples:", n_samples)
    print(f"Using every {config.spacing}'th sample for evaluation...")

    lat, lon = np.load(config.data_directory / "latlon_1979-2018_5.625deg.npz").values()

    # train_config_path = config.model_directory / config.name / "config.json"
    # with open(train_config_path, "r") as f:
    #     train_config = json.load(f)

    filters = 32
    conditioning_times = [0, -6]

    kwargs = {
        "dataset_path": str(config.data_directory / "z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy"),
        "sample_counts": (n_samples, 0, 0),
        "dimensions": (config.num_variables, len(lat), len(lon)),
        "max_horizon": config.max_horizon,
        "norm_factors": norm_factors,
        "device": device,
        "spacing": config.spacing,
        "dtype": "float32",
        "conditioning_times": conditioning_times,
        "lead_time_range": [config.t_direct, config.t_direct, config.t_direct],
        "static_data_path": str(config.data_directory / "orog_lsm_1979-2018_5.625deg.npy"),
        "random_lead_time": 0,
    }

    input_channels = len(conditioning_times) * config.num_variables + config.num_static_fields
    model = DetPrecond(filters=filters, img_channels=input_channels, out_channels=config.num_variables, img_resolution=64)
    model_ckpt = "./models/deterministic-iterative-6h/best_model.pth"
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    full_dataset = ERA5Dataset(lead_time=[config.t_direct], dataset_mode="test", **kwargs)

    if config.sample_size is not None:
        np.random.seed(random_subset_seed)
        subset_indices = np.random.choice(len(full_dataset), size=config.sample_size, replace=False)
        dataset = torch.utils.data.Subset(full_dataset, subset_indices)
        print(f"Using random subset of {config.sample_size} samples (seed={random_subset_seed})")
    else:
        dataset = full_dataset

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    return loader, model, lat, lon, result_path


def save_posterior_statistics(results: dict, output_directory: Path):
    np.save(output_directory / "posterior_samples.npy", results["posterior_samples"])
    np.save(output_directory / "posterior_crps.npy", results["posterior_crps"])
    np.save(output_directory / "posterior_mean.npy", results["posterior_mean"])
    np.save(output_directory / "posterior_variance.npy", results["posterior_variance"])
    np.save(output_directory / "ensemble_mae.npy", results["ensemble_mae"])
    np.save(output_directory / "ensemble_spread.npy", results["ensemble_spread"])

def print_computing_configuration():
    print("\n--- Computing Configuration ---")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python version: {platform.python_version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    print(f"torch version: {torch.__version__}")
    print(f"Number of CPUs: {os.cpu_count()}")
    print(f"Physical CPU cores: {multiprocessing.cpu_count()}")
    print(f"OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS: {os.getenv('MKL_NUM_THREADS')}")
    print(f"OPENBLAS_NUM_THREADS: {os.getenv('OPENBLAS_NUM_THREADS')}")
    print(f"TORCH_NUM_THREADS: {torch.get_num_threads()}")
    print(f"N_WORKERS: {os.getenv('N_WORKERS')}")
    print(f"PARALLEL_BACKEND: {os.getenv('PARALLEL_BACKEND')}")
    print("--------------------------------\n")


def materialise_batches(loader, device, num_variables, max_horizon, latitude, longitude):
    batches = []
    for previous_fields, current_fields, valid_time in loader:
        previous_fields = previous_fields.to(device)
        current_fields = current_fields.view(-1, num_variables, len(latitude), len(longitude)).to(device)
        time_normalised = torch.tensor([valid_time[0]], dtype=torch.float32, device=device) / max_horizon
        batches.append((previous_fields, current_fields, time_normalised))
    return batches
