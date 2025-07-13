import torch
import multiprocessing
import os
import platform

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
