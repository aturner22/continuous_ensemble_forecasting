import torch
import numpy as np
import json
import os
import gc
from tqdm import tqdm
import psutil

from config import Config
from core.io_utils import (
    load_model_and_test_data,
    save_posterior_statistics,
    materialise_batches,
    print_computing_configuration,
)
from core.plotting import (
    produce_trace_and_histogram_plots,
    produce_rank_histograms,
    plot_crps_trace,
)
from core.evaluation import print_posterior_summary
from core.gibbs_abc_threaded_rfp import run_gibbs_abc_rfp, compute_safe_batch_size


def load_or_generate_standardized_reference(config, latitude, longitude) -> np.memmap:
    standardized_path = config.data_directory / "z500_t850_t2m_u10_v10_standardized.npy"
    raw_path = config.data_directory / "z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy"
    norm_path = config.data_directory / "norm_factors.json"

    with open(norm_path, "r") as f:
        norm_stats = json.load(f)

    mean_data = torch.tensor([norm_stats[v]["mean"] for v in config.variable_names], dtype=torch.float32)
    std_data = torch.tensor([norm_stats[v]["std"] for v in config.variable_names], dtype=torch.float32)

    if standardized_path.exists():
        print("Loading precomputed standardized ERA5 tensor...")
        return np.load(standardized_path, mmap_mode='r')

    print("Standardized tensor not found. Standardizing raw dataset...")
    temporal_len = 350640
    shape = (temporal_len, config.num_variables, len(latitude), len(longitude))
    raw_array = np.memmap(raw_path, dtype=np.float32, mode='r', shape=shape)

    full_tensor = torch.tensor(raw_array, dtype=torch.float32)
    del raw_array
    gc.collect()

    full_tensor.sub_(mean_data[:, None, None]).div_(std_data[:, None, None])
    np.save(standardized_path, full_tensor.cpu().numpy())

    del full_tensor, mean_data, std_data
    gc.collect()
    return np.load(standardized_path, mmap_mode='r')

def main():
    print("Initializing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_computing_configuration()
    print(f"Using device: {device}")

    timestamp = os.environ.get("CONFIG_TIMESTAMP")
    config = Config("config.json", timestamp=timestamp)

    print("Preparing model and data loader...")
    loader, model, latitude, longitude, result_path = load_model_and_test_data(config, device, config.SEED)

    print("Materializing input batches...")
    cached_batches = list(tqdm(
        materialise_batches(loader, device, config.num_variables, config.max_horizon, latitude, longitude),
        total=config.sample_size,
        desc="Loading batches"
    ))

    print("Preparing standardized reference tensor...")
    reference_mmap = load_or_generate_standardized_reference(config, latitude, longitude)

    example_input, example_output = cached_batches[0][0], cached_batches[0][1]
    C = example_input.shape[1]
    V = example_output.shape[1]
    H, W = example_input.shape[-2:]

    print("Dynamic batch management will be handled automatically during inference...")

    try:
        print("Commencing ABC-Gibbs inference with RFP perturbations...")
        results = run_gibbs_abc_rfp(
            model=model,
            batches=cached_batches,
            ensemble_size=config.ensemble_size,
            n_steps=config.n_gibbs_steps,
            n_proposals=config.n_proposals_per_variable,
            num_variables=config.num_variables,
            variable_names=config.variable_names,
            max_horizon=config.max_horizon,
            reference_mmap=reference_mmap,
            result_directory=result_path,
            log_diagnostics=True
        )


        print("Saving posterior results...")
        save_posterior_statistics(results, result_path)

        print("Releasing memory before plotting...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        torch.cuda.empty_cache()
        gc.collect()

        results_to_keep = {
            "posterior_samples": results["posterior_samples"],
            "rank_histograms": results["rank_histograms"],
            "step_mean_crps": results["step_mean_crps"],
            "posterior_mean": results["posterior_mean"],
            "posterior_variance": results["posterior_variance"],
        }
        del results
        gc.collect()

        print("Generating posterior plots...")
        produce_trace_and_histogram_plots(
            results_to_keep["posterior_samples"],
            result_path,
            config.variable_names,
            ["alpha_scale"]
        )

        produce_rank_histograms(
            results_to_keep["rank_histograms"],
            result_path,
            config.variable_names,
            config.ensemble_size
        )

        plot_crps_trace(
            results_to_keep["step_mean_crps"],
            result_path
        )

        print("Final posterior parameter summary:")
        print_posterior_summary(
            results_to_keep["posterior_mean"],
            results_to_keep["posterior_variance"],
            config.variable_names,
            ["alpha_scale"]
        )

        print("ABC-Gibbs with RFP complete.")

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()