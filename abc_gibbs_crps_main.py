if __name__ == "__main__":
    from utils import set_optimal_start_method
    print("MP start-method:", set_optimal_start_method())

    import os, multiprocessing as mp, math

    CORES = os.cpu_count() or 1                 # physical + E-cores on macOS
    SOFT_CAP = 0.8                              # keep ~20 % free
    MAX_WORKERS = max(1, math.floor(CORES * SOFT_CAP))

    N_WORKERS = int(os.getenv("N_WORKERS", MAX_WORKERS))
    OMP_NUM_THREADS = max(1, CORES // N_WORKERS)

    os.environ.setdefault("OMP_NUM_THREADS", str(OMP_NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(OMP_NUM_THREADS))
    print(f"Using {N_WORKERS} workers {OMP_NUM_THREADS} BLAS threads")

    import json
    import torch
    import os

    print("Torch threads:", torch.get_num_threads())
    print("NumPy threads:", os.environ.get("OMP_NUM_THREADS", "undefined"))
    device = torch.device("cpu")
    import numpy as np
    import random

    from pathlib import Path
    from tqdm import tqdm

    from core.constants import (
        SAMPLE_SIZE, ENSEMBLE_SIZE, N_GIBBS_STEPS, N_PROPOSALS_PER_VARIABLE,
        VARIABLE_NAMES, NUM_VARIABLES, NUM_STATIC_FIELDS, MAX_HORIZON,
        PARAMETER_DIMENSION, PARAMETER_LABELS,
        DATA_DIRECTORY, MODEL_DIRECTORY, RESULT_DIRECTORY
    )
    from core.io_utils import prepare_model_and_loader, save_posterior_statistics
    from core.helpers import materialise_batches
    from core.gibbs_abc_threaded import run_gibbs_abc as threaded_run_gibbs_abc
    from core.plotting import produce_trace_and_histogram_plots, produce_rank_histograms
    from core.diagnostics import print_posterior_summary

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print(f"Using device: {device}")

    with open("configs/predict/deterministic-iterative-6h.json", "r") as config_file:
        config = json.load(config_file)

    loader, model, latitude, longitude, _ = prepare_model_and_loader(
        device=device,
        data_directory=str(DATA_DIRECTORY),
        result_directory=str(RESULT_DIRECTORY),
        model_directory=str(MODEL_DIRECTORY),
        variable_names=VARIABLE_NAMES,
        num_variables=NUM_VARIABLES,
        num_static_fields=NUM_STATIC_FIELDS,
        max_horizon=MAX_HORIZON,
        random_subset_size=SAMPLE_SIZE,
        random_subset_seed=777,
    )
    model.eval()

    cached_batches = materialise_batches(loader, device, NUM_VARIABLES, MAX_HORIZON, latitude, longitude)

    print("RUNNING:", threaded_run_gibbs_abc.__module__)

    results = threaded_run_gibbs_abc(
        model=model,
        batches=cached_batches,
        ensemble_size=ENSEMBLE_SIZE,
        n_steps=N_GIBBS_STEPS,
        n_proposals=N_PROPOSALS_PER_VARIABLE,
        num_variables=NUM_VARIABLES,
        parameter_dim=PARAMETER_DIMENSION,
        variable_names=VARIABLE_NAMES,
        max_horizon=MAX_HORIZON
    )

    save_posterior_statistics(results, RESULT_DIRECTORY)
    produce_trace_and_histogram_plots(results["posterior_samples"], RESULT_DIRECTORY, VARIABLE_NAMES, PARAMETER_LABELS)
    produce_rank_histograms(results["rank_histograms"], RESULT_DIRECTORY, VARIABLE_NAMES, ENSEMBLE_SIZE)
    print_posterior_summary(results["posterior_mean"], results["posterior_variance"], VARIABLE_NAMES, PARAMETER_LABELS)
