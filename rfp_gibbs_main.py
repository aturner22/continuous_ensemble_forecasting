if __name__ == "__main__":
    import os
    import numpy as np
    import torch
    import json
    from tqdm import tqdm
    import gc

    from core.constants import (
        SAMPLE_SIZE, ENSEMBLE_SIZE, N_GIBBS_STEPS, N_PROPOSALS_PER_VARIABLE,
        VARIABLE_NAMES, NUM_VARIABLES, NUM_STATIC_FIELDS, MAX_HORIZON,
        DATA_DIRECTORY, MODEL_DIRECTORY, RESULT_DIRECTORY
    )
    from core.io_utils import prepare_model_and_loader, save_posterior_statistics
    from core.helpers import materialise_batches, print_computing_configuration
    from core.plotting import produce_trace_and_histogram_plots, produce_rank_histograms
    from core.diagnostics import print_posterior_summary
    from core.gibbs_abc_threaded_rfp import run_gibbs_abc_rfp

    print("Initializing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_computing_configuration()
    print(f"Using device: {device}")

    print("Preparing model and data loader...")
    loader, model, latitude, longitude, result_path = prepare_model_and_loader(
        device=device,
        data_directory=str(DATA_DIRECTORY),
        result_directory=str(RESULT_DIRECTORY / "rfp_variant"),
        model_directory=str(MODEL_DIRECTORY),
        variable_names=VARIABLE_NAMES,
        num_variables=NUM_VARIABLES,
        num_static_fields=NUM_STATIC_FIELDS,
        max_horizon=MAX_HORIZON,
        random_subset_size=SAMPLE_SIZE,
        random_subset_seed=777,
    )

    print("Materializing input batches...")
    cached_batches = list(tqdm(
        materialise_batches(loader, device, NUM_VARIABLES, MAX_HORIZON, latitude, longitude),
        total=SAMPLE_SIZE,
        desc="Loading batches"
    ))

    print("Loading normalization statistics...")
    with open(DATA_DIRECTORY / "norm_factors.json", "r") as f:
        norm_stats = json.load(f)

    mean_data = torch.tensor([norm_stats[k]["mean"] for k in VARIABLE_NAMES], dtype=torch.float32)
    std_data = torch.tensor([norm_stats[k]["std"] for k in VARIABLE_NAMES], dtype=torch.float32)
    

    print("Loading full reference ERA5 dataset (memmap)...")
    full_array = np.memmap(
        DATA_DIRECTORY / "z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy",
        dtype=np.float32,
        mode='r',
        shape=(350640, NUM_VARIABLES, len(latitude), len(longitude))
    )

    print("Standardizing full ERA5 tensor...")
    full_tensor = torch.tensor(full_array, dtype=torch.float32)
    del full_array; gc.collect()

    full_tensor.sub_(mean_data[:, None, None]).div_(std_data[:, None, None])
    if device.type == "cuda":
        full_tensor = full_tensor.pin_memory()
    torch.cuda.empty_cache()

    print("Commencing ABC-Gibbs inference with RFP perturbations...")
    results = run_gibbs_abc_rfp(
        model=model,
        batches=cached_batches,
        ensemble_size=ENSEMBLE_SIZE,
        n_steps=N_GIBBS_STEPS,
        n_proposals=N_PROPOSALS_PER_VARIABLE,
        num_variables=NUM_VARIABLES,
        variable_names=VARIABLE_NAMES,
        max_horizon=MAX_HORIZON,
        reference_tensor=full_tensor,
    )
    del full_tensor
    gc.collect()
    torch.cuda.empty_cache()

    print("Saving posterior results...")
    save_posterior_statistics(results, result_path)

    print("Generating posterior plots...")
    produce_trace_and_histogram_plots(results["posterior_samples"], result_path, VARIABLE_NAMES, ["alpha_scale"])
    produce_rank_histograms(results["rank_histograms"], result_path, VARIABLE_NAMES, ENSEMBLE_SIZE)

    print("Final posterior parameter summary:")
    print_posterior_summary(results["posterior_mean"], results["posterior_variance"], VARIABLE_NAMES, ["alpha_scale"])

    print("ABC-Gibbs with RFP complete.")

    from core.plotting_rfp import plot_rfp_noise_field, plot_ensemble_statistics

    # Visualize perturbation field for variable 2 (e.g., t2m)
    print("Visualizing ensemble statistics at final Gibbs step...")
    final_step = -1
    final_ensemble = results["posterior_samples"][final_step]
    alpha_vec = final_ensemble[:, 0]

    from core.constants import VARIABLE_NAMES
    from core.plotting_rfp import plot_ensemble_statistics

    for var_idx, var_name in enumerate(VARIABLE_NAMES):
        alpha = alpha_vec[var_idx]
        example_field = cached_batches[0][0][0, var_idx]

        members = []
        for _ in range(ENSEMBLE_SIZE):
            τ1, τ2 = np.random.choice(full_tensor.shape[0], size=2, replace=False)
            delta_Y = full_tensor[τ1, var_idx] - full_tensor[τ2, var_idx]
            noise = alpha * delta_Y.to(example_field.device)
            perturbed = example_field + noise

            full_input = cached_batches[0][0].clone()
            variable_tensor = full_input[:, :-2]
            static_tensor = full_input[:, -2:]
            variable_tensor[0, var_idx] = perturbed
            input_tensor = torch.cat([variable_tensor, static_tensor], dim=1)

            output = model(input_tensor, cached_batches[0][2])
            members.append(output)

        ensemble_tensor = torch.stack(members, dim=0).squeeze(1)  # [E, V, H, W]
        target = cached_batches[0][1]

        std_field = ensemble_tensor.std(dim=0)[var_idx].detach().cpu().numpy()
        print(f"Std Dev [{var_name}]: min = {std_field.min():.4f}, max = {std_field.max():.4f}, mean = {std_field.mean():.4f}")

        plot_ensemble_statistics(
            ensemble_tensor,
            target,
            variable_index=var_idx,
            lat=latitude,
            lon=longitude,
            save_prefix=result_path / f"ensemble_stats_{var_name}"
        )
