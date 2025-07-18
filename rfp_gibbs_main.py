if __name__ == "__main__":
    import json
    import gc
    import numpy as np
    import torch
    from tqdm import tqdm
    from config import Config

    from core.io_utils import load_model_and_test_data, save_posterior_statistics, materialise_batches, print_computing_configuration
    from core.plotting import produce_trace_and_histogram_plots, produce_rank_histograms, plot_crps_trace
    from core.evaluation import print_posterior_summary
    from core.gibbs_abc_threaded_rfp import run_gibbs_abc_rfp

    print("Initializing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_computing_configuration()
    print(f"Using device: {device}")
    config = Config("config.json")
    print("Preparing model and data loader...")
    loader, model, latitude, longitude, result_path = load_model_and_test_data(config, device,)

    print("Materializing input batches...")
    cached_batches = list(tqdm(
        materialise_batches(loader, device, config.num_variables, config.max_horizon, latitude, longitude),
        total=config.sample_size,
        desc="Loading batches"
    ))

    print("Loading normalization statistics...")
    with open(config.data_directory / "norm_factors.json", "r") as f:
        norm_stats = json.load(f)

    mean_data = torch.tensor([norm_stats[k]["mean"] for k in config.variable_names], dtype=torch.float32)
    std_data = torch.tensor([norm_stats[k]["std"] for k in config.variable_names], dtype=torch.float32)

    standardized_path = config.data_directory / "z500_t850_t2m_u10_v10_standardized.npy"
    raw_path = config.data_directory / "z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy"

    if standardized_path.exists():
        print("Loading precomputed standardized ERA5 tensor...")
        full_array = np.load(standardized_path, mmap_mode='r')
        full_tensor = torch.from_numpy(full_array)
    else:
        print("Standardized tensor not found. Processing raw ERA5 dataset...")

        total_timesteps = 350640
        spatial_shape = (len(latitude), len(longitude))
        shape = (total_timesteps, config.num_variables, *spatial_shape)

        raw_array = np.memmap(raw_path, dtype=np.float32, mode='r', shape=shape)
        full_tensor = torch.tensor(raw_array, dtype=torch.float32)
        del raw_array
        gc.collect()

        print("Standardizing tensor...")
        full_tensor.sub_(mean_data[:, None, None]).div_(std_data[:, None, None])

        print("Saving standardized tensor for future runs...")
        np.save(standardized_path, full_tensor.cpu().numpy())

    if device.type == "cuda":
        full_tensor = full_tensor.pin_memory()
    torch.cuda.empty_cache()

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
            reference_tensor=full_tensor,
        )

        print("Saving posterior results...")
        save_posterior_statistics(results, result_path)

        # Proactive memory cleanup before plotting
        print("Releasing memory before plotting...")
        torch.cuda.empty_cache()
        del full_tensor
        del mean_data
        del std_data
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
        gc.collect()

        produce_rank_histograms(
            results_to_keep["rank_histograms"],
            result_path,
            config.variable_names,
            config.ensemble_size
        )
        gc.collect()

        plot_crps_trace(
            results_to_keep["step_mean_crps"],
            result_path
        )
        gc.collect()

        print("Final posterior parameter summary:")
        print_posterior_summary(
            results_to_keep["posterior_mean"],
            results_to_keep["posterior_variance"],
            config.variable_names,
            ["alpha_scale"]
        )

        print("ABC-Gibbs with RFP complete.")

        # print("Visualizing ensemble statistics at final Gibbs step...")
        # final_step = -1
        # final_ensemble = results["posterior_samples"][final_step]
        # alpha_vec = final_ensemble[:, 0]

        # with torch.no_grad():
        #     for var_idx, var_name in enumerate(VARIABLE_NAMES):
        #         alpha = alpha_vec[var_idx]
        #         example_field = cached_batches[0][0][0, var_idx]

        #         members = []
        #         for _ in range(ENSEMBLE_SIZE):
        #             τ1, τ2 = np.random.choice(full_tensor.shape[0], size=2, replace=False)
        #             delta_Y = full_tensor[τ1, var_idx] - full_tensor[τ2, var_idx]
        #             noise = alpha * delta_Y.to(example_field.device)
        #             perturbed = example_field + noise

        #             full_input = cached_batches[0][0].clone()
        #             variable_tensor = full_input[:, :-2]
        #             static_tensor = full_input[:, -2:]
        #             variable_tensor[0, var_idx] = perturbed
        #             input_tensor = torch.cat([variable_tensor, static_tensor], dim=1)

        #             output = model(input_tensor, cached_batches[0][2])
        #             members.append(output)

        #         ensemble_tensor = torch.stack(members, dim=0).squeeze(1)  # [E, V, H, W]
        #         target = cached_batches[0][1]

        #         std_field = ensemble_tensor.std(dim=0)[var_idx].detach().cpu().numpy()
        #         print(f"Std Dev [{var_name}]: min = {std_field.min():.4f}, max = {std_field.max():.4f}, mean = {std_field.mean():.4f}")

        #         plot_ensemble_statistics(
        #             ensemble_tensor,
        #             target,
        #             variable_index=var_idx,
        #             lat=latitude,
        #             lon=longitude,
        #             save_prefix=result_path / f"ensemble_stats_{var_name}"
        #         )

    finally:
        # del full_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
