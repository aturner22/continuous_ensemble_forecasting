#!/usr/bin/env python3
# SPDX‑License‑Identifier: MIT
#
# Post‑calibration verification suite for ABC‑Gibbs‑RFP.
#
# The routine assumes that:
#   1.  Posterior artefacts reside in `Config.result_directory`
#       and follow the naming convention specified in §1.
#   2.  Observational reference data are accessed exclusively
#       through memory‑mapped containers; no dense ERA5 tensors
#       are materialised.

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import json
from tqdm.auto import tqdm

# ────────────────────────────────────────────────────────────────────────────────
#  project‑specific imports
# ────────────────────────────────────────────────────────────────────────────────
from config import Config  # model‑agnostic configuration wrapper
from core.io_utils import (
    load_model_and_test_data,
    materialise_batches,
    print_computing_configuration,
)
from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)
from core.gibbs_abc_threaded_rfp import compute_safe_batch_size

# ═══════════════════════════════════════════════════════════════════════════════
#  mathematical utilities
# ═══════════════════════════════════════════════════════════════════════════════


def energy_score_vector(
    ensemble_members: torch.Tensor,  # [K, D]
    observation_vector: torch.Tensor,  # [D]
) -> torch.Tensor:
    distance_member_observation = (
        torch.linalg.norm(ensemble_members - observation_vector, dim=1).mean()
    )
    pairwise_distance = torch.cdist(
        ensemble_members, ensemble_members, p=2.0
    ).mean()  # Von Mises–Fisher formulation
    return distance_member_observation - 0.5 * pairwise_distance


def variogram_score_vector(
    ensemble_members: torch.Tensor, observation_vector: torch.Tensor, p: float = 0.5
) -> torch.Tensor:
    pairwise_difference_ensemble = (
        torch.abs(ensemble_members.unsqueeze(0) - ensemble_members.unsqueeze(1)) ** p
    ).mean()
    difference_observation = torch.abs(
        observation_vector.unsqueeze(0) - observation_vector.unsqueeze(1)
    ) ** p
    return (
        torch.abs(
            pairwise_difference_ensemble
            - difference_observation.mean()  # scalar after reduction
        )
        .mean()
        .sqrt()
    )

# ── helper ────────────────────────────────────────────────────────────────────
def _lead_time_to_tensor(
    lead_time_raw, *, device: torch.device, horizon: int
) -> torch.Tensor:
    """Return shape [N,1] float32 tensor normalised by `horizon`."""
    if torch.is_tensor(lead_time_raw):
        t = lead_time_raw.to(dtype=torch.float32, device=device)
    else:                              # list | int | float
        t = torch.tensor(lead_time_raw, dtype=torch.float32, device=device)
    if t.ndim == 0:                    # scalar → [1]
        t = t.unsqueeze(0)
    return (t / horizon).view(-1, 1)   # ensure column‑vector


# ═══════════════════════════════════════════════════════════════════════════════
#  ensemble generators
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def posterior_mean_ensemble(
    deterministic_network: torch.nn.Module,
    anterior_tensor: torch.Tensor,
    normalised_lead_time: torch.Tensor,
    posterior_mean_matrix: np.ndarray,
    reference_mmap: np.memmap,
    ensemble_cardinality: int,
    gpu_device: torch.device,
) -> torch.Tensor:
    """
    Generates an ensemble by perturbing the deterministic forecast
    with RFP noise parameterised by the posterior mean vector.
    """

    # deterministic backbone
    latent_forecast = deterministic_network(anterior_tensor, normalised_lead_time)   # [1,V,H,W]
    variable_count = latent_forecast.shape[1]
    spatial_height, spatial_width = latent_forecast.shape[-2:]

    posterior_mean_tensor = torch.tensor(
        posterior_mean_matrix.squeeze(-1), device=gpu_device
    )                                                                               # [V]
    reference_tensor = torch.from_numpy(np.array(reference_mmap)).to(gpu_device)    # [T,V,H,W]

    ensemble_tensor = torch.empty(
        ensemble_cardinality,
        variable_count,
        spatial_height,
        spatial_width,
        device=gpu_device,
    )

    idx1 = torch.randint(0, reference_tensor.shape[0], (ensemble_cardinality,), device=gpu_device)
    idx2 = torch.randint(0, reference_tensor.shape[0], (ensemble_cardinality,), device=gpu_device)

    for variable_index in range(variable_count):
        delta_field = reference_tensor[idx1, variable_index] - reference_tensor[idx2, variable_index]   # [K,H,W]
        ensemble_tensor[:, variable_index] = (
            latent_forecast[0, variable_index] + posterior_mean_tensor[variable_index] * delta_field
        )
    return ensemble_tensor


@torch.no_grad()
def posterior_sampled_ensemble(
    deterministic_network: torch.nn.Module,
    anterior_tensor: torch.Tensor,
    normalised_lead_time: torch.Tensor,
    posterior_samples_matrix: np.ndarray,
    reference_mmap: np.memmap,
    ensemble_cardinality: int,
    gpu_device: torch.device,
    random_state: np.random.Generator,
) -> torch.Tensor:
    # deterministic backbone
    latent_forecast = deterministic_network(anterior_tensor, normalised_lead_time)   # [1,V,H,W]
    variable_count = latent_forecast.shape[1]
    spatial_height, spatial_width = latent_forecast.shape[-2:]

    reference_tensor = torch.from_numpy(np.array(reference_mmap)).to(gpu_device)

    ensemble_tensor = torch.empty(
        ensemble_cardinality,
        variable_count,
        spatial_height,
        spatial_width,
        device=gpu_device,
    )

    idx1 = torch.randint(0, reference_tensor.shape[0], (ensemble_cardinality,), device=gpu_device)
    idx2 = torch.randint(0, reference_tensor.shape[0], (ensemble_cardinality,), device=gpu_device)

    sample_indices = random_state.integers(
        0, posterior_samples_matrix.shape[0], size=ensemble_cardinality
    )
    sampled_alphas = torch.tensor(
        posterior_samples_matrix[sample_indices, :, 0], device=gpu_device
    )  # [K,V]

    for variable_index in range(variable_count):
        delta_field = reference_tensor[idx1, variable_index] - reference_tensor[idx2, variable_index]   # [K,H,W]
        alpha = sampled_alphas[:, variable_index].view(-1, 1, 1)                                         # [K,1,1]
        ensemble_tensor[:, variable_index] = latent_forecast[0, variable_index] + alpha * delta_field
    return ensemble_tensor


@torch.no_grad()
def persistence_ensemble(
    anterior_tensor: torch.Tensor,
    variable_count: int,
    ensemble_cardinality: int,
) -> torch.Tensor:
    latest_slice = anterior_tensor[:, :variable_count]              # [1,V,H,W]
    return latest_slice.expand(ensemble_cardinality, -1, -1, -1)    # [K,V,H,W]


@torch.no_grad()
def climatology_gaussian_ensemble(
    variable_mean_tensor: torch.Tensor,  # [V, H, W]
    variable_std_tensor: torch.Tensor,  # [V, H, W]
    ensemble_cardinality: int,
) -> torch.Tensor:
    gaussian_noise = torch.randn(
        ensemble_cardinality,
        *variable_mean_tensor.shape,
        dtype=variable_mean_tensor.dtype,
        device=variable_mean_tensor.device,
    )
    return variable_mean_tensor.add(variable_std_tensor.mul(gaussian_noise))


@torch.no_grad()
def deterministic_plus_isotropic_noise_ensemble(
    deterministic_field: torch.Tensor,              # [1,V,H,W]  OR  [V,H,W]
    isotropic_variance_tensor: torch.Tensor,        # [V,H,W]
    ensemble_cardinality: int,
) -> torch.Tensor:
    base = deterministic_field.squeeze(0).expand(ensemble_cardinality, -1, -1, -1)
    noise = torch.randn_like(base).mul_(isotropic_variance_tensor.sqrt())
    return base + noise


@torch.no_grad()
def ar1_ensemble(
    previous_anomaly_tensor: torch.Tensor,          # [V,H,W]
    autoregressive_coefficient_tensor: torch.Tensor,# [V,1,1]
    white_noise_variance_tensor: torch.Tensor,      # [V,1,1]
    ensemble_cardinality: int,
) -> torch.Tensor:
    deterministic_part = (
        autoregressive_coefficient_tensor * previous_anomaly_tensor
    ).expand(ensemble_cardinality, -1, -1, -1)
    innovations = torch.randn_like(deterministic_part).mul_(
        white_noise_variance_tensor.sqrt()
    )
    return deterministic_part + innovations


# ═══════════════════════════════════════════════════════════════════════════════
#  main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        prog="evaluate_abc_gibbs_rfp",
        description="Posterior verification for ABC‑Gibbs‑RFP calibrated forecasts.",
    )
    argument_parser.add_argument(
        "--config", default="config.json", type=str, metavar="FILE"
    )
    argument_parser.add_argument("--seed", default=777, type=int)
    argument_parser.add_argument("--sample-size", default=1024, type=int)
    argument_parser.add_argument("--coverage-probability", default=0.9, type=float)
    args = argument_parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_computing_configuration()

    configuration = Config(args.config, timestamp=os.getenv("CONFIG_TIMESTAMP"))
    configuration.sample_size = args.sample_size  # override for evaluation run

    computing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        evaluation_loader,
        deterministic_network,
        latitude_vector,
        longitude_vector,
        result_directory,
    ) = load_model_and_test_data(configuration, computing_device, args.seed)

    # --- manual override of result_directory if required -----------------------
    result_directory = Path("/Users/ashleyturner/Development/imperial/msc_research_project/unet_abc/results/rfp_gibbs_abc_higher_hyperparams_2025-07-21T17:46:48Z")
    # --------------------------------------------------------------------------

    posterior_samples_matrix = np.load(result_directory / "posterior_samples.npy")
    posterior_mean_matrix = np.load(result_directory / "posterior_mean.npy")

    reference_mmap = np.load(
        configuration.data_directory / "z500_t850_t2m_u10_v10_standardized.npy",
        mmap_mode="r",
    )

    ensemble_cardinality = configuration.ensemble_size
    variable_count = configuration.num_variables

    # climatology statistics
    with open(configuration.data_directory / "norm_factors.json") as handle:
        climatology_stats = json.load(handle)
    variable_mean_tensor = torch.tensor(
        [climatology_stats[v]["mean"] for v in configuration.variable_names],
        dtype=torch.float32,
        device=computing_device,
    ).view(variable_count, 1, 1)
    variable_std_tensor = torch.tensor(
        [climatology_stats[v]["std"] for v in configuration.variable_names],
        dtype=torch.float32,
        device=computing_device,
    ).view(variable_count, 1, 1)

    # deterministic error variance for isotropic noise
    error_accumulator: List[torch.Tensor] = []
    for anterior_fields, posterior_fields, normalised_time in tqdm(
        evaluation_loader, desc="Calibrating isotropic variance", leave=False
    ):
        with torch.no_grad():
            deterministic_pred = deterministic_network(
                anterior_fields.to(computing_device),
                _lead_time_to_tensor(
                    normalised_time, device=computing_device, horizon=configuration.max_horizon
                ),
            )
        error_accumulator.append(
            (deterministic_pred - posterior_fields.to(computing_device)) ** 2
        )
    isotropic_variance_tensor = torch.vstack(error_accumulator).mean(dim=0)

    # AR(1) coefficient estimation
    prior_anomaly_list: List[torch.Tensor] = []
    posterior_anomaly_list: List[torch.Tensor] = []
    for anterior_fields, posterior_fields, _ in evaluation_loader:
        prior_anomaly_list.append(anterior_fields[:, :variable_count])  # drop static + -6 h slice
        posterior_anomaly_list.append(posterior_fields)
    prior_anomalies = torch.vstack(prior_anomaly_list).to(computing_device)
    posterior_anomalies = torch.vstack(posterior_anomaly_list).to(computing_device)
    numerator = (prior_anomalies * posterior_anomalies).sum(dim=(0, 2, 3))
    denominator = (prior_anomalies ** 2).sum(dim=(0, 2, 3)).clamp(min=1e-8)
    autoregressive_coefficient_tensor = (numerator / denominator).view(variable_count, 1, 1)
    residuals = posterior_anomalies - autoregressive_coefficient_tensor * prior_anomalies
    white_noise_variance_tensor = (
        residuals ** 2
    ).mean(dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)

    del prior_anomaly_list, posterior_anomaly_list, prior_anomalies, posterior_anomalies
    gc.collect()

    # metric accumulators
    metric_registry: Dict[str, Dict[str, float]] = {
        "posterior_mean": dict(),
        "posterior_sample": dict(),
        "persistence": dict(),
        "climatology": dict(),
        "deterministic_noise": dict(),
        "ar1": dict(),
    }
    reliability_rank_histogram: Dict[str, List[int]] = {
        key: [0] * (ensemble_cardinality + 1) for key in metric_registry.keys()
    }
    coverage_hits: Dict[str, int] = {key: 0 for key in metric_registry.keys()}
    sample_counter = 0

    random_state = np.random.default_rng(args.seed)

    for (
        anterior_fields,
        posterior_fields,
        normalised_time,
    ) in tqdm(evaluation_loader, desc="Verification", leave=False):
        anterior_tensor = anterior_fields.to(computing_device)
        posterior_tensor = posterior_fields.to(computing_device).squeeze(0)
        time_tensor = _lead_time_to_tensor(
            normalised_time, device=computing_device, horizon=configuration.max_horizon
        )

        # pre‑compute deterministic forecast once
        deterministic_baseline = deterministic_network(anterior_tensor, time_tensor)
    # ensure climatology tensors have the correct spatial shape
        H, W = posterior_tensor.shape[-2:]
        mu_map = variable_mean_tensor.expand(-1, H, W)      # [V,H,W]
        sd_map = variable_std_tensor.expand(-1, H, W)       # [V,H,W]


        # forecast collections
        ensemble_dictionaries: Dict[str, torch.Tensor] = dict()

        ensemble_dictionaries["posterior_mean"] = posterior_mean_ensemble(
            deterministic_network=deterministic_network,
            anterior_tensor=anterior_tensor,
            normalised_lead_time=time_tensor,
            posterior_mean_matrix=posterior_mean_matrix,
            reference_mmap=reference_mmap,
            ensemble_cardinality=ensemble_cardinality,
            gpu_device=computing_device,
        )

        ensemble_dictionaries["posterior_sample"] = posterior_sampled_ensemble(
            deterministic_network=deterministic_network,
            anterior_tensor=anterior_tensor,
            normalised_lead_time=time_tensor,
            posterior_samples_matrix=posterior_samples_matrix,
            reference_mmap=reference_mmap,
            ensemble_cardinality=ensemble_cardinality,
            gpu_device=computing_device,
            random_state=random_state,
        )

        ensemble_dictionaries["persistence"] = persistence_ensemble(
            anterior_tensor, variable_count, ensemble_cardinality
        )
        ensemble_dictionaries["climatology"] = climatology_gaussian_ensemble(
            mu_map,
            sd_map,ensemble_cardinality)


        ensemble_dictionaries[
            "deterministic_noise"
        ] = deterministic_plus_isotropic_noise_ensemble(
            deterministic_baseline,
            isotropic_variance_tensor,
            ensemble_cardinality,
        )

        anomaly_previous_field = anterior_tensor[:, :variable_count].squeeze(0)  # exclude static + -6 h
        ensemble_dictionaries["ar1"] = ar1_ensemble(
            anomaly_previous_field,
            autoregressive_coefficient_tensor,
            white_noise_variance_tensor,
            ensemble_cardinality,
        )

        # compute verification statistics online
        for method_key, ensemble_tensor in ensemble_dictionaries.items():
            # CRPS
            crps_tensor = continuous_ranked_probability_score(
                ensemble_tensor, posterior_tensor
            )
            metric_registry[method_key].setdefault("crps_sum", 0.0)
            metric_registry[method_key]["crps_sum"] += crps_tensor.mean().item()

            # MAE
            mae_value = compute_mean_absolute_error(ensemble_tensor, posterior_tensor)
            metric_registry[method_key].setdefault("mae_sum", 0.0)
            metric_registry[method_key]["mae_sum"] += mae_value

            # spread
            spread_value = compute_ensemble_spread(ensemble_tensor)
            metric_registry[method_key].setdefault("spread_sum", 0.0)
            metric_registry[method_key]["spread_sum"] += spread_value

            # PIT / rank‑histogram: aggregate counts into K + 1 bins in one shot
            rank_per_point = (ensemble_tensor
                              < posterior_tensor.unsqueeze(0)).sum(dim=0)      # [V,H,W]
            flat_ranks = rank_per_point.flatten().cpu().numpy().astype(int)    # [N_pts]
            counts = np.bincount(flat_ranks, minlength=ensemble_cardinality + 1)
            for k in range(ensemble_cardinality + 1):
                reliability_rank_histogram[method_key][k] += int(counts[k])

            # central prediction‑interval coverage
            lower_quantile = torch.quantile(
                ensemble_tensor, (1.0 - args.coverage_probability) / 2.0, dim=0
            )
            upper_quantile = torch.quantile(
                ensemble_tensor, 1.0 - (1.0 - args.coverage_probability) / 2.0, dim=0
            )
            coverage_hits[method_key] += int(
                torch.logical_and(
                    posterior_tensor >= lower_quantile,
                    posterior_tensor <= upper_quantile,
                ).all()
            )

            # Energy score and variogram score (spatial means)
            ensemble_spatial_mean = ensemble_tensor.mean(dim=(2, 3))  # [K, V]
            target_spatial_mean = posterior_tensor.mean(dim=(1, 2))  # [V]
            es_value = energy_score_vector(ensemble_spatial_mean, target_spatial_mean)
            vs_value = variogram_score_vector(
                ensemble_spatial_mean, target_spatial_mean
            )
            metric_registry[method_key].setdefault("energy_sum", 0.0)
            metric_registry[method_key].setdefault("variogram_sum", 0.0)
            metric_registry[method_key]["energy_sum"] += es_value.item()
            metric_registry[method_key]["variogram_sum"] += vs_value.item()

            # Brier score for extreme positive anomaly event (> 2σ)
            event_indicator = (posterior_tensor > 2.0).float()
            event_probability = (
                ensemble_tensor > 2.0
            ).float().mean()  # scalar probability
            brier_value = (event_probability - event_indicator).pow(2).mean().item()
            metric_registry[method_key].setdefault("brier_sum", 0.0)
            metric_registry[method_key]["brier_sum"] += brier_value

        sample_counter += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # normalise aggregates
    for method_key, statistic_dictionary in metric_registry.items():
        for aggregated_name in list(statistic_dictionary.keys()):
            statistic_dictionary[aggregated_name.replace("_sum", "")] = (
                statistic_dictionary.pop(aggregated_name) / sample_counter
            )
        statistic_dictionary["coverage"] = (
            coverage_hits[method_key] / sample_counter
        )
        # PIT histogram is stored separately

    # export results
    evaluation_output_path = result_directory / "evaluation"
    evaluation_output_path.mkdir(exist_ok=True)
    np.save(
        evaluation_output_path / "verification_metrics.npy",
        np.array(metric_registry, dtype=object),
    )
    np.save(
        evaluation_output_path / "pit_histograms.npy",
        np.array(reliability_rank_histogram, dtype=object),
    )

    # console summary
    print("\n────────── verification summary ──────────\n")
    for method_key, statistic_dictionary in metric_registry.items():
        print(f"{method_key.upper()}:")
        for metric_name, metric_value in statistic_dictionary.items():
            print(f"  {metric_name:<10s} : {metric_value: .6e}")
        print()
    print("───────────────────────────────────────────")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
