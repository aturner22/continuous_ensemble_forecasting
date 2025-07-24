#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# ABC–Gibbs–RFP posterior verification (fast variant).
# – single deterministic forward pass per sample
# – CPU thread-pool / GPU autocast
# – energy‑normalised RFP (matches training Gibbs)
# – variable-wise diagnostics, PIT histograms, plots
# – coverage metrics disabled (can be re‑enabled easily)

from __future__ import annotations
import argparse, os, random, time, json
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from core.io_utils import load_model_and_test_data, print_computing_configuration
from core.evaluation import compute_mean_absolute_error, compute_ensemble_spread

# ─────────────────────────── helpers ───────────────────────────

def _lead_time(raw, *, dev: torch.device, horizon: int) -> torch.Tensor:
    t = raw if torch.is_tensor(raw) else torch.tensor(raw, dtype=torch.float32)
    return (t.to(dev).view(-1) / horizon).view(-1, 1)

def _crps(e: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    # energy score surrogate for CRPS (spatial dimensions averaged)
    return (e - o).abs().mean() - 0.5 * torch.cdist(e.flatten(1).T, e.flatten(1).T).mean() / e[0].numel()

def _energy(e: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    ef, of = e.view(e.shape[0], -1), o.view(-1)
    return torch.cdist(ef, of.unsqueeze(0)).mean() - 0.5 * torch.cdist(ef, ef).mean()

def _variogram(e: torch.Tensor, o: torch.Tensor, p: float = .5) -> torch.Tensor:
    ef, of = e.view(e.shape[0], -1), o.view(-1)
    return ((ef.unsqueeze(0) - ef.unsqueeze(1)).abs().pow(p).mean()
            - (of.unsqueeze(0) - of.unsqueeze(1)).abs().pow(p).mean()).abs().sqrt()

def _rfp(alpha: torch.Tensor, ref: torch.Tensor, i1: torch.Tensor, i2: torch.Tensor):
    """
    Energy‑normalised random field perturbations applied to *all* variables.
    alpha: [K,V] (already expanded for posterior_mean or posterior_sample)
    ref[i] : [K,V,H,W]
    """
    diff = ref[i1] - ref[i2]                           # [K,V,H,W]
    energy = torch.sqrt(diff.pow(2).mean(dim=(2,3), keepdim=True).clamp_min(1e-12))
    return alpha.unsqueeze(-1).unsqueeze(-1) * diff / energy

# ───────────────────────── ensembles ──────────────────────────

def build_ens(net, x0, t, a_mean, a_samp, ref, mu, sig, iso, phi, eps, K, g, dev):
    """
    Construct all evaluation ensembles.
    Layout of x0: [1, (curr[V], past[V], static[2]), H, W]
    """
    T, V, H, W = ref.shape[0], mu.shape[0], ref.shape[-2], ref.shape[-1]

    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        base = net(x0, t).expand(K, -1, -1, -1)        # [K,V,H,W]

    # joint time indices for perturbations
    i1 = torch.randint(0, T, (K,), device=dev, generator=g)
    i2 = torch.randint(0, T, (K,), device=dev, generator=g)

    # posterior mean ensemble
    ens_pm = base + _rfp(a_mean.expand(K, -1), ref, i1, i2)

    # posterior sample ensemble
    idx = torch.randint(0, a_samp.shape[0], (K,), device=dev, generator=g)
    ens_ps = base + _rfp(a_samp[idx], ref, i1, i2)

    # persistence (current state)
    Vslice = x0[:, :V]
    ens_pers = Vslice.expand(K, -1, -1, -1)

    # climatology (μ=0, σ=1 under anomaly standardisation)
    ens_clim = mu + sig * torch.randn(K, V, H, W, device=dev, generator=g)

    # deterministic + isotropic residual noise
    ens_iso = base + torch.randn(base.shape, device=dev, dtype=base.dtype, generator=g) * iso.sqrt()

    # AR(1) anomaly model: previous anomalies in x0 are slice [:V]
    prev = x0[:, :V].squeeze(0)
    ens_ar1 = phi * prev + torch.randn(K, V, H, W, device=dev, generator=g) * eps.sqrt()

    return {
        "posterior_mean": ens_pm,
        "posterior_sample": ens_ps,
        "persistence": ens_pers,
        "climatology": ens_clim,
        "deterministic_noise": ens_iso,
        "ar1": ens_ar1,
    }

# ───────────────────── per-sample verification ────────────────

def verify(x0, y, t_raw, shared, cfg, dev, seed):
    net, a_mean, a_samp, ref, mu, sig, iso, phi, eps = shared
    g = torch.Generator(device=dev).manual_seed(seed ^ (time.time_ns() & 0xFFFFFFFF))
    x0, y = x0.to(dev), y.to(dev).squeeze(0)
    ens = build_ens(net, x0, _lead_time(t_raw, dev=dev, horizon=cfg.max_horizon),
                    a_mean, a_samp, ref, mu, sig, iso, phi, eps,
                    cfg.ensemble_size, g, dev)

    metrics, pit, per_var = {}, {}, {}
    for k, e in ens.items():
        metrics[k] = dict(
            crps=_crps(e, y).item(),
            mae=compute_mean_absolute_error(e, y),
            spread=compute_ensemble_spread(e),
            energy=_energy(e, y).item(),
            variogram=_variogram(e, y).item(),
            brier=((e > 2).float().mean() - (y > 2).float()).pow(2).mean().item(),
        )
        pit[k] = np.bincount((e < y).sum(dim=0).flatten().cpu().numpy(),
                             minlength=cfg.ensemble_size + 1)

        # Variable‑wise diagnostics
        ens_mean = e.mean(0).cpu()                      # [V,H,W]
        per_var[k] = dict(
            bias=(ens_mean - y.cpu()).mean(dim=(1, 2)),
            mse=((ens_mean - y.cpu()) ** 2).mean(dim=(1, 2)),
            var=e.var(dim=0).cpu().mean(dim=(1, 2)),
        )
    return metrics, pit, per_var

# ───────────────────────── plotting ───────────────────────────

def plot_rank_histograms(pit_hist: Dict[str, np.ndarray], out_dir: Path):
    for k, counts in pit_hist.items():
        fig, ax = plt.subplots(figsize=(6, 3))
        p = counts / counts.sum()
        ax.bar(np.arange(len(p)), p, edgecolor="k", linewidth=.4)
        ax.set_title(f"PIT histogram – {k}")
        ax.set_xlabel("Rank"); ax.set_ylabel("Frequency")
        ax.set_ylim(0, 1.1 * p.max())
        fig.tight_layout()
        fig.savefig(out_dir / f"pit_{k}.png", dpi=200)
        plt.close(fig)

def plot_bias_rmse_spread(bias, rmse, spread, variable_names, out_dir: Path):
    methods = list(bias.keys())
    V = len(variable_names)
    for stat_name, tensor_dict in [("bias", bias), ("rmse", rmse), ("spread", spread)]:
        fig, ax = plt.subplots(figsize=(1.5 + 1.2 * V, 3.5))
        data = np.vstack([tensor_dict[m].numpy() for m in methods])
        cmap = "coolwarm" if stat_name == "bias" else "viridis"
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
        ax.set_xticks(range(V)); ax.set_xticklabels(variable_names, rotation=45, ha="right")
        ax.set_title(stat_name)
        fig.colorbar(im, ax=ax, orientation="vertical", shrink=.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stat_name}_matrix.png", dpi=200)
        plt.close(fig)

    # spread / RMSE ratio
    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * V, 3.5))
    ratios = np.vstack([(spread[m] / rmse[m].clamp_min(1e-8)).numpy() for m in methods])
    im = ax.imshow(ratios, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    ax.set_xticks(range(V)); ax.set_xticklabels(variable_names, rotation=45, ha="right")
    ax.set_title("Spread / RMSE")
    fig.colorbar(im, ax=ax, orientation="vertical", shrink=.8)
    fig.tight_layout()
    fig.savefig(out_dir / "spread_rmse_ratio.png", dpi=200)
    plt.close(fig)

# ─────────────────────────── main ─────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="config.json")
    pa.add_argument("--sample-size", type=int, default=1024)
    pa.add_argument("--seed", type=int, default=777)
    pa.add_argument("--cpu-workers", type=int, default=os.cpu_count())
    args = pa.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    print_computing_configuration()

    cfg = Config(args.config, timestamp=os.getenv("CONFIG_TIMESTAMP"))
    cfg.sample_size = args.sample_size
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, net, _, _, res_dir = load_model_and_test_data(cfg, dev, args.seed)
    # Optional manual override:
    res_dir = Path("/Users/ashleyturner/Development/imperial/msc_research_project/unet_abc/results/GREAT_RUN_new_rfp_abc_with_energy_scaling_2025-07-21T23:35:02Z")

    ps_np = np.load(res_dir / "posterior_samples.npy")[:, :, 0]   # [S,V]
    pm_np = np.load(res_dir / "posterior_mean.npy")[:, 0]         # [V]

    ref = torch.from_numpy(np.load(
        cfg.data_directory / "z500_t850_t2m_u10_v10_standardized.npy",
        mmap_mode="r"
    )).to(dev)

    V = cfg.num_variables
    # Under anomaly normalisation climatology mean=0, std=1
    mu = torch.zeros(V, 1, 1, device=dev)
    sig = torch.ones_like(mu)

    net = net.half().eval() if dev.type == "cuda" else net.eval()

    # isotropic residual variance
    iso = torch.vstack([
        (net(x.to(dev), _lead_time(t, dev=dev, horizon=cfg.max_horizon)) - y.to(dev)) ** 2
        for x, y, t in loader
    ]).mean(0)

    # AR(1) coefficients
    prev, nxt = zip(*[(x[:, :V], y) for x, y, _ in loader])
    prev, nxt = torch.vstack(prev).to(dev), torch.vstack(nxt).to(dev)
    phi = ((prev * nxt).sum((0, 2, 3)) / (prev ** 2).sum((0, 2, 3)).clamp_min(1e-8)).view(V, 1, 1)
    eps = ((nxt - phi * prev) ** 2).mean((0, 2, 3), keepdim=True).clamp_min(1e-8)

    shared = (
        net,
        torch.tensor(pm_np, device=dev),
        torch.tensor(ps_np, device=dev),
        ref, mu, sig, iso, phi, eps
    )

    method_keys = ("posterior_mean", "posterior_sample", "persistence",
                   "climatology", "deterministic_noise", "ar1")

    from collections import defaultdict
    metrics = {k: defaultdict(float) for k in method_keys}
    pit_all = {k: np.zeros(cfg.ensemble_size + 1, int) for k in method_keys}
    bias_sum = {k: torch.zeros(V, device="cpu") for k in method_keys}
    mse_sum = {k: torch.zeros(V, device="cpu") for k in method_keys}
    var_sum = {k: torch.zeros(V, device="cpu") for k in method_keys}

    items = list(loader)

    def worker(idx_item):
        idx, (x, y, t) = idx_item
        return verify(x, y, t, shared, cfg, torch.device("cpu"), args.seed + idx)

    from concurrent.futures import ThreadPoolExecutor
    pool_ctx = ThreadPoolExecutor(max_workers=args.cpu_workers) if dev.type == "cpu" else None
    iterator = (
        pool_ctx.map(worker, enumerate(items))
        if pool_ctx else
        (verify(x, y, t, shared, cfg, dev, args.seed + i) for i, (x, y, t) in enumerate(items))
    )

    with pool_ctx or (lambda: (_ for _ in []))():
        for m, pit, per_var in tqdm(iterator, total=len(items), desc="verify", leave=False):
            for k, d in m.items():
                for kk, v in d.items():
                    metrics[k][kk] += v
            for k in pit:
                pit_all[k] += pit[k]
                bias_sum[k] += per_var[k]["bias"]
                mse_sum[k] += per_var[k]["mse"]
                var_sum[k] += per_var[k]["var"]

    n = len(items)
    for k, d in metrics.items():
        for kk in list(d.keys()):
            d[kk] /= n
        # Coverage intentionally disabled
        # d["coverage"] = cov_all[k] / n

    bias_mean = {k: bias_sum[k] / n for k in method_keys}
    rmse_mean = {k: (mse_sum[k] / n).sqrt() for k in method_keys}
    spread_mean = {k: (var_sum[k] / n).sqrt() for k in method_keys}

    out_dir = res_dir / "evaluation"; out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "verification_metrics.npy", np.array(metrics, dtype=object))
    np.save(out_dir / "pit_histograms.npy", np.array(pit_all, dtype=object))
    np.save(out_dir / "bias_rmse_spread.npy",
            np.array(dict(bias=bias_mean, rmse=rmse_mean, spread=spread_mean), dtype=object))

    print("\n── Verification summary ─────────────────────────────────────")
    for k, d in metrics.items():
        print(k.upper())
        for kk, v in d.items():
            print(f"  {kk:<10}: {v: .4e}")
        print()
    print("──────────────────────────────────────────────────────────────\n")

    print("── Diagnostics by variable ─────────────────────────────────")
    for k in method_keys:
        print(f"\n{k.upper():>18s}")
        for v_idx, var_name in enumerate(cfg.variable_names):
            b = bias_mean[k][v_idx].item()
            r = rmse_mean[k][v_idx].item()
            s = spread_mean[k][v_idx].item()
            ratio = s / r if r > 0 else float("nan")
            print(f"  {var_name:<8s} bias={b:+.3e} RMSE={r:.3e} σ={s:.3e} σ/RMSE={ratio:.2f}")
    print("────────────────────────────────────────────────────────────\n")

    for k in method_keys:
        p = pit_all[k] / pit_all[k].sum()
        print(f"{k:>18s} PIT first={p[0]:.3f} centre={p[len(p)//2]:.3f} last={p[-1]:.3f}")
    print()

    plot_rank_histograms(pit_all, out_dir)
    plot_bias_rmse_spread(bias_mean, rmse_mean, spread_mean, cfg.variable_names, out_dir)

if __name__ == "__main__":
    torch.no_grad().__enter__()
    main()
