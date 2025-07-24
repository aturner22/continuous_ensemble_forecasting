import os, psutil, numpy as np, torch
from typing import Any
from tqdm import tqdm
from core.evaluation import (
    continuous_ranked_probability_score,
    compute_rank_histogram,
    compute_mean_absolute_error,
    compute_ensemble_spread,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INITIAL_ALPHA_RANGE = (0.05, 1.5)  
PROPOSAL_SCALE      = 0.05
MIN_ALPHA           = 1e-4
ADAPT_EVERY         = 5
ADAPT_FACTOR        = 0.85
EPS_ENERGY          = 1e-12
CHECKPOINT_FILE     = "gibbs_checkpoint_step.npz"

# --------------------------------------------------------------------------- #
def compute_safe_batch_size(
    ensemble_size: int,
    num_variables: int,
    spatial_height: int,
    spatial_width: int,
    num_outputs: int,
    dtype_bytes: int = 4,
    model_overhead: float = 2.0,
    safety_divisor: float = 10.0,
    available_bytes: int | None = None,
) -> int:
    """Heuristic outer batch size for forward passes."""
    if available_bytes is None:
        if torch.cuda.is_available():
            reserved  = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            total     = torch.cuda.get_device_properties(0).total_memory
            available_bytes = total - max(reserved, allocated)
        else:
            available_bytes = psutil.virtual_memory().available

    spatial = spatial_height * spatial_width
    tensor_bytes = (
        (num_variables + 2) * spatial
        + num_outputs * spatial
        + num_variables * ensemble_size * spatial * 3
    ) * dtype_bytes * model_overhead
    return max(1, int(available_bytes // (tensor_bytes * safety_divisor)))

# --------------------------------------------------------------------------- #
def generate_joint_rfp(
    reference_tensor: torch.Tensor,   # [T,V,H,W]
    alpha_matrix: torch.Tensor,       # [P,V] proposals (P>=1)
    batch_size: int,
    ensemble_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Joint random field perturbations for all proposals simultaneously.

    For each proposal p:
        diff_{b,k,v} = Y_{τ1} - Y_{τ2}
        E_{b,k} = sqrt(mean_{v,h,w} diff^2)
        ξ_{p,b,k,v} = α_{p,v} * diff_{b,k,v} / E_{b,k}

    Returns: [P, batch_size, ensemble_size, V, H, W]
    """
    P = alpha_matrix.shape[0]
    T, V = reference_tensor.shape[0], reference_tensor.shape[1]

    idx1 = torch.randint(0, T, (batch_size, ensemble_size), device=device, generator=generator)
    idx2 = torch.randint(0, T, (batch_size, ensemble_size), device=device, generator=generator)
    diff = reference_tensor[idx1] - reference_tensor[idx2]       # [B,K,V,H,W]

    # joint energy scalar per (B,K)
    energy = torch.sqrt(diff.pow(2).mean(dim=(2, 3, 4), keepdim=True).clamp_min(EPS_ENERGY))  # [B,K,1,1,1]
    diff_norm = diff / energy                               # [B,K,V,H,W]

    # Broadcast proposals: alpha_matrix -> [P,1,1,V,1,1]
    perturb = alpha_matrix.view(P, 1, 1, V, 1, 1) * diff_norm.unsqueeze(0)
    return perturb  # [P,B,K,V,H,W]

# --------------------------------------------------------------------------- #
def batched_forward_proposals(
    *,
    model: torch.nn.Module,
    previous_fields: torch.Tensor,   # [N,C,H,W]
    current_fields: torch.Tensor,    # [N,V,H,W]
    time_normalised: torch.Tensor,   # [N]
    reference_tensor: torch.Tensor,  # [T,V,H,W]
    alpha_matrix: torch.Tensor,      # [P,V] (each row a full alpha vector)
    ensemble_size: int,
    device: torch.device,
    buffers: dict,
    outer_bs: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Evaluate *all* proposals in alpha_matrix jointly.
    Returns ensemble forecasts of shape [P,K,N,V,H,W].
    """
    N, C, H, W = previous_fields.shape
    V = current_fields.shape[1]
    P = alpha_matrix.shape[0]

    current_slice = previous_fields[:, :V]
    past_slice    = previous_fields[:, V:2*V]
    static_slice  = previous_fields[:, -2:]

    # perturbations: [P,N,K,V,H,W]
    perturb = generate_joint_rfp(
        reference_tensor, alpha_matrix, batch_size=N, ensemble_size=ensemble_size,
        device=device, generator=generator
    )

    # Build per‑proposal inputs
    # buffers: curr:[N,K,V,H,W], past:[N,K,V,H,W], stat:[N,K,2,H,W]
    curr_base = current_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)  # [N,K,V,H,W]
    past_base = past_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)
    stat_base = static_slice.unsqueeze(1).expand(-1, ensemble_size, -1, -1, -1)

    outputs = []
    for p in range(P):
        buffers["curr"][:N].copy_(curr_base)
        buffers["curr"][:N].add_(perturb[p])  # add joint perturbation
        buffers["past"][:N].copy_(past_base)
        buffers["stat"][:N].copy_(stat_base)

        full_input = torch.cat([buffers['curr'][:N], buffers['past'][:N], buffers['stat'][:N]], dim=2)
        full_input = full_input.view(N * ensemble_size, C, H, W)
        full_time  = time_normalised.view(-1, 1).expand(-1, ensemble_size).reshape(-1, 1)

        step = outer_bs * ensemble_size
        out_chunks = []
        for start in range(0, N * ensemble_size, step):
            end = min(start + step, N * ensemble_size)
            with torch.no_grad():
                y = model(full_input[start:end], full_time[start:end])
            out_chunks.append(y)
        y_full = torch.cat(out_chunks, dim=0).view(N, ensemble_size, V, H, W)
        outputs.append(y_full.permute(1, 0, 2, 3, 4))  # [K,N,V,H,W]

    return torch.stack(outputs, dim=0)  # [P,K,N,V,H,W]

# --------------------------------------------------------------------------- #
def run_gibbs_abc_rfp(
    *,
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ensemble_size: int,
    n_steps: int,
    n_proposals: int,
    num_variables: int,
    variable_names: list[str],
    max_horizon: int,
    reference_mmap: np.memmap,
    result_directory: str,
    log_diagnostics: bool = True,
) -> dict[str, Any]:
    """
    Gibbs sampling over alpha vector with joint CRPS objective.
    Each coordinate update proposes multiple alpha_v values; for each proposal
    we build a full alpha vector and evaluate **joint mean CRPS** across all variables.
    """
    device   = next(model.parameters()).device
    ref_full = torch.from_numpy(np.array(reference_mmap, copy=True)).to(device)  # [T,V,H,W]

    prev_all = torch.cat([b[0] for b in batches], dim=0).to(device)
    curr_all = torch.cat([b[1] for b in batches], dim=0).to(device)
    time_all = torch.cat([b[2] for b in batches], dim=0).to(device)
    N, C, H, W = prev_all.shape
    V = curr_all.shape[1]

    buffers = {
        "curr": torch.empty((N, ensemble_size, V, H, W), device=device),
        "past": torch.empty((N, ensemble_size, V, H, W), device=device),
        "stat": torch.empty((N, ensemble_size, 2, H, W), device=device),
    }

    outer_bs = compute_safe_batch_size(
        ensemble_size=ensemble_size,
        num_variables=C,
        spatial_height=H,
        spatial_width=W,
        num_outputs=V,
    )
    print(f"[device warm‑up] outer batch = {outer_bs}")
    with torch.no_grad():
        _ = model(prev_all[:1], time_all[:1])

    # Posterior storage
    posterior_samples = np.zeros((n_steps, num_variables, 1), dtype=np.float32)
    posterior_crps    = np.zeros((n_steps, num_variables),    dtype=np.float32)
    step_mean_crps    = np.zeros(n_steps, dtype=np.float32)

    # Diagnostics
    rank_histograms             = [[] for _ in range(num_variables)]
    ensemble_spread_records     = [[] for _ in range(num_variables)]
    mean_absolute_error_records = [[] for _ in range(num_variables)]

    # Initial alpha & proposal std
    current_alpha = np.random.uniform(*INITIAL_ALPHA_RANGE, size=(num_variables, 1))
    proposal_std  = np.full((num_variables, 1), PROPOSAL_SCALE, dtype=np.float32)

    rng       = np.random.default_rng()
    torch_gen = torch.Generator(device=device)

    # Resume from checkpoint if exists
    ckpt_path = os.path.join(result_directory, CHECKPOINT_FILE)
    start_step = 0
    if os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        print(f"[checkpoint] Resuming from step {ck['step']+1}")
        posterior_samples[:ck["step"]+1] = ck["posterior_samples"]
        posterior_crps[:ck["step"]+1]    = ck["posterior_crps"]
        step_mean_crps[:ck["step"]+1]    = ck["step_mean_crps"]
        current_alpha                    = ck["last_alpha"]
        start_step                       = int(ck["step"]) + 1
        del ck

    for s in tqdm(range(start_step, n_steps), desc="Gibbs", position=0):
        print(f"\n[Gibbs step {s+1}/{n_steps}]")
        if s and (s % ADAPT_EVERY == 0):
            proposal_std *= ADAPT_FACTOR
            print(f"[adapt] proposal σ -> {proposal_std.mean():.3f}")

        for v in range(num_variables):
            # Build proposal matrix for coordinate v
            proposals_v = np.clip(
                rng.normal(loc=current_alpha[v], scale=proposal_std[v], size=(n_proposals, 1)),
                MIN_ALPHA, None
            )
            # Alpha matrix for all proposals: start from current, replace column v
            alpha_mat = np.repeat(current_alpha.squeeze(-1)[None, :], n_proposals, axis=0)
            alpha_mat[:, v] = proposals_v.squeeze(-1)

            alpha_tensor = torch.tensor(alpha_mat, device=device, dtype=torch.float32)  # [P,V]

            torch_gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
            ens_prop = batched_forward_proposals(
                model=model,
                previous_fields=prev_all,
                current_fields=curr_all,
                time_normalised=time_all,
                reference_tensor=ref_full,
                alpha_matrix=alpha_tensor,
                ensemble_size=ensemble_size,
                device=device,
                buffers=buffers,
                outer_bs=outer_bs,
                generator=torch_gen,
            )  # [P,K,N,V,H,W]

            # Compute joint mean CRPS for each proposal
            # Loop over variables to avoid huge memory for pairwise distances simultaneously
            joint_scores = []
            for p in range(n_proposals):
                crps_vars = []
                for j in range(num_variables):
                    crps_pj = continuous_ranked_probability_score(
                        ens_prop[p, :, :, j], curr_all[:, j]
                    ).mean()
                    crps_vars.append(crps_pj)
                joint_scores.append(torch.stack(crps_vars).mean().item())
            joint_scores = np.array(joint_scores)

            # Select best proposal (min joint CRPS)
            best_idx = int(joint_scores.argmin())
            current_alpha[v] = proposals_v[best_idx]
            posterior_samples[s, v] = current_alpha[v]
            posterior_crps[s, v]    = joint_scores[best_idx]  # store joint score replicated per variable

            print(f" {variable_names[v]:5s} α*={current_alpha[v,0]:.3f}  jointCRPS={joint_scores[best_idx]: .4f}")

            if log_diagnostics:
                # Use ensembles from best proposal for per-variable diagnostics
                best_members = ens_prop[best_idx]  # [K,N,V,H,W]
                for j in range(num_variables):
                    if j == v:  # update diag for changed coord; optional: all j
                        spread_val = compute_ensemble_spread(best_members[:, :, j].cpu())
                        mae_val    = compute_mean_absolute_error(
                            best_members[:, :, j].cpu(), curr_all[:, j].cpu()
                        )
                        ensemble_spread_records[j].append(spread_val)
                        mean_absolute_error_records[j].append(mae_val)
                        ranks = compute_rank_histogram(
                            best_members[:, :, j], curr_all[:, j], ensemble_size
                        )
                        rank_histograms[j].extend(ranks.tolist())

        step_mean_crps[s] = posterior_crps[s].mean()
        print(f"⇒ mean joint CRPS (all vars) = {step_mean_crps[s]:.4f}")

        # Checkpoint
        np.savez_compressed(
            ckpt_path,
            step=s,
            posterior_samples=posterior_samples,
            posterior_crps=posterior_crps,
            step_mean_crps=step_mean_crps,
            last_alpha=current_alpha,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cleanup checkpoint on success
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {
        "posterior_samples": posterior_samples,
        "posterior_crps": posterior_crps,
        "posterior_mean": posterior_samples.mean(axis=0),
        "posterior_variance": posterior_samples.var(axis=0),
        "rank_histograms": rank_histograms,
        "ensemble_mae": np.array(mean_absolute_error_records, dtype=np.float32),
        "ensemble_spread": np.array(ensemble_spread_records, dtype=np.float32),
        "step_mean_crps": step_mean_crps,
    }
