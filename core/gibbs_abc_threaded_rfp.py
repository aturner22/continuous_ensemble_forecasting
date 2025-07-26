import os
import psutil
import numpy as np
import torch
from typing import Any
from tqdm import tqdm
from core.evaluation import (
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
class DynamicBatchManager:
    """Intelligent dynamic batch size management based on actual GPU memory usage."""
    
    def __init__(self, device, initial_batch_size: int = 64, min_batch_size: int = 1):
        self.device = device
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_successful_batch_size = initial_batch_size
        self.memory_history = []
        
    def get_memory_stats(self):
        """Get current GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            free = total - allocated
            return allocated, reserved, total, free
        return 0, 0, psutil.virtual_memory().total, psutil.virtual_memory().available
    
    def try_batch_size(self, batch_size: int, test_fn):
        """Test if a batch size works by running a test function."""
        try:
            allocated_before, _, _, _ = self.get_memory_stats()
            result = test_fn(batch_size)
            allocated_after, _, _, _ = self.get_memory_stats()
            
            # Record successful batch size and memory usage
            memory_used = allocated_after - allocated_before
            self.memory_history.append((batch_size, memory_used))
            self.max_successful_batch_size = max(self.max_successful_batch_size, batch_size)
            
            return result, True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return None, False
            else:
                raise e
    
    def find_optimal_batch_size(self, test_fn, target_batch_size: int = None):
        """Binary search to find the largest batch size that fits in memory."""
        if target_batch_size is None:
            target_batch_size = self.current_batch_size
            
        # Start with conservative estimate
        low = self.min_batch_size
        high = min(target_batch_size * 2, 1024)  # reasonable upper bound
        
        best_batch_size = self.min_batch_size
        
        print(f"[DynamicBatch] Finding optimal batch size between {low} and {high}")
        
        while low <= high:
            mid = (low + high) // 2
            print(f"[DynamicBatch] Testing batch size: {mid}")
            
            _, success = self.try_batch_size(mid, test_fn)
            
            if success:
                best_batch_size = mid
                low = mid + 1  # Try larger
                print(f"[DynamicBatch] ✓ Batch size {mid} succeeded")
            else:
                high = mid - 1  # Try smaller
                print(f"[DynamicBatch] ✗ Batch size {mid} failed (OOM)")
        
        self.current_batch_size = best_batch_size
        print(f"[DynamicBatch] Optimal batch size found: {best_batch_size}")
        return best_batch_size
    
    def predict_safe_batch_size(self, target_samples: int):
        """Predict safe batch size based on memory history and target load."""
        if not self.memory_history:
            return self.current_batch_size
            
        # Get average memory per sample from history
        recent_history = self.memory_history[-5:]  # Use recent measurements
        if recent_history:
            avg_memory_per_sample = sum(mem / batch for batch, mem in recent_history) / len(recent_history)
            _, _, total_memory, free_memory = self.get_memory_stats()
            
            # Reserve 20% of total memory for safety
            usable_memory = min(free_memory, total_memory * 0.8)
            predicted_batch_size = int(usable_memory / (avg_memory_per_sample * target_samples))
            
            # Clamp to reasonable bounds
            predicted_batch_size = max(self.min_batch_size, min(predicted_batch_size, self.max_successful_batch_size))
            
            print(f"[DynamicBatch] Predicted safe batch size: {predicted_batch_size} "
                  f"(free: {free_memory/1e9:.1f}GB, avg per sample: {avg_memory_per_sample/1e6:.1f}MB)")
            
            return predicted_batch_size
        
        return self.current_batch_size

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
    """Conservative initial batch size estimate - will be refined by DynamicBatchManager."""
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
    
    # Start conservative and let DynamicBatchManager optimize
    initial_estimate = max(1, int(available_bytes // (tensor_bytes * safety_divisor * 4)))
    return min(initial_estimate, 16)  # Very conservative starting point

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
def compute_crps_for_proposal(
    ensemble_output: torch.Tensor,  # [K,N,V,H,W]
    target: torch.Tensor,          # [N,V,H,W]
    num_variables: int,
) -> float:
    """Compute joint CRPS score for a single proposal to avoid memory accumulation."""
    from core.evaluation import continuous_ranked_probability_score
    
    crps_vars = []
    for j in range(num_variables):
        crps_pj = continuous_ranked_probability_score(
            ensemble_output[:, :, j], target[:, j]
        ).mean()
        crps_vars.append(crps_pj)
    return torch.stack(crps_vars).mean().item()

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
    batch_manager: DynamicBatchManager,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[float]]:
    """
    Evaluate proposals one by one to avoid large memory accumulation.
    Returns best proposal output [K,N,V,H,W] and all CRPS scores.
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

    def test_batch_size(test_batch_size):
        """Test function for dynamic batch size detection."""
        # Test with a small subset first
        test_n = min(N, 2)  # Use only 2 samples for testing
        
        test_input = torch.cat([
            curr_base[:test_n], 
            past_base[:test_n], 
            stat_base[:test_n]
        ], dim=2)
        test_input = test_input.view(test_n * ensemble_size, C, H, W)
        test_time = time_normalised[:test_n].view(-1, 1).expand(-1, ensemble_size).reshape(-1, 1)
        
        # Test with the proposed batch size
        step = test_batch_size * ensemble_size
        with torch.no_grad():
            test_end = min(step, test_n * ensemble_size)
            _ = model(test_input[:test_end], test_time[:test_end])
        
        return test_batch_size

    # Find optimal batch size using dynamic manager
    if not hasattr(batch_manager, '_calibrated_for_this_config'):
        print(f"[DynamicBatch] Calibrating for N={N}, ensemble_size={ensemble_size}")
        initial_guess = batch_manager.predict_safe_batch_size(ensemble_size)
        optimal_batch_size = batch_manager.find_optimal_batch_size(test_batch_size, initial_guess)
        batch_manager._calibrated_for_this_config = True
    else:
        optimal_batch_size = batch_manager.current_batch_size

    # Process proposals one by one to avoid memory accumulation
    joint_scores = []
    best_proposal_output = None
    best_score = float('inf')
    
    for p in range(P):
        buffers["curr"][:N].copy_(curr_base)
        buffers["curr"][:N].add_(perturb[p])  # add joint perturbation
        buffers["past"][:N].copy_(past_base)
        buffers["stat"][:N].copy_(stat_base)

        full_input = torch.cat([buffers['curr'][:N], buffers['past'][:N], buffers['stat'][:N]], dim=2)
        full_input = full_input.view(N * ensemble_size, C, H, W)
        full_time  = time_normalised.view(-1, 1).expand(-1, ensemble_size).reshape(-1, 1)

        # Use dynamic batch size with fallback mechanism
        step = optimal_batch_size * ensemble_size
        out_chunks = []
        
        start = 0
        while start < N * ensemble_size:
            end = min(start + step, N * ensemble_size)
            
            try:
                with torch.no_grad():
                    y = model(full_input[start:end], full_time[start:end])
                out_chunks.append(y)
                start = end
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    
                    # Halve batch size and retry
                    optimal_batch_size = max(1, optimal_batch_size // 2)
                    step = optimal_batch_size * ensemble_size
                    batch_manager.current_batch_size = optimal_batch_size
                    
                    print(f"[DynamicBatch] OOM encountered, reducing batch size to {optimal_batch_size}")
                    
                    # Retry with smaller batch
                    end = min(start + step, N * ensemble_size)
                    with torch.no_grad():
                        y = model(full_input[start:end], full_time[start:end])
                    out_chunks.append(y)
                    start = end
                else:
                    raise e
        
        y_full = torch.cat(out_chunks, dim=0).view(N, ensemble_size, V, H, W)
        proposal_output = y_full.permute(1, 0, 2, 3, 4)  # [K,N,V,H,W]
        
        # Compute CRPS immediately to avoid storing large tensors
        joint_score = compute_crps_for_proposal(proposal_output, current_fields, V)
        joint_scores.append(joint_score)
        
        # Keep track of best proposal
        if joint_score < best_score:
            best_score = joint_score
            # Store only the best proposal output
            if best_proposal_output is not None:
                del best_proposal_output
                torch.cuda.empty_cache()
            best_proposal_output = proposal_output.clone()
        
        # Clean up to save memory
        del proposal_output, y_full, out_chunks
        torch.cuda.empty_cache()

    return best_proposal_output, joint_scores

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

    # Initialize dynamic batch manager
    initial_batch_size = compute_safe_batch_size(
        ensemble_size=ensemble_size,
        num_variables=C,
        spatial_height=H,
        spatial_width=W,
        num_outputs=V,
    )
    batch_manager = DynamicBatchManager(device, initial_batch_size=initial_batch_size)
    print(f"[DynamicBatch] Initial conservative batch size: {initial_batch_size}")
    
    # Device warm-up
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

            # Memory status before inference
            allocated_before, _, total_mem, _ = batch_manager.get_memory_stats()
            print(f" {variable_names[v]:5s} | Mem: {allocated_before/1e9:.1f}/{total_mem/1e9:.1f}GB, "
                  f"Batch: {batch_manager.current_batch_size}")

            torch_gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
            best_ensemble, joint_scores = batched_forward_proposals(
                model=model,
                previous_fields=prev_all,
                current_fields=curr_all,
                time_normalised=time_all,
                reference_tensor=ref_full,
                alpha_matrix=alpha_tensor,
                ensemble_size=ensemble_size,
                device=device,
                buffers=buffers,
                batch_manager=batch_manager,
                generator=torch_gen,
            )  # Returns [K,N,V,H,W] and list of scores

            joint_scores = np.array(joint_scores)

            # Select best proposal (min joint CRPS)
            best_idx = int(joint_scores.argmin())
            current_alpha[v] = proposals_v[best_idx]
            posterior_samples[s, v] = current_alpha[v]
            posterior_crps[s, v]    = joint_scores[best_idx]  # store joint score replicated per variable

            print(f" {variable_names[v]:5s} α*={current_alpha[v,0]:.3f}  jointCRPS={joint_scores[best_idx]: .4f}")

            if log_diagnostics:
                # Use best ensemble output for per-variable diagnostics
                for j in range(num_variables):
                    if j == v:  # update diag for changed coord; optional: all j
                        spread_val = compute_ensemble_spread(best_ensemble[:, :, j].cpu())
                        mae_val    = compute_mean_absolute_error(
                            best_ensemble[:, :, j].cpu(), curr_all[:, j].cpu()
                        )
                        ensemble_spread_records[j].append(spread_val)
                        mean_absolute_error_records[j].append(mae_val)
                        ranks = compute_rank_histogram(
                            best_ensemble[:, :, j], curr_all[:, j], ensemble_size
                        )
                        rank_histograms[j].extend(ranks.tolist())
            
            # Clean up memory
            del best_ensemble
            torch.cuda.empty_cache()

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
