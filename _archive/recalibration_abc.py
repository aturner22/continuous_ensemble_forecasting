import torch
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kstest, entropy
from prepare_model import prepare_model_and_loader

# Config
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Settings
SAMPLE_SIZE = 20
ENSEMBLE_SIZE = 100
N_GIBBS_STEPS = 15
N_CANDIDATES = 15
VARIABLE_NAMES = ['z500', 't850', 't2m', 'u10', 'v10']
NUM_VARIABLES = len(VARIABLE_NAMES)
NUM_STATIC_FIELDS = 2
MAX_HORIZON = 240
PARAM_DIM = 4
param_labels = ['alpha_bias', 'beta_bias', 'alpha_scale', 'beta_scale']

data_dir = './data'
model_dir = './models'
result_dir = './results/calibration_abc'
Path(result_dir).mkdir(parents=True, exist_ok=True)

# Load model and data
loader, model, lat, lon, _ = prepare_model_and_loader(
    device=device,
    data_directory=data_dir,
    result_directory=result_dir,
    model_directory=model_dir,
    variable_names=VARIABLE_NAMES,
    num_variables=NUM_VARIABLES,
    num_static_fields=NUM_STATIC_FIELDS,
    max_horizon=MAX_HORIZON,
    random_subset_size=SAMPLE_SIZE,
    random_subset_seed=777
)
model.eval()

posterior_samples = np.zeros((N_GIBBS_STEPS, NUM_VARIABLES, PARAM_DIM))
final_scores = np.zeros((N_GIBBS_STEPS, NUM_VARIABLES))

rank_histograms = [[] for _ in range(NUM_VARIABLES)]
mae_list = [[] for _ in range(NUM_VARIABLES)]
spread_list = [[] for _ in range(NUM_VARIABLES)]

def compute_calibration_score(predictions, targets):
    ranks = (predictions < targets.unsqueeze(0)).sum(dim=0).flatten().cpu().numpy()
    return ranks / ENSEMBLE_SIZE

def compute_mae(predictions, targets):
    mean_pred = predictions.mean(dim=0)
    return torch.mean(torch.abs(mean_pred - targets)).item()

def compute_spread(predictions):
    return torch.std(predictions, dim=0).mean().item()

def load_batches():
    batches = []
    for prev, curr, times in loader:
        prev = prev.to(device)
        curr = curr.view(-1, NUM_VARIABLES, len(lat), len(lon)).to(device)
        time = torch.tensor([times[0]], dtype=torch.float32).to(device) / MAX_HORIZON
        batches.append((prev, curr, time))
    return batches

def plot_posteriors_and_traces(samples, scores, path):
    for p in range(PARAM_DIM):
        plt.figure(figsize=(10, 6))
        for v in range(NUM_VARIABLES):
            plt.plot(samples[:, v, p], label=VARIABLE_NAMES[v])
        plt.title(f"Trace: {param_labels[p]}")
        plt.xlabel("Gibbs Iteration")
        plt.ylabel(param_labels[p])
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(path) / f"trace_{param_labels[p]}.png")
        plt.close()

    for p in range(PARAM_DIM):
        plt.figure(figsize=(12, 6))
        for v in range(NUM_VARIABLES):
            plt.hist(samples[:, v, p], bins=30, alpha=0.6, label=VARIABLE_NAMES[v], density=True)
        plt.title(f"Posterior Histogram: {param_labels[p]}")
        plt.xlabel(param_labels[p])
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(path) / f"histogram_{param_labels[p]}.png")
        plt.close()

def print_diagnostics(posterior_mean, posterior_var):
    print("\nPosterior Parameter Summary")
    print("---------------------------")
    for v in range(NUM_VARIABLES):
        print(f"{VARIABLE_NAMES[v]}:")
        for i in range(PARAM_DIM):
            μ = posterior_mean[v, i]
            σ2 = posterior_var[v, i]
            print(f"   {param_labels[i]}: μ = {μ:+.4f}, σ² = {σ2:.4e}")

def plot_rank_histograms(histograms, path):
    for v in range(NUM_VARIABLES):
        plt.figure(figsize=(8, 4))
        plt.hist(histograms[v], bins=ENSEMBLE_SIZE+1, range=(0, 1), density=True, edgecolor='black')
        plt.title(f"Rank Histogram – {VARIABLE_NAMES[v]}")
        plt.xlabel("Normalized Rank")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(Path(path) / f"rank_hist_{VARIABLE_NAMES[v]}.png")
        plt.close()

print("Beginning ABC-Gibbs sampling...")

with torch.inference_mode():
    init_param = np.random.uniform(
        low=[-0.2, -0.1, -1.0, -1.0],
        high=[0.2, 0.1, 1.0, 1.0],
        size=(NUM_VARIABLES, PARAM_DIM)
    )

    for t in tqdm(range(N_GIBBS_STEPS), desc='Gibbs Iteration'):
        print(f"Starting Gibbs step {t+1}/{N_GIBBS_STEPS}")
        batches = load_batches()

        for v in range(NUM_VARIABLES):
            print(f"  Sampling for variable: {VARIABLE_NAMES[v]}")
            proposals = np.random.normal(loc=init_param[v], scale=[0.05]*PARAM_DIM, size=(N_CANDIDATES, PARAM_DIM))
            best_score = float('inf')
            best_param = init_param[v].copy()

            for c in range(N_CANDIDATES):
                alpha_b, beta_b, alpha_s, beta_s = proposals[c]
                calibration_ranks = []
                all_members = []
                all_targets = []

                for prev, curr, time_tensor in batches:
                    var_in = prev[:, :-NUM_STATIC_FIELDS, :, :]
                    static_in = prev[:, -NUM_STATIC_FIELDS:, :, :]
                    base = var_in[:, v, :, :]

                    members = []
                    scale = torch.exp(alpha_s + beta_s * base)
                    bias = alpha_b + beta_b * base

                    for _ in range(ENSEMBLE_SIZE):
                        noise = torch.randn_like(base)
                        perturbed = base + (noise + bias) * scale
                        perturbed_input = var_in.clone()
                        perturbed_input[:, v, :, :] = perturbed
                        full_input = torch.cat([perturbed_input, static_in], dim=1)
                        out = model(full_input, time_tensor)
                        members.append(out)

                    ens_stack = torch.stack(members, dim=0)
                    calibration_ranks.append(compute_calibration_score(ens_stack[:, :, v], curr[:, v, :, :]))
                    all_members.append(ens_stack[:, :, v])
                    all_targets.append(curr[:, v, :, :])

                rank_concat = np.concatenate(calibration_ranks)
                ks_stat, _ = kstest(rank_concat, 'uniform')
                if ks_stat < best_score:
                    best_score = ks_stat
                    best_param = proposals[c]
                    best_members = all_members
                    best_targets = all_targets
                    best_ranks = rank_concat

            init_param[v] = best_param
            posterior_samples[t, v] = best_param
            final_scores[t, v] = best_score

            rank_histograms[v].extend(best_ranks)
            spread_list[v].append(np.mean([compute_spread(m) for m in best_members]))
            mae_list[v].append(np.mean([compute_mae(m, t) for m, t in zip(best_members, best_targets)]))

np.save(f"{result_dir}/posterior_calibration_samples.npy", posterior_samples)
np.save(f"{result_dir}/posterior_calibration_scores.npy", final_scores)

posterior_mean = posterior_samples.mean(axis=0)
posterior_var = posterior_samples.var(axis=0)
np.save(f"{result_dir}/posterior_calibration_mean.npy", posterior_mean)
np.save(f"{result_dir}/posterior_calibration_var.npy", posterior_var)

print_diagnostics(posterior_mean, posterior_var)
plot_posteriors_and_traces(posterior_samples, final_scores, result_dir)
plot_rank_histograms(rank_histograms, result_dir)

np.save(f"{result_dir}/ensemble_mae.npy", np.array(mae_list))
np.save(f"{result_dir}/ensemble_spread.npy", np.array(spread_list))

print("Inference complete.")
