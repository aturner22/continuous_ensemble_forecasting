import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
import seaborn as sns

VARIABLE_NAMES = ['z500', 't850', 't2m', 'u10', 'v10']
param_labels = ['alpha_bias', 'beta_bias', 'alpha_scale', 'beta_scale']
NUM_VARIABLES = len(VARIABLE_NAMES)
param_dim = len(param_labels)

result_path = Path('./results/gibbs_abc/deterministic-iterative-6h')
sample_files = sorted(result_path.glob("accepted_parameters_t*.npy"))
accepted_samples = np.stack([np.load(f) for f in sample_files])
# Shape: (T, G, V, D)
T, G, V, D = accepted_samples.shape

# Placeholder if actual MAE is not saved
raw_mae = np.zeros((T, G))

def epanechnikov_kernel(u):
    return np.maximum(0, 1 - u**2)

def regression_adjustment_full(params, maes):
    T, G, V, D = params.shape
    adjusted = np.zeros((T, G, V, D))
    for t in range(T):
        for v in range(V):
            for d in range(D):
                X = maes[t, :].reshape(-1, 1)
                y = params[t, :, v, d]
                std_X = np.std(X)
                if std_X == 0 or np.isnan(std_X) or np.isinf(std_X):
                    adjusted[t, :, v, d] = y
                    continue
                h = std_X * (len(X) ** (-1 / 5))
                if h == 0 or np.isnan(h) or np.isinf(h):
                    adjusted[t, :, v, d] = y
                    continue
                weights = epanechnikov_kernel((X - np.median(X)) / h).flatten()
                if np.any(np.isnan(weights)) or np.all(weights == 0):
                    adjusted[t, :, v, d] = y
                    continue
                reg = LinearRegression()
                reg.fit(X, y, sample_weight=weights)
                y_adj = y - reg.coef_[0] * (X.flatten() - np.median(X))
                adjusted[t, :, v, d] = y_adj
    return adjusted

adjusted_samples = regression_adjustment_full(accepted_samples, raw_mae)
posterior_means = np.mean(adjusted_samples, axis=1)  # shape: (T, V, D)
posterior_stds = np.std(adjusted_samples, axis=1)    # shape: (T, V, D)

print("\nRegression-Adjusted Posterior Statistics Over Time")
print("----------------------------------------------------")
for v in range(NUM_VARIABLES):
    for d in range(param_dim):
        time_series = posterior_means[:, v, d]
        std_series = posterior_stds[:, v, d]
        print(f"\n{VARIABLE_NAMES[v]} – {param_labels[d]}")
        for t in range(0,T,10):
            μ = time_series[t]
            σ = std_series[t]
            print(f"t = {t:03d} | mean = {μ:.5f} | std = {σ:.5f}")

# Plot mean trajectories
for d in range(param_dim):
    plt.figure(figsize=(14, 6))
    for v in range(NUM_VARIABLES):
        plt.plot(posterior_means[:, v, d], label=VARIABLE_NAMES[v])
    plt.title(f"Regression-Adjusted Posterior Means Over Time: {param_labels[d]}")
    plt.xlabel("Timestep")
    plt.ylabel("Posterior Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path / f"posterior_mean_trajectory_adjusted_{param_labels[d]}.png")
    plt.close()

# Plot std trajectories
for d in range(param_dim):
    plt.figure(figsize=(14, 6))
    for v in range(NUM_VARIABLES):
        plt.plot(posterior_stds[:, v, d], label=VARIABLE_NAMES[v])
    plt.title(f"Regression-Adjusted Posterior Std Dev Over Time: {param_labels[d]}")
    plt.xlabel("Timestep")
    plt.ylabel("Posterior Std Dev")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_path / f"posterior_std_trajectory_adjusted_{param_labels[d]}.png")
    plt.close()
