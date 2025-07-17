import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from prepare_model import prepare_model_and_loader

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Config
SAMPLE_SIZE = 100
ENSEMBLE_SIZE = 1
param_dim = 2
param_labels = ['alpha_bias', 'beta_bias']
VARIABLE_NAMES = ['z500', 't850', 't2m', 'u10', 'v10']
NUM_VARIABLES = len(VARIABLE_NAMES)
NUM_STATIC_FIELDS = 2
MAX_HORIZON = 240

data_directory = './data'
result_directory = './results/global_gibbs_abc_fixed_variance/deterministic-iterative-6h'
model_directory = './models'
posterior_path = Path(result_directory)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load posterior mean and variance
posterior_mean = np.load(posterior_path / "posterior_mean.npy")
posterior_var = np.load(posterior_path / "posterior_var.npy")

# Load model and data
loader, model, lat, lon, _ = prepare_model_and_loader(
    device=device,
    data_directory=data_directory,
    result_directory=result_directory,
    model_directory=model_directory,
    variable_names=VARIABLE_NAMES,
    num_variables=NUM_VARIABLES,
    num_static_fields=NUM_STATIC_FIELDS,
    max_horizon=MAX_HORIZON,
    random_subset_size=SAMPLE_SIZE,
    random_subset_seed=999
)
model.eval()

# Evaluation
mae_plain = []
mae_recalibrated = []

with torch.inference_mode():
    for previous, current, time_labels in tqdm(loader, total=SAMPLE_SIZE):
        previous = previous.to(device)
        current = current.view(-1, NUM_VARIABLES, len(lat), len(lon)).to(device)
        time_tensor = torch.tensor([time_labels[0]], dtype=torch.float32).to(device) / MAX_HORIZON

        # Plain forecast
        output_plain = model(previous, time_tensor)
        mae_plain.append(torch.mean(torch.abs(output_plain - current)).item())

        # Recalibrated forecast
        var_channels = previous[:, :-NUM_STATIC_FIELDS, :, :]
        static_channels = previous[:, -NUM_STATIC_FIELDS:, :, :]

        ensemble_outputs = []
        for _ in range(ENSEMBLE_SIZE):
            perturbed = var_channels.clone()
            for v in range(NUM_VARIABLES):
                alpha, beta = posterior_mean[v]
                base = var_channels[:, v, :, :]
                noise = torch.randn_like(base)
                perturbation = base + (alpha + beta * base)
                perturbed[:, v, :, :] = perturbation

            full_input = torch.cat([perturbed, static_channels], dim=1)
            ensemble_outputs.append(model(full_input, time_tensor))

        output_ensemble = torch.stack(ensemble_outputs, dim=0).mean(dim=0)
        mae_recalibrated.append(torch.mean(torch.abs(output_ensemble - current)).item())

# Results
mean_plain = np.mean(mae_plain)
mean_recal = np.mean(mae_recalibrated)

print("\nMAE Comparison over 100 Samples")
print("------------------------------")
print(f"Plain Forecast MAE:        {mean_plain:.4f}")
print(f"Recalibrated Forecast MAE: {mean_recal:.4f}")
