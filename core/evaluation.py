import torch

def validate_tensor_shapes(ensemble: torch.Tensor, target: torch.Tensor):
    if ensemble.dim() < 2:
        raise ValueError("Ensemble tensor must be at least 2-dimensional [E, ...]")
    if ensemble.shape[1:] != target.shape:
        raise ValueError(f"Incompatible shapes: {ensemble.shape[1:]} vs {target.shape}")

def print_posterior_summary(posterior_mean, posterior_variance, variable_names, parameter_labels):
    print("\nPosterior parameter moments:")
    print("---------------------------")
    for variable_index, variable in enumerate(variable_names):
        print(f"{variable}:")
        for parameter_index, label in enumerate(parameter_labels):
            mu = posterior_mean[variable_index, parameter_index]
            sigma_square = posterior_variance[variable_index, parameter_index]
            print(f"  {label}: mu = {mu:+.4f}, sigma_square = {sigma_square:.4e}")

def continuous_ranked_probability_score(ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    validate_tensor_shapes(ensemble, target)
    absolute_error = torch.abs(ensemble - target.unsqueeze(0)).mean(dim=0)
    pairwise = torch.abs(ensemble.unsqueeze(0) - ensemble.unsqueeze(1)).mean(dim=(0, 1))
    return absolute_error - 0.5 * pairwise

def compute_rank_histogram(ensemble: torch.Tensor, target: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    validate_tensor_shapes(ensemble, target)
    with torch.no_grad():
        rank_counts = (ensemble < target.unsqueeze(0)).sum(dim=0)
    histogram = rank_counts.view(-1).float().cpu().numpy() / ensemble_size
    return histogram

def compute_mean_absolute_error(ensemble: torch.Tensor, target: torch.Tensor) -> float:
    validate_tensor_shapes(ensemble, target)
    return torch.abs(ensemble.mean(dim=0) - target).mean().item()

def compute_ensemble_spread(ensemble: torch.Tensor) -> float:
    if ensemble.dim() < 2:
        raise ValueError("Ensemble tensor must be at least 2-dimensional")
    return ensemble.std(dim=0).mean().item()
