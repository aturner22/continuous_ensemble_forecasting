import torch

def print_posterior_summary(posterior_mean, posterior_variance, variable_names, parameter_labels):
    print("\nPosterior parameter moments:")
    print("---------------------------")
    for variable_index, variable in enumerate(variable_names):
        print(f"{variable}:")
        for parameter_index, label in enumerate(parameter_labels):
            μ = posterior_mean[variable_index, parameter_index]
            sigma2 = posterior_variance[variable_index, parameter_index]
            print(f"  {label}: μ = {μ:+.4f}, σ² = {sigma2:.4e}")


def continuous_ranked_probability_score(ensemble_tensor: torch.Tensor, target_field: torch.Tensor) -> torch.Tensor:
    absolute_error_term = torch.mean(torch.abs(ensemble_tensor - target_field.unsqueeze(0)))
    pairwise_differences = torch.abs(ensemble_tensor.unsqueeze(0) - ensemble_tensor.unsqueeze(1))
    second_moment_term = 0.5 * torch.mean(pairwise_differences)
    return absolute_error_term - second_moment_term

def compute_rank_histogram(predictions: torch.Tensor, targets: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    ranks = (predictions < targets.unsqueeze(0)).sum(dim=0).flatten().cpu().numpy()
    return ranks / ensemble_size

def compute_mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    mean_prediction = predictions.mean(dim=0)
    return torch.mean(torch.abs(mean_prediction - targets)).item()

def compute_ensemble_spread(predictions: torch.Tensor) -> float:
    return torch.std(predictions, dim=0).mean().item()
