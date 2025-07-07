def print_posterior_summary(posterior_mean, posterior_variance, variable_names, parameter_labels):
    print("\nPosterior parameter moments:")
    print("---------------------------")
    for variable_index, variable in enumerate(variable_names):
        print(f"{variable}:")
        for parameter_index, label in enumerate(parameter_labels):
            μ = posterior_mean[variable_index, parameter_index]
            σ2 = posterior_variance[variable_index, parameter_index]
            print(f"  {label}: μ = {μ:+.4f}, σ² = {σ2:.4e}")
