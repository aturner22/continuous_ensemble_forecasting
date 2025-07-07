import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def produce_trace_and_histogram_plots(samples: np.ndarray, output_directory: Path, variable_names, parameter_labels):
    num_variables = samples.shape[1]
    parameter_dim = samples.shape[2]

    for parameter_index in range(parameter_dim):
        plt.figure(figsize=(10, 6))
        for variable_index in range(num_variables):
            plt.plot(samples[:, variable_index, parameter_index], label=variable_names[variable_index])
        plt.title(f"Trace – {parameter_labels[parameter_index]}")
        plt.xlabel("Gibbs iteration")
        plt.ylabel(parameter_labels[parameter_index])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_directory / f"trace_{parameter_labels[parameter_index]}.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        for variable_index in range(num_variables):
            plt.hist(
                samples[:, variable_index, parameter_index],
                bins=30,
                alpha=0.6,
                label=variable_names[variable_index],
                density=True,
            )
        plt.title(f"Posterior – {parameter_labels[parameter_index]}")
        plt.xlabel(parameter_labels[parameter_index])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_directory / f"hist_{parameter_labels[parameter_index]}.png")
        plt.close()

def produce_rank_histograms(histograms, output_directory: Path, variable_names, ensemble_size: int):
    for variable_index, ranks in enumerate(histograms):
        plt.figure(figsize=(8, 4))
        plt.hist(
            ranks,
            bins=ensemble_size + 1,
            range=(0, 1),
            density=True,
            edgecolor="black",
        )
        plt.title(f"Rank histogram – {variable_names[variable_index]}")
        plt.xlabel("Normalised rank")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(output_directory / f"rank_hist_{variable_names[variable_index]}.png")
        plt.close()
