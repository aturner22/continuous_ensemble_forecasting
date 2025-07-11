import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_rfp_noise_field(
    reference_tensor: torch.Tensor,
    alpha: float,
    variable_index: int,
    lat: np.ndarray,
    lon: np.ndarray,
    sample_indices: tuple[int, int] = (0, 1),
    save_path: str | None = None,
) -> None:
    field_1 = reference_tensor[sample_indices[0], variable_index]
    field_2 = reference_tensor[sample_indices[1], variable_index]
    perturbation = alpha * (field_1 - field_2)
    perturbation_np = perturbation.detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.Robinson())
    image = ax.pcolormesh(
        lon, lat, perturbation_np,
        transform=ccrs.PlateCarree(), cmap='RdBu_r'
    )
    ax.coastlines()
    ax.set_title(f'RFP Perturbation (Variable {variable_index}, α = {alpha:.2e})')
    plt.colorbar(image, ax=ax, orientation='horizontal', label='Perturbation Magnitude')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_ensemble_statistics(
    ensemble_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    variable_index: int,
    lat: np.ndarray,
    lon: np.ndarray,
    save_prefix: str | None = None,
) -> None:
    ensemble_mean = ensemble_tensor.mean(dim=0)[variable_index].detach().cpu().numpy()
    ensemble_std = ensemble_tensor.std(dim=0)[variable_index].detach().cpu().numpy()
    truth = target_tensor[0, variable_index].detach().cpu().numpy()
    std_map = ensemble_tensor.std(dim=0)[variable_index].detach().cpu().numpy()
    print("Std Dev: min =", std_map.min(), "max =", std_map.max(), "mean =", std_map.mean())
    std_tensor = ensemble_tensor.std(dim=0)[variable_index]
    if torch.allclose(std_tensor, torch.zeros_like(std_tensor), atol=1e-6):
        print("Degenerate ensemble: standard deviation is numerically zero.")
    else:
        print("Non-zero std deviation detected. Adjust plotting range or inspect ensemble diversity.")



    def _render_field(data: np.ndarray, title: str, path: str | None = None) -> None:
        fig = plt.figure(figsize=(6, 4))
        ax = plt.axes(projection=ccrs.Robinson())
        vmin, vmax = np.percentile(data, [2, 98])
        image = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=vmin, vmax=vmax)

        ax.coastlines()
        ax.set_title(title)
        plt.colorbar(image, ax=ax, orientation='horizontal')
        if path:
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    if save_prefix is not None:
        _render_field(truth, "Ground Truth", f"{save_prefix}_truth.png")
        _render_field(ensemble_mean, "Ensemble Mean", f"{save_prefix}_mean.png")
        _render_field(ensemble_std, "Ensemble Std Dev", f"{save_prefix}_std.png")
    else:
        _render_field(truth, "Ground Truth")
        _render_field(ensemble_mean, "Ensemble Mean")
        _render_field(ensemble_std, "Ensemble Std Dev")
