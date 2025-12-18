import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers.pytorch_utils import Conv1D
import logging
import os

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)

log = logging.getLogger(__name__)


class WandaAnalysis:
    """
    Collects true Wanda activation statistics using forward hooks and computes
    exact Wanda importance metrics for each Linear/Conv1D layer.
    """

    def __init__(self, model, pruning_type, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.pruning_type = pruning_type

        # Per-module activation statistics collected via hooks
        self._module_stats = {}

        # Stores final Wanda metrics
        self.wanda_mean = {}
        self.wanda_var = {}

        # Store actiations
        self.activations_mean = {}
        self.activations_var = {}

        # Module -> qualified name lookup for better reporting
        self._module_names = {}

        # active hooks
        self.handles = []

        # register hooks
        self._register_hooks()

    # ------------------------------
    # Internal methods
    # ------------------------------

    def _register_hooks(self):
        """Attach forward hooks to Linear and Conv1D modules."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, Conv1D):
                h = module.register_forward_hook(self._hook_fn)
                self.handles.append(h)
                self._module_names[module] = name or module.__class__.__name__

    def _hook_fn(self, module, inputs, output):
        """Forward hook: accumulates feature-wise activation norms and collects activation stats."""
        x = inputs[0]
        if x is None or not isinstance(x, torch.Tensor):
            return

        # Move to CPU float for numerical stability
        x = x.detach().to(dtype=torch.float32)

        if x.ndim == 2:
            # Some modules use (batch, hidden)
            x = x.unsqueeze(1)

        B, S, H = x.shape
        x = x.cpu().to(dtype=torch.float64)

        # Collapse only token dimension when computing aggregate norms
        flat = x.reshape(B * S, H)
        batch_sum_sq = (flat**2).sum(dim=0)  # (H,)

        # Per-sample feature norms (||X_j||_2) for mean/std statistics
        sample_sq = (x**2).sum(dim=1)  # (B, H), sum over sequence len
        sample_l2 = torch.sqrt(sample_sq + 1e-12)

        # --- Collect mean and variance of activations ---
        activ_mean = flat.mean(dim=0)  # (H,)
        activ_var = flat.var(dim=0, unbiased=False)  # (H,)
        if not hasattr(self, "_activ_samples"):
            self._activ_samples = {}
        if module not in self.activations_mean:
            self.activations_mean[module] = activ_mean.clone()
            self.activations_var[module] = activ_var.clone()
            self._activ_samples[module] = 1
        else:
            self.activations_mean[module] += activ_mean
            self.activations_var[module] += activ_var
            self._activ_samples[module] += 1

        stats = self._module_stats.get(module)
        if stats is None:
            zeros = torch.zeros(H, dtype=torch.float64)
            stats = {
                "sum_sq": zeros.clone(),  # aggregate \sum x^2 for Eq. (1)
                "sum_l2": zeros.clone(),  # accumulate ||X_j|| across samples
                "sum_l2_sq": zeros.clone(),  # accumulate ||X_j||^2 across samples
                "samples": 0,
            }
            self._module_stats[module] = stats
        stats["sum_sq"] += batch_sum_sq
        stats["sum_l2"] += sample_l2.sum(dim=0)
        stats["sum_l2_sq"] += (sample_l2**2).sum(dim=0)
        stats["samples"] += B

    @staticmethod
    def _get_weight(module):
        """Handles Conv1D transposition."""
        if isinstance(module, Conv1D):
            return module.weight.t()  # Conv1D uses reversed dims
        else:
            return module.weight

    def compute_activations_stats(self):
        """Calcola la media e la varianza finali delle attivazioni per ogni modulo."""
        for module in self.activations_mean:
            n = self._activ_samples[module]
            self.activations_mean[module] = (
                (self.activations_mean[module] / n).cpu().numpy()
            )
            self.activations_var[module] = (
                (self.activations_var[module] / n).cpu().numpy()
            )

    # ------------------------------
    # Public API
    # ------------------------------

    def collect(self, dataloader):
        """Runs calibration data through the model and collects activations."""
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                _ = self.model(**batch)

    def compute_scores(self):
        """Computes mean and std Wanda metrics for each module."""
        self.wanda_mean.clear()
        self.wanda_var.clear()

        for module, stats in self._module_stats.items():
            sum_sq = stats["sum_sq"]
            samples = max(stats["samples"], 1)

            # Eq. (1): ||X_j||_2 aggregated across all tokens in calibration set
            feature_norm = torch.sqrt(sum_sq + 1e-12)

            # Mean/std of ||X_j||_2 across calibration samples
            mean_l2 = stats["sum_l2"] / samples
            mean_l2_sq = stats["sum_l2_sq"] / samples
            feature_var = torch.clamp(mean_l2_sq - mean_l2**2, min=0.0)

            W = self._get_weight(module).detach().float().cpu().abs()
            wanda_mean = W * feature_norm.float().unsqueeze(0)
            wanda_var = W * feature_var.float().unsqueeze(0)

            self.wanda_mean[module] = wanda_mean.numpy()
            self.wanda_var[module] = wanda_var.numpy()

    def plot(self, save_path="wanda_analysis.pdf", max_layers=None):
        """Creates a PDF with Wanda heatmaps."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        modules = list(self.wanda_mean.keys())
        if max_layers is not None:
            modules = modules[:max_layers]

        rows = len(modules)
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), layout="constrained")
        if rows == 1:
            axes = np.array([axes])

        with open(save_path.replace(".pdf", ".txt"), "w") as f:
            f.write(
                "pruning_type,layer,wanda_max,wanda_mean,wanda_min,wanda_median,activations_max,activations_mean,activations_min,activations_median\n"
            )
            for i, module in enumerate(modules):
                mean_map = self.wanda_mean[module]
                # var_map = self.wanda_var[module]
                mean_activ = self.activations_mean[module]
                # var_activ = self.activations_var[module]
                name = self._module_names.get(module, module.__class__.__name__)

                # ax1 = axes[i, 0]
                # im1 = ax1.imshow(
                #     mean_map,
                #     aspect="auto",
                #     cmap="viridis",
                #     vmin=0.0,
                #     vmax=mean_map.max(),
                #     interpolation="nearest",
                # )
                # ax1.set_title(f"{name} — Mean Wanda Score")
                # plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                # ax2 = axes[i, 1]
                # im2 = ax2.imshow(
                #     var_map,
                #     aspect="auto",
                #     cmap="magma",
                #     vmin=0.0,
                #     vmax=var_map.max(),
                #     interpolation="nearest",
                # )
                # ax2.set_title(f"{name} — Variance Component")
                # plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

                f.write(
                    f"{self.pruning_type}, {name}, {mean_map.max():.3f}, {mean_map.mean():.3f}, {mean_map.min():.3f}, {np.median(mean_map):.3f}, {mean_activ.max():.3f}, {mean_activ.mean():.3f}, {mean_activ.min():.3f}, {np.median(mean_activ):.3f}\n"
                )

            # plt.suptitle(
            #     "Wanda Analysis using " + self.pruning_type + " Pruning", fontsize=16
            # )
            # plt.savefig(save_path, dpi=150)
            # plt.close()
            log.info(f"Wanda plots saved to {save_path}")

    def remove_hooks(self):
        """Detach all hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []
