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

        # Store activations
        self.activations_mean = {}
        self.activations_var = {}

        # Store results for different techniques
        self.results_by_technique = {}

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
        # Move to CPU float for numerical stability, but keep float32 to save RAM
        x = x.detach().cpu().to(dtype=torch.float32)

        # Collapse only token dimension when computing aggregate stats
        flat = x.reshape(B * S, H)

        # --- Collect stats for mean and variance ---
        # Avoid creating multiple copies, use precise accumulation
        sum_x = flat.sum(dim=0, dtype=torch.float64)
        sum_sq = (flat**2).sum(dim=0, dtype=torch.float64)
        num_tokens = B * S

        # Per-sample feature norms (||X_j||_2) for mean/std statistics
        sample_sq = (x.to(dtype=torch.float64)**2).sum(dim=1)  # (B, H), sum over sequence len
        sample_l2 = torch.sqrt(sample_sq + 1e-12)

        stats = self._module_stats.get(module)
        if stats is None:
            stats = {
                "sum_x": sum_x,
                "sum_sq": sum_sq,  # aggregate \sum x^2 for Eq. (1)
                "sum_l2": sample_l2.sum(dim=0),  # accumulate ||X_j|| across samples
                "sum_l2_sq": (sample_l2**2).sum(dim=0),  # accumulate ||X_j||^2 across samples
                "samples": B,
                "total_tokens": num_tokens,
            }
            self._module_stats[module] = stats
        else:
            stats["sum_x"] += sum_x
            stats["sum_sq"] += sum_sq
            stats["sum_l2"] += sample_l2.sum(dim=0)
            stats["sum_l2_sq"] += (sample_l2**2).sum(dim=0)
            stats["samples"] += B
            stats["total_tokens"] += num_tokens

        # Explicitly delete temporary tensors to help GC
        del flat
        del x
        del sample_sq
        del sample_l2

    @staticmethod
    def _get_weight(module):
        """Handles Conv1D transposition."""
        if isinstance(module, Conv1D):
            return module.weight.t()  # Conv1D uses reversed dims
        else:
            return module.weight

    def compute_activations_stats(self):
        """Calcola la media e la varianza finali delle attivazioni per ogni modulo."""
        for module, stats in self._module_stats.items():
            N = max(stats["total_tokens"], 1)
            mean = stats["sum_x"] / N
            sq_mean = stats["sum_sq"] / N
            var = torch.clamp(sq_mean - mean**2, min=0.0)

            self.activations_mean[module] = mean.cpu().numpy()
            self.activations_var[module] = var.cpu().numpy()

    def clear_stats(self):
        """Resets the accumulated statistics for the next collection."""
        self._module_stats = {}
        self.wanda_mean = {}
        self.wanda_var = {}
        self.activations_mean = {}
        self.activations_var = {}
        if hasattr(self, "_activ_samples"):
            self._activ_samples = {}

    def store_results(self, technique_name):
        """Stores the current computed stats under a technique name."""
        self.results_by_technique[technique_name] = {
            "activations_mean": {m: v.copy() for m, v in self.activations_mean.items()},
            "activations_var": {m: v.copy() for m, v in self.activations_var.items()},
            "feature_norm": {m: torch.sqrt(s["sum_sq"] + 1e-12).cpu().numpy() for m, s in self._module_stats.items()},
        }
        self.clear_stats()

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
        """Creates a PDF with Wanda heatmaps and saves stats to CSV."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        modules = list(self.wanda_mean.keys())
        if max_layers is not None:
            modules = modules[:max_layers]

        rows = len(modules)
        fig, axes = plt.subplots(rows, 2, figsize=(16, 5 * rows), layout="constrained")
        if rows == 1:
            axes = np.expand_dims(axes, 0)

        csv_path = save_path.replace(".pdf", ".csv")
        with open(csv_path, "w") as f:
            f.write(
                "pruning_type,layer,wanda_max,wanda_mean,wanda_min,wanda_median,activations_max,activations_mean,activations_min,activations_median,activations_var\n"
            )
            for i, module in enumerate(modules):
                mean_map = self.wanda_mean[module]
                mean_activ = self.activations_mean[module]
                name = self._module_names.get(module, module.__class__.__name__)

                # Plotting Mean Wanda Score
                ax1 = axes[i, 0]
                im1 = ax1.imshow(
                    mean_map,
                    aspect="auto",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=mean_map.max() if mean_map.max() > 0 else 1.0,
                    interpolation="nearest",
                )
                ax1.set_title(f"{name} — Mean Wanda Score")
                plt.colorbar(im1, ax=ax1)

                # Plotting Mean Activations
                ax2 = axes[i, 1]
                # We show activations as a 1D heatmap (extended for visibility)
                activ_heatmap = np.tile(mean_activ, (mean_map.shape[0] // 10 or 1, 1))
                im2 = ax2.imshow(
                    activ_heatmap,
                    aspect="auto",
                    cmap="magma",
                    interpolation="nearest"
                )
                ax2.set_title(f"{name} — Mean Activations")
                plt.colorbar(im2, ax=ax2)

                f.write(
                    f"{self.pruning_type}, {name}, {mean_map.max():.3f}, {mean_map.mean():.3f}, {mean_map.min():.3f}, {np.median(mean_map):.3f}, {mean_activ.max():.3f}, {mean_activ.mean():.3f}, {mean_activ.min():.3f}, {np.median(mean_activ):.3f}, {mean_activ.var():.3f}\n"
                )

        plt.suptitle(f"Wanda Analysis: {self.pruning_type}", fontsize=16)
        plt.savefig(save_path, dpi=150)
        plt.close()
        log.info(f"Wanda plots saved to {save_path} and stats to {csv_path}")

    def plot_diff(
        self, technique, reference="random", save_path="wanda_diff.pdf", max_layers=None
    ):
        """
        Computes the difference in statistics between a technique and a reference,
        and plots the resulting mean and variance differences as heatmaps.
        """
        if technique not in self.results_by_technique:
            log.error(f"Technique {technique} not found in results.")
            return
        if reference not in self.results_by_technique:
            log.error(f"Reference {reference} not found in results.")
            return

        tech_res = self.results_by_technique[technique]
        ref_res = self.results_by_technique[reference]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        modules = list(tech_res["activations_mean"].keys())
        if max_layers is not None:
            modules = modules[:max_layers]

        rows = len(modules)
        fig, axes = plt.subplots(rows, 2, figsize=(16, 5 * rows), layout="constrained")
        if rows == 1:
            axes = np.expand_dims(axes, 0)

        for i, module in enumerate(modules):
            name = self._module_names.get(module, str(module))

            # Difference of activations means
            mean_diff = (
                tech_res["activations_mean"][module]
                - ref_res["activations_mean"][module]
            )
            # Variance of the difference (sum of variances assuming independence)
            var_diff = (
                tech_res["activations_var"][module] + ref_res["activations_var"][module]
            )

            W = self._get_weight(module).detach().float().cpu().abs().numpy()

            # Heatmap values: Weight * Activation_Stat (broadcasting O, I * I)
            score_mean_diff = W * mean_diff[np.newaxis, :]
            score_std_diff = W * np.sqrt(var_diff + 1e-12)[np.newaxis, :]

            # Plot Mean Difference
            ax1 = axes[i, 0]
            im1 = ax1.imshow(
                score_mean_diff,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
            )
            ax1.set_title(f"{name}\nMean Diff ({technique} - {reference})")
            plt.colorbar(im1, ax=ax1)

            # Plot Std of Difference
            ax2 = axes[i, 1]
            im2 = ax2.imshow(
                score_std_diff, aspect="auto", cmap="viridis", interpolation="nearest"
            )
            ax2.set_title(f"{name}\nStd of Diff")
            plt.colorbar(im2, ax=ax2)

        plt.suptitle(
            f"Wanda Difference Analysis: {technique} vs {reference}", fontsize=16
        )
        plt.savefig(save_path, dpi=150)
        plt.close()
        log.info(f"Wanda difference plots saved to {save_path}")

    def plot_summary_heatmaps(self, reference="random", save_path="wanda_summary.pdf"):
        """
        Creates two global heatmaps (Mean and Variance) where:
        - Rows: Sampling techniques
        - Columns: Model layers
        - Cell: Stat of the difference matrix (Tech - Reference)
        """
        techniques = [t for t in self.results_by_technique.keys() if t != reference]
        if not techniques:
            log.warning("No techniques to compare against reference.")
            return

        # Get modules from the reference results
        modules = list(self.results_by_technique[reference]["activations_mean"].keys())
        module_names = [self._module_names.get(m, "Layer").split(".")[-1] for m in modules]

        mean_grid = np.zeros((len(techniques), len(modules)))
        var_grid = np.zeros((len(techniques), len(modules)))

        for t_idx, tech in enumerate(techniques):
            tech_res = self.results_by_technique[tech]
            ref_res = self.results_by_technique[reference]
            for m_idx, module in enumerate(modules):
                # Avoid keeping full weight matrix in memory if possible, but we need it locally
                W = self._get_weight(module).detach().float().cpu().abs().numpy()
                
                # Difference in activation norms (H,)
                # Wanda score difference uses the aggregated feature norm (L2)
                diff_norm = (
                    tech_res["feature_norm"][module]
                    - ref_res["feature_norm"][module]
                )
                
                # Matrix of score differences (O, I)
                score_diff_matrix = W * diff_norm[np.newaxis, :]
                
                mean_grid[t_idx, m_idx] = np.mean(score_diff_matrix)
                var_grid[t_idx, m_idx] = np.var(score_diff_matrix)
                
                # Explicitly delete large matrix
                del W
                del score_diff_matrix

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(modules)//4), 10), layout="constrained")
        
        # Plot Mean
        im1 = ax1.imshow(mean_grid, aspect="auto", cmap="viridis")
        ax1.set_title(f"Mean of Wanda Score Difference ({self.pruning_type})")
        ax1.set_yticks(np.arange(len(techniques)), labels=techniques)
        ax1.set_xticks(np.arange(len(modules)), labels=module_names, rotation=90)
        ax1.set_ylabel("Techniques")
        plt.colorbar(im1, ax=ax1)

        # Plot Variance
        im2 = ax2.imshow(var_grid, aspect="auto", cmap="viridis")
        ax2.set_title(f"Variance of Wanda Score Difference ({self.pruning_type})")
        ax2.set_yticks(np.arange(len(techniques)), labels=techniques)
        ax2.set_xticks(np.arange(len(modules)), labels=module_names, rotation=90)
        ax2.set_ylabel("Techniques")
        plt.colorbar(im2, ax=ax2)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        log.info(f"Summary heatmaps saved to {save_path}")

    def remove_hooks(self):
        """Detach all hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []
