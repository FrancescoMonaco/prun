"""
Plot scalability results: selection time vs. data-pool size for our method and COLA.

Usage:
    python plot/scalability_plot.py --input results/scalability_results.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── Publication-quality defaults ────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 15,
        "legend.title_fontsize": 11,
        "figure.titlesize": 12,
        "font.family": "serif",
        "text.usetex": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── Consistent palette ──────────────────────────────────────────────────────────
OURS_COLOR = "#2ca02c"
COLA_PALETTE = {
    "3B": "#1f77b4",
    "11B": "#ff7f0e",
    "70B": "#d62728",
}

OURS_MARKER = "s"
COLA_MARKERS = {"3B": "o", "11B": "^", "70B": "D"}


def _short_model_label(name: str) -> str:
    """Turn 'meta-llama/Llama-3.2-70B' → 'Llama-3.2-70B' and extract size tag."""
    short = name.split("/")[-1] if "/" in name else name
    return short


def _size_tag(name: str) -> str:
    """Extract '3B' / '11B' / '70B' from a model name string."""
    for tag in ("70B", "11B", "3B"):
        if tag.lower() in name.lower():
            return tag
    return name.split("/")[-1]


def plot_scalability(df: pd.DataFrame, output_dir: str):
    """
    Produce one figure per dataset.
    X-axis = number of samples in the pool (n_used).
    Y-axis = total selection time (s).
    One line for "ours", one line per COLA model.
    """
    datasets = df["dataset"].unique()

    for ds in datasets:
        sub = df[df["dataset"] == ds].copy()

        fig, ax = plt.subplots(figsize=(5.5, 4))

        # ── Our method ──
        ours = sub[sub["method"] == "ours"].sort_values("n_used")
        if not ours.empty:
            ax.plot(
                ours["n_used"],
                ours["total_time_s"],
                marker=OURS_MARKER,
                color=OURS_COLOR,
                linewidth=2,
                markersize=7,
                label="Ours (model-free)",
                zorder=5,
            )

        # ── COLA models ──
        cola = sub[sub["method"] == "cola"]
        for model_name, grp in cola.groupby("model"):
            grp = grp.sort_values("n_used")
            tag = _size_tag(model_name)
            color = COLA_PALETTE.get(tag, "#9467bd")
            marker = COLA_MARKERS.get(tag, "o")
            label = f"COLA – {_short_model_label(model_name)}"

            ax.plot(
                grp["n_used"],
                grp["total_time_s"],
                marker=marker,
                color=color,
                linewidth=2,
                markersize=7,
                label=label,
                zorder=4,
            )

        ax.set_xlabel("Pool size (number of samples)")
        ax.set_ylabel("Total selection time (s)")
        ax.set_title(f"Scalability — {ds}")
        ax.legend(frameon=True, fancybox=False, edgecolor="0.8")

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"scalability_{ds}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_combined(df: pd.DataFrame, output_dir: str):
    """
    Single figure with one subplot per dataset.
    """
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes_flat[idx]
        sub = df[df["dataset"] == ds]

        # Our method
        ours = sub[sub["method"] == "ours"].sort_values("n_used")
        if not ours.empty:
            ax.plot(
                ours["n_used"],
                ours["total_time_s"],
                marker=OURS_MARKER,
                color=OURS_COLOR,
                linewidth=2,
                markersize=7,
                label="Ours (model-free)",
                zorder=5,
            )

        # COLA models
        cola = sub[sub["method"] == "cola"]
        for model_name, grp in cola.groupby("model"):
            grp = grp.sort_values("n_used")
            tag = _size_tag(model_name)
            color = COLA_PALETTE.get(tag, "#9467bd")
            marker = COLA_MARKERS.get(tag, "o")
            label = f"COLA – {_short_model_label(model_name)}"

            ax.plot(
                grp["n_used"],
                grp["total_time_s"],
                marker=marker,
                color=color,
                linewidth=2,
                markersize=7,
                label=label,
                zorder=4,
            )

        ax.set_xlabel("Pool size")
        ax.set_ylabel("Time (s)")
        ax.set_title(ds)

    # Remove unused subplots
    for idx in range(n, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    # Single shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(handles), 4),
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        bbox_to_anchor=(0.5, 1.05),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, "scalability_combined.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_breakdown(df: pd.DataFrame, output_dir: str):
    """
    Stacked bar chart: tokenization vs selection time, grouped by method and pool fraction.
    One figure per dataset.
    """
    datasets = df["dataset"].unique()

    for ds in datasets:
        sub = df[df["dataset"] == ds].copy()
        sub["label"] = sub.apply(
            lambda r: "Ours" if r["method"] == "ours" else f"COLA {_size_tag(r['model'])}",
            axis=1,
        )

        methods = sub["label"].unique()
        fractions = sorted(sub["fraction"].unique())
        n_methods = len(methods)
        x = np.arange(len(fractions))
        width = 0.8 / n_methods

        fig, ax = plt.subplots(figsize=(6, 4))

        for i, method in enumerate(methods):
            m_data = sub[sub["label"] == method].sort_values("fraction")
            tok = m_data["tokenization_time_s"].values
            sel = m_data["selection_time_s"].values

            offset = x + (i - n_methods / 2 + 0.5) * width

            ax.bar(offset, tok, width, label=f"{method} – tokenization", alpha=0.7)
            ax.bar(offset, sel, width, bottom=tok, label=f"{method} – selection", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(f*100)}%" for f in fractions])
        ax.set_xlabel("Fraction of dataset")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Time breakdown — {ds}")
        ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="0.8")

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"scalability_breakdown_{ds}.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scalability results.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/scalability_results.csv",
        help="Path to scalability CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/scalability",
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_scalability(df, args.output_dir)
    plot_combined(df, args.output_dir)
    plot_breakdown(df, args.output_dir)
