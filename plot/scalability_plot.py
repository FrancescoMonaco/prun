"""
Plot scalability results: selection time vs. data-pool size for our method and COLA.

Usage:
    python plot/scalability_plot.py --input results/scalability_results.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams.update(
        {
            #"text.usetex": True,
            #"text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 15,
            "legend.title_fontsize": 11,
            "figure.titlesize": 12,
            "axes.spines.right": False, # Disable top and left spines by default (Tufte style)
            "axes.spines.top": False,
            
        }
    )

sns.set_theme(palette="muted", style="white", font_scale=1.5)

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


def plot_all_in_one(df: pd.DataFrame, output_dir: str):
    """
    Single figure: all datasets on the same axes.
    X-axis  = fraction of the dataset.
    Y-axis  = total selection time (s).
    Colors  = method / model  (Ours, COLA-3B, COLA-11B, COLA-70B).
    Markers = dataset.
    """
    from matplotlib.lines import Line2D

    datasets = sorted(df["dataset"].unique())

    # One marker per dataset
    _marker_pool = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "+"]
    dataset_markers = {
        ds: _marker_pool[i % len(_marker_pool)] for i, ds in enumerate(datasets)
    }

    fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [4, 1]})

    # Track which method/model keys have already been added to the legend
    method_legend: dict = {}

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        mkr = dataset_markers[ds]

        # ── Our method ──
        ours = sub[sub["method"] == "ours"].sort_values("fraction")
        if not ours.empty:
            key = "Ours (model-free)"
            (line,) = ax.plot(
                ours["fraction"],
                ours["total_time_s"],
                marker=mkr,
                color=OURS_COLOR,
                linewidth=1.5,
                markersize=7,
                linestyle="-",
                label="_nolegend_",
                zorder=5,
            )
            if key not in method_legend:
                method_legend[key] = line

        # ── COLA models ──
        cola = sub[sub["method"] == "cola"]
        for model_name, grp in cola.groupby("model"):
            grp = grp.sort_values("fraction")
            tag = _size_tag(model_name)
            color = COLA_PALETTE.get(tag, "#9467bd")
            key = f"COLA \u2013 {_short_model_label(model_name)}"

            (line,) = ax.plot(
                grp["fraction"],
                grp["total_time_s"],
                marker=mkr,
                color=color,
                linewidth=1.5,
                markersize=7,
                linestyle="-",
                label="_nolegend_",
                zorder=4,
            )
            if key not in method_legend:
                method_legend[key] = line

    # ── Build two-part legend ──────────────────────────────────────────────────
    # Part 1: colors → methods/models
    color_handles = list(method_legend.values())
    color_labels = list(method_legend.keys())

    # Part 2: markers → datasets  (neutral gray line so only the shape reads)
    marker_handles = [
        Line2D(
            [0], [0],
            marker=dataset_markers[ds],
            color="0.4",
            linestyle="-",
            markersize=7,
            linewidth=1.5,
            label=ds,
        )
        for ds in datasets
    ]

    # Separator entry (invisible) between the two groups
    sep = Line2D([], [], linestyle="none", label="")

    all_handles = color_handles + [sep] + marker_handles
    all_labels = color_labels + [""] + datasets

    # Put legend in the right subfigure
    ax_legend.axis("off")
    leg = ax_legend.legend(
        all_handles,
        all_labels,
        frameon=True,
        edgecolor="0.8",
        loc="center left",
        ncol=1,
        fontsize=10,
        borderaxespad=0.0,
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    ax.set_xlabel("Fraction of dataset")
    ax.set_ylabel("Total selection time (s)")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_title("Scalability: selection time vs. dataset fraction")
    # Tufte styling, left axis spanning only along our max min time range
    sns.despine(ax=ax)
    ax.spines["left"].set_bounds(df[df["method"] == "ours"]["total_time_s"].min(), 1.5 * df[df["method"] == "ours"]["total_time_s"].max())
    fig.tight_layout()
    out_path = os.path.join(output_dir, "scalability_all.pdf")
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

    plot_all_in_one(df, args.output_dir)