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
            "font.size": 16,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 11,
            "figure.titlesize": 13,
            "axes.spines.right": False, # Disable top and left spines by default (Tufte style)
            "axes.spines.top": False,
            
        }
    )

sns.set_theme(palette="muted", style="white", font_scale=1.5)

# ── Consistent palette ──────────────────────────────────────────────────────────
# Colors for methods
OURS_COLOR = "#2ca02c"
COLA_COLOR = "#1f77b4"

# Line styles for COLA models
COLA_LINESTYLES = {"3B": "-", "8B": "--", "70B": ":"}

# Markers for datasets
DATASET_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "+"]


def _short_model_label(name: str) -> str:
    """Turn 'meta-llama/Llama-3.2-70B' → 'Llama-3.2-70B' and extract size tag."""
    short = name.split("/")[-1] if "/" in name else name
    return short


def _size_tag(name: str) -> str:
    """Extract '3B' / '8B' / '70B' from a model name string."""
    for tag in ("70B", "8B", "3B"):
        if tag.lower() in name.lower():
            return tag
    return name.split("/")[-1]


def plot_all_in_one(df: pd.DataFrame, output_dir: str):
    """
    Two-panel figure (1 row × 2 cols), one subplot per dataset, shared y-axis.
    X-axis  = fraction of the dataset.
    Y-axis  = total selection time (s), log scale, shared across panels.
    Colors  = method (Ours, COLA).
    Line styles = COLA model size (3B solid, 8B dashed, 70B dotted).
    Tufte-style: no top/right spines, left spine bounded to data range.
    """
    from matplotlib.lines import Line2D

    datasets = sorted(df["dataset"].unique())
    DATASET_LABELS = {
        "arc_challenge": "ARC-C",
        "winogrande": "WinoGrande",
        "arc_challenge (n≈10⁴)": "ARC-C",
        "winogrande (n≈10⁵)": "WinoGrande",
    }

    # Two equal columns, legend placed at the top of the figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, wspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    data_axes = [ax0, ax1]

    # Track legend entries: key -> (color, linestyle)
    legend_entries: dict = {}

    ours_df = df[df["method"] == "ours"]
    y_min = ours_df["total_time_s"].min()
    y_max = 1.5 * ours_df["total_time_s"].max()

    for ax, ds in zip(data_axes, datasets):
        sub = df[df["dataset"] == ds]

        # ── Our method ──
        ours = sub[sub["method"] == "ours"].sort_values("fraction")
        if not ours.empty:
            key = "ZipCal"
            ax.plot(
                ours["fraction"],
                ours["total_time_s"],
                marker="o",
                color=OURS_COLOR,
                linewidth=4.0,
                markersize=7,
                linestyle="-",
                label="_nolegend_",
                zorder=5,
            )
            if key not in legend_entries:
                legend_entries[key] = (OURS_COLOR, "-")

        # ── COLA models ──
        cola = sub[sub["method"] == "cola"]
        for model_name, grp in cola.groupby("model"):
            grp = grp.sort_values("fraction")
            tag = _size_tag(model_name)
            linestyle = COLA_LINESTYLES.get(tag, "-")
            key = f"COLA \u2013 {_short_model_label(model_name)}"
            ax.plot(
                grp["fraction"],
                grp["total_time_s"],
                marker="o",
                color=COLA_COLOR,
                linewidth=4.0,
                markersize=7,
                linestyle=linestyle,
                label="_nolegend_",
                zorder=4,
            )
            if key not in legend_entries:
                legend_entries[key] = (COLA_COLOR, linestyle)

        ax.set_title(DATASET_LABELS.get(ds, ds))
       # ax.set_xlabel("Fraction of dataset")
        ticks = sorted(sub["fraction"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            ["1" if t == 1.0 else f"{t:g}" for t in ticks]
        )
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_yticks([1, 60, 3600])
        ax.set_yticklabels(["1s", "1m", "1h"])
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

        # Dotted reference lines dividing seconds / minutes / hours
        for boundary in (1, 60, 3600):
            ax.axhline(boundary, color="0.6", linewidth=1.8, linestyle=":", zorder=1)


        # Tufte styling: remove top/right spines; bound left spine to data range
        sns.despine(ax=ax)
        ax.spines["left"].set_bounds(y_min, y_max)

    # Y-axis label and ticks only on the left panel
    ax0.set_ylabel("Total selection time")
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.tick_params(axis="y", which="both", left=False)
    ax1.spines["left"].set_visible(False)

    # ── Legend at the top of the figure ──────────────────────────────────────
    method_handles = [
        Line2D(
            [0], [0],
            marker="",
            color=color,
            linestyle=ls,
            linewidth=2.9,
            label=key,
        )
        for key, (color, ls) in legend_entries.items()
    ]

    leg = fig.legend(
        method_handles,
        list(legend_entries.keys()),
        frameon=False,
        loc="upper center",
        ncol=len(legend_entries),
        fontsize=16,
        # bbox_to_anchor=(0.5, 1.02),
        # borderaxespad=0.0,
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    #fig.suptitle("Scalability: selection time vs. dataset fraction", y=1.10, fontsize=18)
    out_path = os.path.join(output_dir, "scalability_all.pdf")
    fig.supxlabel("Fraction of dataset", y=0.02)
    fig.savefig(out_path, dpi=400)#, bbox_inches="tight")
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