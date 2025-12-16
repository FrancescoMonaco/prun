import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib
import argparse

matplotlib.use("WebAgg")  # Use a non-interactive backend


def extract_layer_num(s):
    # extract first integer in the string (returns np.nan if none)
    if pd.isna(s):
        return np.nan
    import re

    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else np.nan


if __name__ == "__main__":
    # Pass in input the dataset we want to plot the variance for
    parser = argparse.ArgumentParser(description="Plot Wanda Variance Distributions")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to plot variance distributions for",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Name of the model to plot variance distributions for",
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.model.replace("/", "-")

    # The data is in files named 'wanda_{model_name}_*_{dataset_name}.txt' where * is the pruning type
    data_dir = "results"
    files = [
        f
        for f in os.listdir(data_dir)
        if f.startswith(f"wanda_{model_name}_") and f.endswith(f"_{dataset_name}.txt")
    ]
    # Add to a DataFram with columns: 'pruning_type', 'layer', 'wanda_mean', 'wanda_variance', 'mean_activations','var_activations'
    all_data = pd.DataFrame()
    for file in files:
        data = pd.read_csv(os.path.join(data_dir, file))
        # Add the column names
        all_data = pd.concat([all_data, data], ignore_index=True)

    # TODO Remove this line after rerunnning the experiments
    # Strip column names to remove the initial space
    all_data.columns = all_data.columns.str.strip()

    # Create the column for the layer type
    # Structure is usually: model.layers.0.self_attn.q_proj
    all_data["component"] = all_data["layer"].apply(lambda x: x.split(".")[-1])
    # FInd the layer number with no split on the point
    all_data["layer_num"] = all_data["layer"].apply(extract_layer_num)
    # Drop rows with NaN layer_num
    all_data = all_data.dropna(subset=["layer_num"])

    all_data["layer_num"] = all_data["layer_num"].astype(int)
    all_data = all_data.sort_values(by="layer_num")

    # Calcola il coefficiente di variazione (CV)

    all_data["cv"] = (
        np.sqrt(all_data["var_activations"]) / all_data["mean_activations"].abs()
    )

    # FOCUS on just a few layers for visibility
    # all_data = all_data[all_data[" layer"] >= (all_data[" layer"].max() - 7)]
    nrows = all_data["component"].nunique()
    fig, axs = plt.subplots(
        figsize=(12, 16),
        ncols=3,
        nrows=nrows,
        layout="constrained",
        gridspec_kw={"width_ratios": [3, 3, 0.5]},
        sharex=True,
    )
    sns.set_theme(style="white")
    gs = axs[0, -1].get_gridspec()
    for ax in axs[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])
    # Map pruning types to colors so lines and mean dashes match
    pruning_types = sorted(list(all_data["pruning_type"].unique()))
    palette = sns.color_palette("tab10", n_colors=max(1, len(pruning_types)))
    color_map = dict(zip(pruning_types, palette))

    for i, component in enumerate(sorted(all_data["component"].unique())):
        component_data = all_data[all_data["component"] == component]
        # Line plot of variance per layer for each pruning type

        sns.lineplot(
            data=component_data,
            x="layer_num",
            y="activations_variance",
            hue="pruning_type",
            marker="o",
            errorbar=None,
            ax=axs[i, 0],
            palette=color_map,
        )

        for pruning_type in pruning_types:
            subset = component_data[component_data["pruning_type"] == pruning_type]
            mean_pruning = subset["cv"].mean()
            color = color_map.get(pruning_type)
            axs[i, 0].hlines(
                y=mean_pruning,
                xmin=component_data["layer_num"].min(),
                xmax=component_data["layer_num"].max(),
                colors=color,
                linestyles="dashed",
                linewidth=1.5,
                alpha=0.9,
            )
        axs[i, 0].set_title(f"Variance per Layer - {component}")
        axs[i, 0].set_xlabel("")
        axs[i, 0].set_ylabel("")
        # axs[i, 0].set_yscale("log")
        axs[i, 0].set_xticklabels(axs[i, 0].get_xticklabels(), rotation=45)

        # Set global y label for the first column
        if i == nrows // 2:
            axs[i, 0].set_ylabel("Activations Variance")
        if i == nrows - 1:
            axs[i, 0].set_xlabel("Layer")

        # Strip plot for mean activations
        sns.stripplot(
            data=component_data,
            x="layer_num",
            y="mean_activations",
            hue="pruning_type",
            dodge=True,
            alpha=0.7,
            ax=axs[i, 1],
            jitter=0.3,
            legend=False,
            palette=color_map,
        )
        if i == nrows // 2:
            axs[i, 1].set_ylabel("Mean Activations")
        else:
            axs[i, 1].set_ylabel("")
        axs[i, 1].set_xlabel("")
        axs[i, 1].set_xticklabels(axs[i, 1].get_xticklabels(), rotation=45)
        if i == nrows - 1:
            axs[i, 1].set_xlabel("Layer")

    # Remove legend from all plots
    for i in range(nrows):
        if getattr(axs[i, 0], "legend_", None) is not None:
            axs[i, 0].legend_.remove()
        if getattr(axs[i, 1], "legend_", None) is not None:
            axs[i, 1].legend_.remove()

    # Build a consistent legend in the third panel with proxy artists
    from matplotlib.lines import Line2D

    proxy_handles = [
        Line2D([0], [0], color=color_map[pt], marker="o", linestyle="-")
        for pt in pruning_types
    ]
    axbig.axis("off")
    axbig.legend(proxy_handles, pruning_types, title="Pruning Type", loc="center")

    plt.suptitle(f"Variance Analysis for {model_name} on {dataset_name}", fontsize=16)
    plt.savefig(f"variance_analysis_{model_name}_{dataset_name}.pdf", dpi=200)
