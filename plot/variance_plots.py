import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib
import argparse
matplotlib.use("WebAgg")  # Use a non-interactive backend

if __name__ == "__main__":
    # Pass in input the dataset we want to plot the variance for
    parser = argparse.ArgumentParser(description="Plot Wanda Variance Distributions")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to plot variance distributions for",
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    
    # The data is in files named 'wanda_*_{dataset_name}.txt'
    data_dir = "results"
    files = [f for f in os.listdir(data_dir) if f.endswith(f"_{dataset_name}.txt")]
    # Add to a DataFram with columns: 'pruning_type', 'layer', 'mean', 'variance'
    all_data = pd.DataFrame()
    for file in files:
        data = pd.read_csv(os.path.join(data_dir, file))
        # Add the column names
        all_data = pd.concat([all_data, data], ignore_index=True)
    
    # Calcola il coefficiente di variazione (CV)

    all_data["cv"] = np.sqrt(all_data[" var_activations"]) / all_data[" mean_activations"].abs()
    
    # Select only the last 20 layers, since the layes column contains names "model.layers.6.self_attn.o_proj"
    all_data = all_data[all_data[" layer"].str.contains("layers")]
    all_data[" layer"] = all_data[" layer"].apply(lambda x: x.split(".")[2])  # Extract the layer number
    all_data[" layer"] = all_data[" layer"].astype(int)
    all_data = all_data.sort_values(by=" layer")
    # all_data = all_data[all_data[" layer"] >= (all_data[" layer"].max() - 7)]
    
    fig, ax = plt.subplots(figsize=(15, 6), ncols=3, nrows=1, layout="constrained", gridspec_kw={'width_ratios': [3, 3, 0.5]})
    sns.set_theme(style="white")

    # Map pruning types to colors so lines and mean dashes match
    pruning_types = sorted(list(all_data["pruning_type"].unique()))
    palette = sns.color_palette("tab10", n_colors=max(1, len(pruning_types)))
    color_map = dict(zip(pruning_types, palette))

    sns.lineplot(
        data=all_data,
        x=" layer",
        y=" var_activations",
        hue="pruning_type",
        marker="o",
        errorbar=None,
        ax=ax[0],
        palette=color_map,
    )

    # capture the legend handles/labels before removing local legend
    handles, labels = ax[0].get_legend_handles_labels()
    if getattr(ax[0], "legend_", None) is not None:
        ax[0].legend_.remove()

    # Draw for each pruning_type the mean (dashed) line using the same mapped color
    for pruning_type in pruning_types:
        subset = all_data[all_data["pruning_type"] == pruning_type]
        mean_pruning = subset[" var_activations"].mean()
        color = color_map.get(pruning_type)
        ax[0].hlines(
            y=mean_pruning,
            xmin=all_data[" layer"].min(),
            xmax=all_data[" layer"].max(),
            colors=color,
            linestyles="dashed",
            linewidth=1.5,
            alpha=0.9,
        )
        

    # strip1 = sns.stripplot(
    #     data=all_data,
    #     x=" layer",
    #     y="cv",
    #     hue="pruning_type",
    #     dodge=True,
    #     alpha=0.7,
    #     ax=ax[0],
    #     jitter=0.3
    # )
    # Rotate ticks
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    ax[0].set_title(f"Variance Distribution per Layer")
    ax[0].set_xlabel("Layer")
    ax[0].set_ylabel("Activations Variance")
    #ax[0].set_yscale("log")
    # Remove legend from the first plot if present (safe)
    if getattr(ax[0], "legend_", None) is not None:
        ax[0].legend_.remove()

    strip2 = sns.stripplot(
        data=all_data,
        x=" layer",
        y=" mean_activations",
        hue="pruning_type",
        dodge=True,
        alpha=0.7,
        ax=ax[1],
        jitter=0.3,
        legend=False,
        palette=color_map,
    )
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    ax[1].set_title(f"Mean Distribution per Layer")
    ax[1].set_xlabel("Layer")
    ax[1].set_ylabel("Wanda Mean")

    # Build a consistent legend in the third panel with proxy artists
    from matplotlib.lines import Line2D

    proxy_handles = [
        Line2D([0], [0], color=color_map[pt], marker='o', linestyle='-')
        for pt in pruning_types
    ]
    ax[2].axis('off')
    ax[2].legend(proxy_handles, pruning_types, title="Pruning Type", loc='center')

    plt.suptitle(f"Wanda Analysis on {dataset_name}", fontsize=16)
    plt.savefig(f"wanda_variance_{dataset_name}.pdf", dpi=200)
        
        