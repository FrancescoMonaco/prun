import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
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
    all_data = pd.DataFrame(columns=["pruning_type", "layer", "mean", "variance"])
    for file in files:
        data = pd.read_csv(os.path.join(data_dir, file))
        # Add the column names
        data.columns = ["pruning_type", "layer", "mean", "variance"]
        all_data = pd.concat([all_data, data], ignore_index=True)
    
    print(all_data.head())
    fig, ax = plt.subplots(figsize=(15, 6), ncols=3, nrows=1, layout="constrained", gridspec_kw={'width_ratios': [3, 3, 0.5]})
    sns.set_theme(style="white")

    strip1 = sns.stripplot(
        data=all_data,
        x="layer",
        y="variance",
        hue="pruning_type",
        dodge=True,
        alpha=0.7,
        ax=ax[0],
        jitter=0.3
    )
    # Rotate ticks
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
    ax[0].set_title(f"Variance Distribution per Layer")
    ax[0].set_xlabel("Layer")
    ax[0].set_ylabel("Wanda Variance")
    # Remove legend from the first plot
    ax[0].legend_.remove()

    strip2 = sns.stripplot(
        data=all_data,
        x="layer",
        y="mean",
        hue="pruning_type",
        dodge=True,
        alpha=0.7,
        ax=ax[1],
        jitter=0.3,
        legend=False
    )
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    ax[1].set_title(f"Mean Distribution per Layer")
    ax[1].set_xlabel("Layer")
    ax[1].set_ylabel("Wanda Mean")

    # Add legend to the third plot
    handles, labels = ax[0].get_legend_handles_labels()
    ax[2].axis('off')
    ax[2].legend(handles, labels, title="Pruning Type", loc='center')

    plt.suptitle(f"Wanda Analysis on {dataset_name}", fontsize=16)
    plt.savefig(f"wanda_variance_{dataset_name}.pdf", dpi=200)
        
        