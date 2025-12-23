import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compare activations across pruning types")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Qwen-Qwen3-1.7B)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., winogrande)")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of samples")
    parser.add_argument("--output_dir", type=str, default="plots/comparison", help="Directory to save plots")
    
    args = parser.parse_args()
    
    model_name = args.model.replace("/", "-")
    dataset_name = args.dataset
    nsamples = args.nsamples
    
    results_root = "results"
    model_path = os.path.join(results_root, model_name)
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return
    
    all_data = []
    
    # Find all pruning types
    pruning_types = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    
    for pt in pruning_types:
        # Check if nsamples directory exists
        nsamples_path = os.path.join(model_path, pt, str(nsamples))
        if not os.path.exists(nsamples_path):
            continue
            
        file_path = os.path.join(nsamples_path, f"{dataset_name}.txt")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, skipinitialspace=True)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"Warning: File {file_path} not found.")
            
    if not all_data:
        print(f"No data found for model {model_name}, dataset {dataset_name}, nsamples {nsamples}.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate Coefficient of Variation (CV) and Normalized Variance
    # CV = std / mean. This relates variance to the magnitude of activations.
    combined_df["activations_cv"] = np.sqrt(combined_df["activations_var"]) / (combined_df["activations_mean"] + 1e-8)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set style
    sns.set_theme(style="white", palette="muted")
    
    # Shorten layer names for better display
    def shorten_layer(name):
        return name.replace("model.layers.", "L").replace(".self_attn", "").replace(".mlp", "")

    combined_df["layer_short"] = combined_df["layer"].apply(shorten_layer)
    
    # Ensure layers are in the original order
    layer_order = all_data[0]["layer"].tolist()
    layer_short_order = [shorten_layer(l) for l in layer_order]
    
    metrics = ["activations_mean", "activations_var", "activations_cv", "wanda_mean"]
    titles = {
        "activations_mean": "Mean Activations",
        "activations_var": "Activation Variance",
        "activations_cv": "Coefficient of Variation (Std/Mean)",
        "wanda_mean": "Mean Wanda Score"
    }
    
    # --- 1. Line Plots (Original Metrics) ---
    for metric in metrics:
        # 1a. Standard Scale
        plt.figure(figsize=(20, 10))
        sns.lineplot(
            data=combined_df, 
            x="layer_short", 
            y=metric, 
            hue="pruning_type", 
            marker="o", 
            linewidth=2,
            markersize=6
        )
        plt.title(f"{titles[metric]} across Layers\nModel: {model_name} | Dataset: {dataset_name} | N={nsamples}", fontsize=16)
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.legend(title="Pruning Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_name = f"compare_{metric}_{model_name}_{dataset_name}.pdf"
        plt.savefig(os.path.join(args.output_dir, save_name), bbox_inches='tight')
        plt.close()

        # 1b. Log Scale (to handle outliers)
        plt.figure(figsize=(20, 10))
        sns.lineplot(
            data=combined_df, 
            x="layer_short", 
            y=metric, 
            hue="pruning_type", 
            marker="o", 
            linewidth=2,
            markersize=6
        )
        plt.yscale('log')
        plt.title(f"{titles[metric]} (Log Scale)\nModel: {model_name} | Dataset: {dataset_name}", fontsize=16)
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel(f"{metric.replace('_', ' ').title()} (Log)", fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.legend(title="Pruning Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_name = f"compare_log_{metric}_{model_name}_{dataset_name}.pdf"
        plt.savefig(os.path.join(args.output_dir, save_name), bbox_inches='tight')
        plt.close()

    # --- 1c. Scatter Plot: Mean vs Variance (Relationship Analysis) ---
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=combined_df,
        x="activations_mean",
        y="activations_var",
        hue="pruning_type",
        alpha=0.6,
        edgecolor=None
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Activation Mean vs Variance (Log-Log)\nModel: {model_name} | Dataset: {dataset_name}", fontsize=16)
    plt.xlabel("Mean Activation (Log)", fontsize=12)
    plt.ylabel("Activation Variance (Log)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    save_name = f"scatter_mean_vs_var_{model_name}_{dataset_name}.pdf"
    plt.savefig(os.path.join(args.output_dir, save_name), bbox_inches='tight')
    plt.close()
    print(f"Saved {save_name}")

    # --- 2. Difference Analysis (Relative to Random) ---
    if "random" in combined_df["pruning_type"].unique():
        random_df = combined_df[combined_df["pruning_type"] == "random"].copy()
        random_df = random_df.set_index("layer")
        
        diff_data = []
        for pt in combined_df["pruning_type"].unique():
            if pt == "random": continue
            pt_df = combined_df[combined_df["pruning_type"] == pt].copy()
            pt_df = pt_df.set_index("layer")
            
            for metric in metrics:
                # Relative difference: (pt - random) / abs(random)
                pt_df[f"{metric}_diff"] = (pt_df[metric] - random_df[metric]) / (random_df[metric].abs() + 1e-8)
            
            pt_df["pruning_type"] = pt
            pt_df["layer_short"] = pt_df.index.map(shorten_layer)
            diff_data.append(pt_df.reset_index())
            
        if diff_data:
            diff_df = pd.concat(diff_data, ignore_index=True)
            
            # --- 2a. Summary Bar Plot (Median Relative Difference) ---
            plt.figure(figsize=(12, 8))
            summary_data = []
            for metric in metrics:
                temp = diff_df.groupby("pruning_type")[f"{metric}_diff"].median().reset_index()
                temp["metric"] = titles[metric]
                temp.columns = ["pruning_type", "median_diff", "metric"]
                summary_data.append(temp)
            
            summary_df = pd.concat(summary_data)
            
            ax = sns.barplot(data=summary_df, x="metric", y="median_diff", hue="pruning_type")
            plt.axhline(0, color='black', linestyle='-', alpha=0.3)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.title(f"Median Relative Difference vs Random (Robust to Outliers)\nModel: {model_name} | Dataset: {dataset_name}", fontsize=16)
            plt.ylabel("Median Relative Difference", fontsize=12)
            plt.legend(title="Pruning Type", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            save_name = f"summary_median_diff_{model_name}_{dataset_name}.pdf"
            plt.savefig(os.path.join(args.output_dir, save_name), bbox_inches='tight')
            plt.close()
            print(f"Saved {save_name}")

            # --- 2b. Heatmap of Differences ---
            # Good for seeing patterns across layers without being dominated by outliers
            for metric in metrics:
                plt.figure(figsize=(20, 8))
                pivot_df = diff_df.pivot(index="pruning_type", columns="layer_short", values=f"{metric}_diff")
                # Reorder columns to match layer order
                pivot_df = pivot_df[layer_short_order]
                
                # Use a robust scale for the heatmap to handle outliers
                v_limit = np.nanpercentile(np.abs(pivot_df.values), 95)
                if v_limit == 0: v_limit = 1.0 # Avoid division by zero or flat scale
                
                sns.heatmap(pivot_df, cmap="RdBu_r", center=0, vmin=-v_limit, vmax=v_limit,
                            cbar_kws={'label': 'Relative Difference'})
                
                plt.title(f"Heatmap: Relative Difference in {titles[metric]} (vs Random)\nModel: {model_name} | Dataset: {dataset_name}", fontsize=16)
                plt.xlabel("Layer", fontsize=12)
                plt.ylabel("Pruning Type", fontsize=12)
                plt.tight_layout()
                save_name = f"heatmap_diff_{metric}_{model_name}_{dataset_name}.pdf"
                plt.savefig(os.path.join(args.output_dir, save_name), bbox_inches='tight')
                plt.close()
                print(f"Saved {save_name}")

            # --- 2c. Boxplot of Differences ---
            # Shows the distribution of differences across layers
            for metric in metrics:
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=diff_df, x="pruning_type", y=f"{metric}_diff")
                plt.axhline(0, color='black', linestyle='--', alpha=0.5)
                plt.title(f"Distribution of Relative Differences in {titles[metric]}\nModel: {model_name} | Dataset: {dataset_name}", fontsize=16)
                plt.ylabel("Relative Difference", fontsize=12)
                plt.xlabel("Pruning Type", fontsize=12)
                plt.tight_layout()
                save_name = f"boxplot_diff_{metric}_{model_name}_{dataset_name}.pdf"
                plt.savefig(os.path.join(args.output_dir, save_name), bbox_inches='tight')
                plt.close()
                print(f"Saved {save_name}")

            # --- 2d. Line Plot with SymLog Scale ---
            # Handles large differences while keeping small ones visible
            for metric in metrics:
                plt.figure(figsize=(20, 10))
                ax = sns.lineplot(
                    data=diff_df, 
                    x="layer_short", 
                    y=f"{metric}_diff", 
                    hue="pruning_type", 
                    marker="s", 
                    linewidth=2
                )
                plt.axhline(0, color='black', linestyle='--', alpha=0.5)
                plt.yscale('symlog', linthresh=0.1) # Linear around 0, log elsewhere
                plt.title(f"Relative Difference in {titles[metric]} (SymLog Scale)\nModel: {model_name} | Dataset: {dataset_name}", fontsize=16)
                plt.xlabel("Layer", fontsize=12)
                plt.ylabel("Relative Difference (SymLog)", fontsize=12)
                plt.xticks(rotation=90, fontsize=8)
                plt.legend(title="Pruning Type", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                save_name = f"diff_symlog_{metric}_{model_name}_{dataset_name}.pdf"
                plt.savefig(os.path.join(args.output_dir, save_name), bbox_inches='tight')
                plt.close()
                print(f"Saved {save_name}")

    print(f"\nAll comparison plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
