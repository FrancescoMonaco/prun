import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import re

def parse_layer_index(layer_name):
    # Extracts number from "layer_0_..."
    match = re.search(r'layer_(\d+)_', layer_name)
    if match:
        return int(match.group(1))
    return -1

def main():
    parser = argparse.ArgumentParser(description="Plot First Layers Overlap")
    parser.add_argument("--csv", type=str, default="results/first_layers_overlap.csv", help="Path to results CSV")
    parser.add_argument("--output_dir", type=str, default="plots/first_layers", help="Directory to save plots")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV file {args.csv} not found.")
        return

    df = pd.read_csv(args.csv)
    
    # Extract layer index
    df['layer_idx'] = df['layer'].apply(parse_layer_index)
    
    # Filter out invalid parsing
    df = df[df['layer_idx'] >= 0]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    models = df['model'].unique()
    
    # Plot 1: Aggregated overlap per layer (averaged across modules) for each dataset
    # Facet by Model
    sns.set_style("whitegrid")
    
    for model in models:
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Determine if we want to average across modules (q,k,v,o, etc) or show distribution
        # Let's plot the mean overlap per layer index, with hue=dataset
        
        sns.lineplot(
            data=model_df,
            x='layer_idx',
            y='overlap',
            hue='dataset',
            marker='o',
            linewidth=2
        )
        
        plt.title(f"Mask Overlap (Random vs Unique Tokens) - {model.split('/')[-1]}")
        plt.xlabel("Layer Index")
        plt.ylabel("Mask Overlap")
        plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        model_name_clean = model.replace('/', '_')
        plt.savefig(os.path.join(args.output_dir, f"{model_name_clean}_overlap_line.png"), dpi=300)
        plt.close()
        
        # Plot 2: Detailed boxplot per module type if needed?
        # Maybe just a heatmap?
        # Let's do a heatmap of (Dataset x Layer) for Overlap
        
        # Pivot table: Index=Dataset, Columns=Layer_Idx, Values=Overlap (mean over modules)
        pivot_df = model_df.groupby(['dataset', 'layer_idx'])['overlap'].mean().unstack()
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".3f")
        plt.title(f"Mean Overlap per Layer - {model.split('/')[-1]}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{model_name_clean}_overlap_heatmap.png"), dpi=300)
        plt.close()

    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
