import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def plot_ks_heatmap(df, m1, m2, output_dir, model_name):
    """
    Heatmap of KS statistics for a specific pair of methods (m1 vs m2).
    """
    metrics = df['metric'].unique()
    
    for metric in metrics:
        subset = df[df['metric'] == metric].copy()
        if len(subset) == 0: continue
        
        # Create a pivot table: Index=Layer, Columns=Dataset, Values=KS Statistic
        pivot_table = subset.pivot_table(index='layer', columns='dataset', values='ks_statistic')
        
        # Adjust figure size based on amount of data
        plt.figure(figsize=(12, len(pivot_table) * 0.4 + 3))
        sns.heatmap(pivot_table, annot=True, cmap="magma", fmt=".3f", linewidths=.5)
        plt.title(f"KS Statistic: {m1} vs {m2}\nmetric: {metric}\nModel: {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_ks_{metric}_{m1}_vs_{m2}_{model_name}.png"))
        plt.close()

def plot_stat_comparison(df, m1, m2, metric_name, stat_type, output_dir, model_name):
    """
    Bar plot comparing m1 vs m2 for a specific statistic (e.g. mean, max).
    We use the columns 'mean_1' and 'mean_2' etc from the dataframe.
    """
    subset = df[df['metric'] == metric_name].copy()
    if len(subset) == 0: return
    
    col_1 = f"{stat_type}_1"
    col_2 = f"{stat_type}_2"
    
    if col_1 not in subset.columns: # fallback if column names missing
         return

    # Melt for seaborn
    melted = subset.melt(
        id_vars=['dataset', 'layer'], 
        value_vars=[col_1, col_2],
        var_name='Sampling', 
        value_name='Value'
    )
    
    # Rename for legend
    melted['Sampling'] = melted['Sampling'].map({col_1: m1, col_2: m2})
    
    g = sns.catplot(
        data=melted, kind="bar",
        x="Value", y="layer", hue="Sampling",
        col="dataset", col_wrap=3,
        height=5, aspect=1.5,
        palette="muted", sharex=False
    )
    
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Comparison of {stat_type.capitalize()} for {metric_name}\n({m1} vs {m2}) - {model_name}")
    
    save_path = os.path.join(output_dir, f"comparison_{metric_name}_{stat_type}_{m1}_vs_{m2}_{model_name}.png")
    g.savefig(save_path)
    plt.close()

def plot_difference_distribution(df, m1, m2, output_dir, model_name):
    """
    Plot distribution of differences (m2 - m1) or usually (Zipf - Baseline).
    """
    # Columns are mean_diff, max_diff
    cols = ['mean_diff', 'max_diff']
    
    for col in cols:
        if col not in df.columns: continue
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue="metric", kde=True, element="step", multiple="dodge")
        plt.axvline(x=0, color='black', linestyle='--')
        plt.title(f"Distribution of {col} ({m2} - {m1})\nModel: {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dist_{col}_{m1}_vs_{m2}_{model_name}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot analysis from P-Values CSV")
    parser.add_argument("--input_csv", type=str, default="results/pvalues_detailed.csv", help="Input CSV file")
    parser.add_argument("--output_dir", type=str, default="plots/pvalue_analysis", help="Output directory for plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"File not found: {args.input_csv}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # Handle legacy CSVs without 'model' column
    if 'model' not in df.columns:
        df['model'] = 'default'
    
    # If using old CSV format (no method_1/method_2 columns), we can't do much with this script
    if 'method_1' not in df.columns:
        print("Error: CSV format appears to be legacy (missing 'method_1' column).")
        return

    # Iterate over Models
    unique_models = df['model'].unique()
    for model_name in unique_models:
        model_df = df[df['model'] == model_name]
        safe_model_name = str(model_name).replace("/", "-")
        
        print(f"Processing Model: {model_name}")
        
        # Determine unique pairs like (random, zipf), (lorem, zipf)
        # We process each pair separately to keep plots clean
        pairs = model_df[['method_1', 'method_2']].drop_duplicates().values
        
        for m1, m2 in pairs:
            print(f"  Comparing {m1} vs {m2}...")
            pair_df = model_df[(model_df['method_1'] == m1) & (model_df['method_2'] == m2)]
            
            # 1. Heatmap
            plot_ks_heatmap(pair_df, m1, m2, args.output_dir, safe_model_name)
            
            # 2. Bar Charts
            # For each metric type (wanda_score, activation_rms...)
            unique_metrics = pair_df['metric'].unique()
            for met in unique_metrics:
                plot_stat_comparison(pair_df, m1, m2, met, "mean", args.output_dir, safe_model_name)
                plot_stat_comparison(pair_df, m1, m2, met, "max", args.output_dir, safe_model_name)
            
            # 3. Distributions
            plot_difference_distribution(pair_df, m1, m2, args.output_dir, safe_model_name)

    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
