import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from matplotlib.lines import Line2D

def main():
    parser = argparse.ArgumentParser(description="Plot Intersection Analysis Results")
    parser.add_argument("--csv", type=str, default="results/intersection_test.csv", help="Path to results CSV")
    parser.add_argument("--output", type=str, default="plots/intersection_analysis.png", help="Path to save plot")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV file {args.csv} not found.")
        return

    df = pd.read_csv(args.csv)
    df['label'] = df['label'].astype(str)

    # Select the correct accuracy column
    acc_cols = [c for c in df.columns if 'acc' in c.lower() and 'stderr' not in c.lower()]
    if not acc_cols:
        print(f"Available columns: {df.columns.tolist()}")
        print("No accuracy column found in CSV.")
        return
    
    # Create a single accuracy column by taking the first non-null or max value from accuracy columns
    df['accuracy'] = df[acc_cols].max(axis=1)
    
    # Filter out rows where accuracy is null
    df = df[df['accuracy'].notnull()]

    models = sorted(df['model'].unique())
    calibration_datasets = sorted(df['calibration_dataset'].unique())
    n_models = len(models)
    n_calib = len(calibration_datasets)
    
    if n_models == 0:
        print("No models found in the data.")
        return

    # Color palette for tasks
    all_tasks = sorted(df['target_task'].unique())
    task_colors = sns.color_palette("husl", len(all_tasks))
    task_to_color = dict(zip(all_tasks, task_colors))

    fig, axes = plt.subplots(n_calib, n_models, figsize=(7 * n_models, 5 * n_calib), squeeze=False, sharey=True, layout='constrained')
    sns.set_style("white")

    for r, cal_ds in enumerate(calibration_datasets):
        for c, model_name in enumerate(models):
            ax = axes[r, c]
            subset_df = df[(df['model'] == model_name) & (df['calibration_dataset'] == cal_ds)]
            tasks = sorted(subset_df['target_task'].unique())
            
            if len(subset_df) == 0:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            for task in tasks:
                task_df = subset_df[subset_df['target_task'] == task].copy()
                task_df = task_df.sort_values('n_samples')
                
                # Intersection point
                intersection_df = task_df[task_df['label'] == 'intersection']
            
                # Tail addition
                tail_df = task_df[task_df['label'].str.contains('tail_add', na=False)]
                # Combine intersection and tail
                tail_series = pd.concat([intersection_df, tail_df]).drop_duplicates('n_samples').sort_values('n_samples')
            
                # Random addition
                random_df = task_df[task_df['label'].str.contains('random_add', na=False)]
                # Combine intersection and random
                random_series = pd.concat([intersection_df, random_df]).drop_duplicates('n_samples').sort_values('n_samples')
            
                if len(tail_series) > 0:
                    sns.lineplot(
                        x='n_samples',
                        y='accuracy',
                        data=tail_series,
                        marker='o',
                        ax=ax,
                        color=task_to_color[task],
                        linestyle='-'
                    )

                if len(random_series) > 0:
                    sns.lineplot(
                        x='n_samples',
                        y='accuracy',
                        data=random_series,
                        marker='x',
                        ax=ax,
                        color=task_to_color[task],
                        linestyle='--',
                        alpha=0.8
                    )

            if r == 0:
                ax.set_title(f"Model: {model_name.split('/')[-1]}", fontsize=14)
            
            if c == 0:
                ax.set_ylabel(f"{cal_ds}\nAccuracy", fontsize=12)
            else:
                ax.set_ylabel("")
                
            ax.set_xlabel("N Samples", fontsize=12)

    # Create common legend handles
    task_legend_handles = [Line2D([0], [0], color=task_to_color[t], lw=2, label=str(t)) for t in all_tasks]
    style_legend_handles = [
        Line2D([0], [0], color='gray', linestyle='-', marker='o', label='Tail Addition'),
        Line2D([0], [0], color='gray', linestyle='--', marker='x', label='Random Addition')
    ]
    
    # Place legend on the last axis or outside
    axes[0, -1].legend(handles=task_legend_handles + style_legend_handles, 
                       title="Tasks & Methods", 
                       loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.suptitle("Impact of Tail Samples (Unique Tokens) vs Random Samples on Pruning", fontsize=16)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
