import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from matplotlib.lines import Line2D
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot Wanda Score Evolution")
    parser.add_argument("--csv", type=str, default="results/intersection_test_with_activations.csv", help="Path to results CSV")
    parser.add_argument("--output", type=str, default="plots/wanda_evolution.png", help="Path to save plot")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV file {args.csv} not found.")
        return

    df = pd.read_csv(args.csv)
    df['label'] = df['label'].astype(str)

    # Identifica le colonne dei punteggi (Wanda o Activations)
    wanda_cols = [c for c in df.columns if 'mean_wanda_layer_' in c or 'mean_act_layer_' in c]
    avg_col = 'mean_wanda_last_layers_avg' if 'mean_wanda_last_layers_avg' in df.columns else 'mean_act_last_layers_avg'
    
    if not wanda_cols and avg_col not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        print("No score columns found in CSV.")
        return

    models = sorted(df['model'].unique())
    calibration_datasets = sorted(df['calibration_dataset'].unique())
    n_models = len(models)
    n_calib = len(calibration_datasets)
    
    if n_models == 0:
        print("No models found in the data.")
        return

    fig, axes = plt.subplots(n_calib, n_models, figsize=(8 * n_models, 6 * n_calib), squeeze=False, sharey=False, layout='constrained')
    sns.set_style("whitegrid")

    # Colori per i diversi strati (o per l'avg)
    # Useremo colori diversi per ogni colonna di score e stili diversi per Tail vs Random
    all_score_cols = sorted(wanda_cols) + ([avg_col] if avg_col in df.columns else [])
    palette = sns.color_palette("muted", len(all_score_cols))
    score_to_color = dict(zip(all_score_cols, palette))

    for r, cal_ds in enumerate(calibration_datasets):
        for c, model_name in enumerate(models):
            ax = axes[r, c]
            subset_df = df[(df['model'] == model_name) & (df['calibration_dataset'] == cal_ds)]
            
            if len(subset_df) == 0:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            # Per ogni colonna di score, plottiamo Tail vs Random
            for score_col in all_score_cols:
                if score_col not in subset_df.columns:
                    continue
                
                # Aggreghiamo i dati per n_samples e label prendendo la media (per gestire i duplicati dovuti ai target_task)
                plot_data = subset_df.groupby(['label', 'n_samples'])[score_col].mean().reset_index()
                
                # Intersection point
                intersection_df = plot_data[plot_data['label'] == 'intersection']
            
                # Tail addition
                tail_df = plot_data[plot_data['label'].str.contains('tail_add', na=False)]
                tail_series = pd.concat([intersection_df, tail_df]).sort_values('n_samples')
            
                # Random addition
                random_df = plot_data[plot_data['label'].str.contains('random_add', na=False)]
                random_series = pd.concat([intersection_df, random_df]).sort_values('n_samples')
            
                color = score_to_color[score_col]
                label_prefix = score_col.replace('mean_wanda_', '').replace('mean_act_', '')

                if len(tail_series) > 0:
                    ax.plot(
                        tail_series['n_samples'],
                        tail_series[score_col],
                        marker='o',
                        color=color,
                        linestyle='-',
                        label=f"{label_prefix} (Tail)"
                    )

                if len(random_series) > 0:
                    ax.plot(
                        random_series['n_samples'],
                        random_series[score_col],
                        marker='x',
                        color=color,
                        linestyle='--',
                        alpha=0.7,
                        label=f"{label_prefix} (Random)"
                    )

            if r == 0:
                ax.set_title(f"Model: {model_name.split('/')[-1]}", fontsize=14)
            
            if c == 0:
                ax.set_ylabel(f"{cal_ds}\nTop 5% Activations Mean", fontsize=12)
            
            ax.set_xlabel("N Samples", fontsize=12)
            
            # Aggiungi legenda solo se siamo nell'ultimo subplot per non affollare
            if r == 0 and c == n_models - 1:
                ax.legend(title="Top 5% Activations", loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.suptitle("Evolution of Mean Top 5% Activations (Last Layers) adding Samples", fontsize=16)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
