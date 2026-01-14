import pandas as pd
import os


def summarize_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Group by key parameters and pivot pruning types
    # We keep 'original_value' as it should be the same for all pruning types of the same model/task

    # First, let's handle the case where there might be multiple runs for the same configuration
    # by taking the mean (though in this case it should be unique)
    summary = (
        df.groupby(
            [
                "model",
                "task",
                "metric",
                "sparsity",
                "nsamples",
                "calibration_datasets",
                "pruning_type",
            ]
        )["pruned_value"]
        .mean()
        .unstack()
    )

    # Get original values
    orig_values = df.groupby(
        ["model", "task", "metric", "sparsity", "nsamples", "calibration_datasets"]
    )["original_value"].first()

    # Combine
    final_table = summary.copy()
    final_table["original"] = orig_values

    # Reorder columns to have 'original' first
    cols = ["original"] + [c for c in final_table.columns if c != "original"]
    final_table = final_table[cols]

    # Save to file
    output_md = "results/summary.md"
    with open(output_md, "w") as f:
        f.write("# Pruning Experiment Summary\n\n")
        f.write("## Results Summary by Task\n\n")
        f.write(final_table.to_markdown())

        # Calculate average across tasks for each model/sparsity/nsamples
        f.write("\n\n## Average Performance across Tasks\n\n")
        avg_summary = (
            df.groupby(
                [
                    "model",
                    "sparsity",
                    "nsamples",
                    "calibration_datasets",
                    "pruning_type",
                ]
            )["pruned_value"]
            .mean()
            .unstack()
        )
        avg_orig = df.groupby(
            ["model", "sparsity", "nsamples", "calibration_datasets"]
        )["original_value"].mean()

        avg_table = avg_summary.copy()
        avg_table["original"] = avg_orig
        cols = ["original"] + [c for c in avg_table.columns if c != "original"]
        avg_table = avg_table[cols]
        f.write(avg_table.to_markdown())

        # Average across calibration datasets, keeping tasks separate
        f.write("\n\n## Average across Calibration Datasets (per Task)\n\n")
        calib_avg_summary = (
            df.groupby(
                ["model", "task", "metric", "sparsity", "nsamples", "pruning_type"]
            )["pruned_value"]
            .mean()
            .unstack()
        )
        calib_avg_orig = df.groupby(
            ["model", "task", "metric", "sparsity", "nsamples"]
        )["original_value"].mean()

        calib_avg_table = calib_avg_summary.copy()
        calib_avg_table["original"] = calib_avg_orig
        cols = ["original"] + [c for c in calib_avg_table.columns if c != "original"]
        calib_avg_table = calib_avg_table[cols]
        f.write(calib_avg_table.to_markdown())

        # Global average across tasks AND calibration datasets
        f.write("\n\n## Global Average (Across Tasks and Calibration Datasets)\n\n")
        global_summary = (
            df.groupby(["model", "sparsity", "nsamples", "pruning_type"])[
                "pruned_value"
            ]
            .mean()
            .unstack()
        )
        global_orig = df.groupby(["model", "sparsity", "nsamples"])[
            "original_value"
        ].mean()

        global_table = global_summary.copy()
        global_table["original"] = global_orig
        cols = ["original"] + [c for c in global_table.columns if c != "original"]
        global_table = global_table[cols]
        f.write(global_table.to_markdown())

    print(f"Summary saved to {output_md}")
    print("\n### Global Average (Across Tasks and Calibration Datasets)")
    print(global_table.to_markdown())


if __name__ == "__main__":
    summarize_results("results/experiment_results.csv")
