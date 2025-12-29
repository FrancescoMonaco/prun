import pandas as pd
import os

# Define groups from data.py
groups = {
    "General": ["c4", "oscar", "redpajama", "pile"],
    "Arithmetic Reasoning": ["gsm8k", "svamp", "mawps"],
    "NLU Inference": ["anli_r1", "esnli", "rte"],
    "Commonsense QA": ["boolq", "commonsense_qa", "race", "winogrande"],
    "Translation": ["wmt14", "iwslt"],
    "Coding": ["opc", "ds1000", "mbpp"],
}

def get_group(dataset_name):
    if "_" in dataset_name:
        return "Mixed"
    for group_name, datasets in groups.items():
        if dataset_name in datasets:
            return group_name
    return "Other"

def reorganize_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Add group column
    df['group'] = df['calibration_datasets'].apply(get_group)
    
    # Filter out rows where pruned_value is NaN
    df = df.dropna(subset=['pruned_value'])

    models = df['model'].unique()
    nsamples_list = df['nsamples'].unique()

    output_dir = "results/reorganized"
    os.makedirs(output_dir, exist_ok=True)

    for model in models:
        model_name_safe = model.replace("/", "-")
        for nsamples in nsamples_list:
            model_ns_df = df[(df['model'] == model) & (df['nsamples'] == nsamples)]
            if model_ns_df.empty:
                continue
            
            # Average across calibration groups keeping tasks and pruning types separate
            # First, average within each group for each (task, pruning_type)
            group_avg = model_ns_df.groupby(['task', 'pruning_type', 'group'])['pruned_value'].mean().reset_index()
            
            # Then average across groups
            final_avg = group_avg.groupby(['task', 'pruning_type'])['pruned_value'].mean().unstack()
            
            # Get original values (should be same for all pruning types/groups)
            orig_values = model_ns_df.groupby('task')['original_value'].first()
            final_avg.insert(0, 'original', orig_values)

            # Prepare Markdown version with bolding for top 3
            md_avg = final_avg.copy().astype(str)
            for task, row in final_avg.iterrows():
                cols_to_check = [c for c in final_avg.columns if c != 'original']
                vals = row[cols_to_check].dropna()
                top3 = vals.nlargest(3).index.tolist()
                for col in final_avg.columns:
                    val = row[col]
                    if pd.isna(val):
                        md_avg.at[task, col] = "nan"
                    elif col in top3:
                        md_avg.at[task, col] = f"**{val:.4f}**"
                    else:
                        md_avg.at[task, col] = f"{val:.4f}"

            # Prepare LaTeX version with highlighting
            def format_latex_table(df):
                header = " & " + " & ".join(df.columns) + " \\\\\n\\midrule\n"
                rows = []
                for task, row in df.iterrows():
                    cols_to_check = [c for c in df.columns if c != 'original']
                    vals = row[cols_to_check].dropna()
                    top3 = vals.nlargest(3)
                    top3_indices = top3.index.tolist()
                    
                    row_str = f"{task}"
                    for col in df.columns:
                        val = row[col]
                        if pd.isna(val):
                            row_str += " & nan"
                        elif col in top3_indices:
                            rank = top3_indices.index(col)
                            color = ["green!40", "green!25", "green!10"][rank]
                            row_str += f" & \\cellcolor{{{color}}} {val:.4f}"
                        else:
                            row_str += f" & {val:.4f}"
                    row_str += " \\\\"
                    rows.append(row_str)
                
                return "\\begin{tabular}{l" + "r" * len(df.columns) + "}\n\\toprule\n" + header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"

            latex_table_str = format_latex_table(final_avg)

            # Save to Markdown
            filename = f"{model_name_safe}_ns{nsamples}.md"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(f"# Results for {model} (nsamples={nsamples})\n\n")
                f.write("## Average across Calibration Groups\n\n")
                f.write(md_avg.to_markdown())
                f.write("\n\n")
                
                # Also provide LaTeX version
                f.write("## LaTeX Table\n\n")
                f.write("Note: Requires `\\usepackage[table]{xcolor}` in your LaTeX preamble.\n\n")
                f.write("```latex\n")
                f.write(latex_table_str)
                f.write("\n```\n")

            print(f"Saved summary for {model} ns={nsamples} to {filepath}")

if __name__ == "__main__":
    reorganize_results("results/experiment_results.csv")
