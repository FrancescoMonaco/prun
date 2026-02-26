#!/usr/bin/env python3
"""
Generate comparison tables (Markdown + LaTeX) for COLA vs. our pruning methods.

Usage:
    python3 generate_table.py --compression_type pruning --pruning_type random \
                              --nsamples 128 --model google/gemma-7b meta-llama/Llama-2-7b

    Multiple models are placed in the same table as separate macro-row groups.
    Each model group ends with a Mean row computed over that model's tasks (row-wise mean).

    For cola_results_new.csv the pruning_type column is always "cola", so the
    --pruning_type flag selects only from experiment_results_new.csv.
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
COLA_CSV = SCRIPT_DIR / "results" / "cola_results_new.csv"
EXP_CSV  = SCRIPT_DIR / "results" / "experiment_results_new.csv"

METRIC_PREFERRED = "acc_norm,none"  # prefer normalised accuracy
METRIC_FALLBACK  = "acc,none"       # fall back for tasks w/o acc_norm

# Calibration datasets excluded from our method's row-mean / row-gmean
EXCLUDED_FROM_EXP_MEAN = {"winogrande_arc_challenge_boolq_hellaswag_openbookqa_rte"}

# Sentinel key pairs for (COLA, our method) mean columns – used for boldening
MEAN_SENTINEL_PAIRS = [
    ("__cola_mean__",  "__exp_mean__"),
    ("__cola_gmean__", "__exp_gmean__"),
]

# All aggregate/sentinel keys – excluded from top-3 green coloring
AGGREGATE_SENTINELS = frozenset({
    "__original__",
    "__cola_mean__", "__cola_gmean__",
    "__exp_mean__",  "__exp_gmean__",
})


def _gmean_normalized(scores: list, original: float) -> float:
    """
    Normalized geometric mean: gmean(s_i / original) for each score s_i.
    Values <= 0 or NaN are skipped. Returns NaN if no valid scores or original <= 0.
    """
    if not original or math.isnan(original) or original <= 0:
        return float("nan")
    ratios = [
        float(s) / original
        for s in scores
        if not math.isnan(float(s)) and float(s) > 0
    ]
    if not ratios:
        return float("nan")
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))

# ── Helpers ──────────────────────────────────────────────────────────────────

def _pick_metric(df: pd.DataFrame, task_col: str = "task") -> pd.DataFrame:
    """Keep acc_norm,none rows where available; fall back to acc,none."""
    preferred = df[df["metric"] == METRIC_PREFERRED]
    fallback  = df[df["metric"] == METRIC_FALLBACK]
    tasks_with_preferred = set(preferred[task_col].unique())
    fallback_only = fallback[~fallback[task_col].isin(tasks_with_preferred)]
    return pd.concat([preferred, fallback_only], ignore_index=True)


def load_cola(compression_type: str, nsamples: int, model: str) -> pd.DataFrame:
    """Return a pivot DataFrame (task × calibration_dataset) for COLA values."""
    df = pd.read_csv(COLA_CSV, header=None)
    df.columns = [
        "task", "metric", "value", "model",
        "compression_type", "pruning_type",
        "nsamples", "sparsity", "calibration_datasets",
    ]
    mask = (
        (df["compression_type"] == compression_type)
        & (df["nsamples"] == nsamples)
        & (df["model"] == model)
    )
    sub = _pick_metric(df.loc[mask])
    sub = sub[["task", "calibration_datasets", "value"]].copy()
    if sub.empty:
        return sub
    pivot = sub.pivot(index="task", columns="calibration_datasets", values="value")
    return pivot


def load_experiment(
    compression_type: str, pruning_type: str, nsamples: int, model: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (pivot of pruned_value, series of original_value) indexed by task."""
    df = pd.read_csv(EXP_CSV)
    mask = (
        (df["compression_type"] == compression_type)
        & (df["pruning_type"] == pruning_type)
        & (df["nsamples"] == nsamples)
        & (df["model"] == model)
    )
    sub = _pick_metric(df.loc[mask])
    sub = sub[["task", "calibration_datasets", "original_value", "pruned_value"]].copy()
    if sub.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Original value should be the same across calibration_datasets per task
    originals = sub.groupby("task")["original_value"].first()

    pivot = sub.pivot(index="task", columns="calibration_datasets", values="pruned_value")
    return pivot, originals


def build_table(
    compression_type: str,
    pruning_type: str,
    nsamples: int,
    model: str,
) -> tuple[list[str], pd.DataFrame, list[str], list[str]]:
    """
    Build the combined table.

    Returns
    -------
    tasks : list[str]          – row labels
    table : pd.DataFrame       – (n_tasks × n_cols) numeric values
    col_headers : list[str]    – first-level column names
    group_labels : list[str]   – macro-group each column belongs to
                                 ("Original", "COLA", pruning_type)
    """
    cola_pivot = load_cola(compression_type, nsamples, model)
    exp_pivot, originals = load_experiment(compression_type, pruning_type, nsamples, model)

    if exp_pivot.empty:
        print(f"[ERROR] No experiment data for {model}, {compression_type}, "
              f"{pruning_type}, nsamples={nsamples}", file=sys.stderr)
        sys.exit(1)

    tasks = sorted(exp_pivot.index)
    cola_calib_ds = sorted(cola_pivot.columns) if not cola_pivot.empty else []
    exp_calib_ds  = sorted(exp_pivot.columns)

    # Assemble columns – use unique internal keys to avoid duplicate names
    col_keys: list[str] = []      # unique internal column identifiers
    col_display: list[str] = []   # display names (calibration dataset or "Original")
    group_labels: list[str] = []
    rows: dict[str, list[float]] = {t: [] for t in tasks}

    # 1) Original column
    col_keys.append("__original__")
    col_display.append("Original")
    group_labels.append("Original")
    for t in tasks:
        rows[t].append(originals.get(t, float("nan")))

    # 2) COLA columns + row-mean
    for cd in cola_calib_ds:
        col_keys.append(f"cola__{cd}")
        col_display.append(cd)
        group_labels.append("COLA")
        for t in tasks:
            val = cola_pivot.at[t, cd] if (t in cola_pivot.index and cd in cola_pivot.columns) else float("nan")
            rows[t].append(val)
    # COLA row-mean and row-gmean columns
    col_keys.append("__cola_mean__")
    col_display.append("Mean")
    group_labels.append("COLA")
    for t in tasks:
        vals = [
            cola_pivot.at[t, cd]
            for cd in cola_calib_ds
            if t in cola_pivot.index and cd in cola_pivot.columns
        ]
        rows[t].append(float(pd.Series(vals, dtype=float).mean()) if vals else float("nan"))
    col_keys.append("__cola_gmean__")
    col_display.append("GMean")
    group_labels.append("COLA")
    for t in tasks:
        scores = [
            cola_pivot.at[t, cd]
            for cd in cola_calib_ds
            if t in cola_pivot.index and cd in cola_pivot.columns
        ]
        orig = float(originals.get(t, float("nan")))
        rows[t].append(_gmean_normalized(scores, orig))

    # 3) Our method columns + row-mean + row-gmean (excluding EXCLUDED_FROM_EXP_MEAN)
    for cd in exp_calib_ds:
        col_keys.append(f"exp__{cd}")
        col_display.append(cd)
        group_labels.append(pruning_type)
        for t in tasks:
            val = exp_pivot.at[t, cd] if (t in exp_pivot.index and cd in exp_pivot.columns) else float("nan")
            rows[t].append(val)
    exp_mean_ds = [cd for cd in exp_calib_ds if cd not in EXCLUDED_FROM_EXP_MEAN]
    col_keys.append("__exp_mean__")
    col_display.append("Mean")
    group_labels.append(pruning_type)
    for t in tasks:
        vals = [
            exp_pivot.at[t, cd]
            for cd in exp_mean_ds
            if t in exp_pivot.index and cd in exp_pivot.columns
        ]
        rows[t].append(float(pd.Series(vals, dtype=float).mean()) if vals else float("nan"))
    col_keys.append("__exp_gmean__")
    col_display.append("GMean")
    group_labels.append(pruning_type)
    for t in tasks:
        scores = [
            exp_pivot.at[t, cd]
            for cd in exp_mean_ds
            if t in exp_pivot.index and cd in exp_pivot.columns
        ]
        orig = float(originals.get(t, float("nan")))
        rows[t].append(_gmean_normalized(scores, orig))

    table = pd.DataFrame(rows, index=col_keys).T
    table.index.name = "task"
    return tasks, table, col_display, group_labels


def build_multi_table(
    compression_type: str,
    pruning_type: str,
    nsamples: int,
    models: list[str],
) -> tuple[
    dict[str, list[str]],
    dict[str, pd.DataFrame],
    dict[str, list[str]],
    dict[str, list[str]],
    list[str],
    list[str],
]:
    """
    Build per-model tables, each with its own column schema.

    Returns
    -------
    model_tasks        : {model: [task, ...]}
    model_tables       : {model: pd.DataFrame}
    model_col_headers  : {model: [display_header, ...]}
    model_group_labels : {model: [group_label, ...]}
    first_col_headers  : col_headers of the first model (for table header rendering)
    first_group_labels : group_labels of the first model
    """
    model_tasks: dict[str, list[str]] = {}
    model_tables: dict[str, pd.DataFrame] = {}
    model_col_headers: dict[str, list[str]] = {}
    model_group_labels: dict[str, list[str]] = {}
    first_col_headers: list[str] = []
    first_group_labels: list[str] = []

    for model in models:
        tasks, table, ch, gl = build_table(compression_type, pruning_type, nsamples, model)
        model_tasks[model] = tasks
        model_tables[model] = table
        model_col_headers[model] = ch
        model_group_labels[model] = gl
        if not first_col_headers:
            first_col_headers = ch
            first_group_labels = gl

    return model_tasks, model_tables, model_col_headers, model_group_labels, first_col_headers, first_group_labels


# ── Top-3 highlighting logic ────────────────────────────────────────────────

def top3_indices(row: pd.Series) -> list[int]:
    """Return positional indices of the top 3 values (descending), skipping aggregate columns."""
    valid = [
        (i, v) for i, (k, v) in enumerate(zip(row.index, row))
        if pd.notna(v) and k not in AGGREGATE_SENTINELS
    ]
    if not valid:
        return []
    valid.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in valid[:3]]


# Gold / Silver / Bronze in green shades for markdown
GREEN_BG_MD = [
    '<span style="background-color:#1b5e20;color:white;padding:2px 4px">{v}</span>',   # dark green
    '<span style="background-color:#4caf50;color:white;padding:2px 4px">{v}</span>',   # green
    '<span style="background-color:#a5d6a7;padding:2px 4px">{v}</span>',               # light green
]

LATEX_GREEN = [
    r"\cellcolor{green!60}",   # 1st
    r"\cellcolor{green!35}",   # 2nd
    r"\cellcolor{green!15}",   # 3rd
]


def fmt(v: float) -> str:
    if pd.isna(v):
        return "--"
    return f"{v:.4f}"


def _apply_top3_md(cells: list[str], row: pd.Series) -> list[str]:
    """Return cells with green-shade HTML applied to the top-3 values."""
    top3 = top3_indices(row)
    result = []
    for ci, s in enumerate(cells):
        if ci in top3:
            s = GREEN_BG_MD[top3.index(ci)].format(v=s)
        result.append(s)
    return result


def _bold_winners(cells: list[str], row: pd.Series, latex: bool = False) -> list[str]:
    """For each (cola_key, exp_key) sentinel pair, bold the cell with the higher value."""
    key_to_pos = {k: i for i, k in enumerate(row.index)}
    result = list(cells)
    for cola_key, exp_key in MEAN_SENTINEL_PAIRS:
        ci = key_to_pos.get(cola_key)
        ei = key_to_pos.get(exp_key)
        if ci is None or ei is None:
            continue
        cv = float(row.iloc[ci]) if pd.notna(row.iloc[ci]) else None
        ev = float(row.iloc[ei]) if pd.notna(row.iloc[ei]) else None
        if cv is None and ev is None:
            continue
        best = ci if (cv is not None and (ev is None or cv >= ev)) else ei
        if latex:
            result[best] = r"\textbf{" + result[best] + "}"
        else:
            result[best] = f"**{result[best]}**"
    return result


def _bold_higher_mean_md(cells: list[str], row: pd.Series) -> list[str]:
    return _bold_winners(cells, row, latex=False)


def _bold_higher_mean_latex(cells: list[str], row: pd.Series) -> list[str]:
    return _bold_winners(cells, row, latex=True)
    return result


# ── Markdown ─────────────────────────────────────────────────────────────────

def generate_markdown(
    model_tasks: dict[str, list[str]],
    model_tables: dict[str, pd.DataFrame],
    model_col_headers: dict[str, list[str]],
    model_group_labels: dict[str, list[str]],
    col_headers: list[str],
    group_labels: list[str],
    models: list[str],
    compression_type: str,
    pruning_type: str,
    nsamples: int,
) -> str:
    lines: list[str] = []
    lines.append(
        f"## {', '.join(models)} | {compression_type} | nsamples={nsamples} "
        f"| metric={METRIC_PREFERRED} (fallback: {METRIC_FALLBACK})\n"
    )

    n_data = len(col_headers)

    # Header row 1: macro-groups
    groups_row: list[str] = []
    prev = None
    for g in group_labels:
        if g != prev:
            groups_row.append(g)
            prev = g
        else:
            groups_row.append("")
    lines.append("| Model | Task | " + " | ".join(groups_row) + " |")

    # Header row 2: calibration dataset sub-labels
    sub_headers: list[str] = []
    for h, g in zip(col_headers, group_labels):
        if g == "Original":
            sub_headers.append("")
        elif h in ("Mean", "GMean"):
            sub_headers.append(f"**{h}**")
        else:
            sub_headers.append(h)
    lines.append("| | | " + " | ".join(sub_headers) + " |")

    # Separator
    lines.append("|" + "---|" * (n_data + 2))

    # Data rows: one group per model
    for mi, model in enumerate(models):
        tasks = model_tasks[model]
        table = model_tables[model]
        n_data_model = len(model_col_headers[model])

        for ri, t in enumerate(tasks):
            row   = table.loc[t]
            cells = _apply_top3_md([fmt(v) for v in row], row)
            cells = _bold_higher_mean_md(cells, row)
            model_cell = model if ri == 0 else ""
            lines.append(f"| {model_cell} | {t} | " + " | ".join(cells) + " |")

        # Per-model Mean row (mean over this model's tasks)
        means      = table.loc[tasks].mean()
        mean_cells = _apply_top3_md([fmt(v) for v in means], means)
        mean_cells = _bold_higher_mean_md(mean_cells, means)
        lines.append("| | **Mean** | " + " | ".join(mean_cells) + " |")

        # Blank separator between models
        if mi < len(models) - 1:
            lines.append("|" + " |" * (n_data_model + 2))

    return "\n".join(lines) + "\n"


# ── LaTeX ────────────────────────────────────────────────────────────────────

def _escape_latex(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("#", r"\#")
         .replace("$", r"\$")
         .replace("~", r"\textasciitilde{}")
         .replace("^", r"\textasciicircum{}")
    )


def _latex_cells(row: pd.Series) -> list[str]:
    """Format a data row applying top-3 green shading (skipping aggregate columns)."""
    top3 = top3_indices(row)
    cells = []
    for ci, (k, v) in enumerate(zip(row.index, row)):
        s = fmt(v)
        if ci in top3 and k not in AGGREGATE_SENTINELS:
            s = LATEX_GREEN[top3.index(ci)] + " " + s
        cells.append(s)
    return cells


def generate_latex(
    model_tasks: dict[str, list[str]],
    model_tables: dict[str, pd.DataFrame],
    model_col_headers: dict[str, list[str]],
    model_group_labels: dict[str, list[str]],
    col_headers: list[str],
    group_labels: list[str],
    models: list[str],
    compression_type: str,
    pruning_type: str,
    nsamples: int,
) -> str:
    n_data = len(col_headers)
    total_cols = n_data + 2  # Model | Task | data...

    # Column spec: Model col | Task col | data cols (vertical rule before Mean/GMean cols)
    col_parts: list[str] = []
    for h in col_headers:
        col_parts.append("|c" if h in ("Mean", "GMean") else "c")
    col_spec = "l|l|" + "".join(col_parts)

    # Macro-group spans (data columns only)
    spans: list[tuple[str, int]] = []
    cur_label, cur_count = group_labels[0], 1
    for g in group_labels[1:]:
        if g == cur_label:
            cur_count += 1
        else:
            spans.append((cur_label, cur_count))
            cur_label, cur_count = g, 1
    spans.append((cur_label, cur_count))

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: macro-groups
    macro_parts: list[str] = []
    for label, span in spans:
        escaped = _escape_latex(label)
        macro_parts.append(
            r"\multicolumn{" + str(span) + r"}{c}{" + escaped + "}"
            if span > 1
            else r"\multicolumn{1}{c}{" + escaped + "}"
        )
    lines.append(r"Model & Task & " + " & ".join(macro_parts) + r" \\")

    # cmidrules under each multi-column group
    cmidrules: list[str] = []
    col_idx = 3  # data starts at col 3 (after Model, Task)
    for label, span in spans:
        if span > 1:
            cmidrules.append(
                r"\cmidrule(lr){" + str(col_idx) + "-" + str(col_idx + span - 1) + "}"
            )
        col_idx += span
    if cmidrules:
        lines.append(" ".join(cmidrules))

    # Header row 2: calibration dataset sub-labels
    sub_parts: list[str] = []
    for h, g in zip(col_headers, group_labels):
        if g == "Original":
            sub_parts.append("")
        elif h in ("Mean", "GMean"):
            sub_parts.append(r"\textbf{" + h + "}")
        else:
            sub_parts.append(_escape_latex(h))
    lines.append(" & & " + " & ".join(sub_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows: one macro-group per model
    for mi, model in enumerate(models):
        tasks  = model_tasks[model]
        table  = model_tables[model]
        mch    = model_col_headers[model]
        mgl    = model_group_labels[model]
        n_data_model = len(mch)
        total_cols_model = n_data_model + 2
        # multirow spans tasks + 1 Mean row
        multirow_span = len(tasks) + 1
        # Use the last part of the model path for a compact display name
        model_display = r"\textbf{" + _escape_latex(model.split("/")[-1]) + "}"
        model_cell = r"\multirow{" + str(multirow_span) + r"}{*}{" + model_display + "}"

        for ri, t in enumerate(tasks):
            row   = table.loc[t]
            cells = _latex_cells(row)
            cells = _bold_higher_mean_latex(cells, row)
            mc    = model_cell if ri == 0 else ""
            lines.append(mc + " & " + _escape_latex(t) + " & " + " & ".join(cells) + r" \\")

        # Per-model Mean row
        lines.append(r"\cmidrule{2-" + str(total_cols_model) + "}")
        means      = table.loc[tasks].mean()
        mean_cells = _latex_cells(means)
        mean_cells = _bold_higher_mean_latex(mean_cells, means)
        lines.append(r" & \textbf{Mean} & " + " & ".join(mean_cells) + r" \\")

        if mi < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("}")  # close resizebox

    model_str = ", ".join(_escape_latex(m) for m in models)
    caption = (
        f"{model_str}, {_escape_latex(compression_type)}, "
        f"nsamples={nsamples}, metric={_escape_latex(METRIC_PREFERRED)}"
    )
    lines.append(r"\caption{" + caption + "}")
    lines.append(
        r"\label{tab:" + compression_type + "_" + pruning_type + "_n" + str(nsamples) + "}"
    )
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown + LaTeX comparison tables for COLA vs. experiment results."
    )
    parser.add_argument("--compression_type", required=True,
                        choices=["pruning", "quantization"],
                        help="Compression type to filter on.")
    parser.add_argument("--pruning_type", required=True,
                        help="Pruning type from experiment_results (e.g. random, unique_tokens).")
    parser.add_argument("--nsamples", required=True, type=int,
                        help="Number of calibration samples.")
    parser.add_argument("--model", required=True, nargs="+",
                        help="One or more model names (e.g. google/gemma-7b meta-llama/Llama-2-7b). "
                             "Multiple models are shown as separate macro-row groups in one table.")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to write output files. Defaults to results/tables/.")
    args = parser.parse_args()

    models = args.model

    model_tasks, model_tables, model_col_headers, model_group_labels, col_headers, group_labels = build_multi_table(
        args.compression_type, args.pruning_type, args.nsamples, models,
    )

    md = generate_markdown(
        model_tasks, model_tables, model_col_headers, model_group_labels,
        col_headers, group_labels,
        models, args.compression_type, args.pruning_type, args.nsamples,
    )
    tex = generate_latex(
        model_tasks, model_tables, model_col_headers, model_group_labels,
        col_headers, group_labels,
        models, args.compression_type, args.pruning_type, args.nsamples,
    )

    # Determine output dir
    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_models = "_".join(m.replace("/", "_") for m in models)
    stem = f"{safe_models}_{args.compression_type}_{args.pruning_type}_n{args.nsamples}"

    md_path  = out_dir / f"{stem}.md"
    tex_path = out_dir / f"{stem}.tex"

    md_path.write_text(md, encoding="utf-8")
    tex_path.write_text(tex, encoding="utf-8")

    print(f"Markdown → {md_path}")
    print(f"LaTeX    → {tex_path}")

    # Also print to stdout for quick preview
    print("\n" + "=" * 80)
    print("MARKDOWN PREVIEW")
    print("=" * 80)
    print(md)
    print("=" * 80)
    print("LATEX PREVIEW")
    print("=" * 80)
    print(tex)


if __name__ == "__main__":
    main()
