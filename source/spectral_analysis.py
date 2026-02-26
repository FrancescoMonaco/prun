"""
Spectral Analysis of Pruned Models.

Compares the spectral properties (singular values, effective rank, spectral norm,
subspace alignment) of weight matrices between the original model and models
pruned with different calibration strategies (random vs unique_tokens).

This complements activation-based and weight-based analyses by examining how pruning
distorts the linear transformations learned by the model.

Performance notes
-----------------
- SVD is computed **on GPU** (10-100x faster than CPU for large matrices).
- Only **key projection layers** are analysed (q/k/v/o_proj, gate/up/down_proj),
  cutting the number of SVDs by ~3-5x while keeping the most informative layers.
- The original model is loaded **once**; its state-dict is backed up on CPU and
  restored between calibration strategies, avoiding repeated disk I/O.
- All datasets share the same original model load.
"""

import argparse
import os
import gc
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
from sentence_transformers import SentenceTransformer

from data import get_dataset, get_text_from_item
from similarity_check import prepare_calibration

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)

# Regex matching the key projection layers inside transformer blocks.
# Covers LLaMA, Gemma, Qwen, Mistral, GPT-2, etc.
_KEY_LAYER_RE = re.compile(
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    r"|query|key|value|dense|fc1|fc2|c_attn|c_proj|w1|w2|w3)"
    r"$"
)


# ---------------------------------------------------------------------------
# Tokenization helper (mirrors run_experiment.py)
# ---------------------------------------------------------------------------
def get_tokenized_data(dataset, tokenizer, dataset_name, max_length=128):
    texts = [get_text_from_item(item, dataset_name) for item in dataset]
    processed = []
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        for j in range(len(batch)):
            processed.append({
                "input_ids": encoded["input_ids"][j],
                "attention_mask": encoded["attention_mask"][j],
            })
    return processed


# ---------------------------------------------------------------------------
# Spectral metrics  (GPU-accelerated)
# ---------------------------------------------------------------------------

def _svd_gpu(W: torch.Tensor, device: torch.device):
    """
    Compute compact SVD of a 2-D weight matrix on *device* (GPU when available).
    Returns U, S, Vh on CPU to free VRAM immediately.
    """
    W = W.to(device=device, dtype=torch.float32)
    if W.dim() != 2:
        W = W.reshape(W.size(0), -1)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U.cpu(), S.cpu(), Vh.cpu()


def effective_rank(S: torch.Tensor) -> float:
    """
    Effective rank (Roy & Vetterli, 2007).
        erank = exp(H(p))   where  p_i = sigma_i / sum(sigma)
    """
    S = S[S > 0]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    H = -(p * torch.log(p)).sum()
    return torch.exp(H).item()


def stable_rank(S: torch.Tensor) -> float:
    """Stable rank = ||W||_F^2 / ||W||_2^2."""
    if len(S) == 0:
        return 0.0
    return (S ** 2).sum().item() / (S[0] ** 2).item()


def subspace_overlap(U_orig: torch.Tensor, U_pruned: torch.Tensor, k: int) -> float:
    """
    Grassmann subspace overlap between the top-k left singular vectors.
    ||U1_k^T @ U2_k||_F^2 / k  in [0, 1].
    """
    U1 = U_orig[:, :k].float()
    U2 = U_pruned[:, :k].float()
    overlap = torch.linalg.norm(U1.T @ U2, ord="fro") ** 2 / k
    return overlap.item()


def spectral_metrics_for_layer(
    W_orig: torch.Tensor, W_pruned: torch.Tensor,
    top_k: int = 10, device: torch.device = torch.device("cpu"),
):
    """Compute spectral metrics comparing original vs pruned weight (GPU-accelerated)."""
    U_o, S_o, _ = _svd_gpu(W_orig, device)
    U_p, S_p, _ = _svd_gpu(W_pruned, device)

    spectral_norm_orig = S_o[0].item()
    spectral_norm_pruned = S_p[0].item()
    nuclear_orig = S_o.sum().item()
    nuclear_pruned = S_p.sum().item()
    frobenius_orig = torch.sqrt((S_o ** 2).sum()).item()
    frobenius_pruned = torch.sqrt((S_p ** 2).sum()).item()

    k = min(top_k, U_o.shape[1], U_p.shape[1])

    return {
        "spectral_norm_orig": spectral_norm_orig,
        "spectral_norm_pruned": spectral_norm_pruned,
        "spectral_norm_ratio": spectral_norm_pruned / max(spectral_norm_orig, 1e-12),
        "nuclear_norm_orig": nuclear_orig,
        "nuclear_norm_pruned": nuclear_pruned,
        "nuclear_norm_ratio": nuclear_pruned / max(nuclear_orig, 1e-12),
        "frobenius_norm_orig": frobenius_orig,
        "frobenius_norm_pruned": frobenius_pruned,
        "frobenius_norm_ratio": frobenius_pruned / max(frobenius_orig, 1e-12),
        "effective_rank_orig": effective_rank(S_o),
        "effective_rank_pruned": effective_rank(S_p),
        "stable_rank_orig": stable_rank(S_o),
        "stable_rank_pruned": stable_rank(S_p),
        f"subspace_overlap_top{k}": subspace_overlap(U_o, U_p, k),
        "singular_values_orig": S_o.numpy(),
        "singular_values_pruned": S_p.numpy(),
    }


# ---------------------------------------------------------------------------
# Weight extraction  (only key projection layers by default)
# ---------------------------------------------------------------------------

def extract_linear_weights(model, key_only: bool = True) -> dict:
    """
    Returns {qualified_name: weight_tensor (CPU)} for Linear / Conv1D layers.
    When *key_only* is True only transformer projection layers are kept,
    skipping the embedding, lm_head and layernorm — much fewer SVDs.
    """
    import transformers.pytorch_utils as ptu
    weights = {}
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, ptu.Conv1D)):
            continue
        if key_only and not _KEY_LAYER_RE.search(name):
            continue
        weights[name] = module.weight.data.detach().cpu()
    return weights


# ---------------------------------------------------------------------------
# Pruning / calibration helpers
# ---------------------------------------------------------------------------

def _prepare_calibration_dataset(
    tokenized_datasets, sentence_transformer, tokenizer,
    nsamples, calib_type, model_name, dataset_name,
):
    """Prepare calibration with a specific strategy and return a HF Dataset."""
    calibration_type_map = {
        "random": "random_sample",
        "unique_tokens": "unique_tokens",
        "most_similar": "prototype",
    }
    calib_data = prepare_calibration(
        model=sentence_transformer,
        dataloader=tokenized_datasets,
        nsamples=nsamples,
        type=calibration_type_map[calib_type],
        distance="flatten",
        model_name=model_name,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
    )
    data_list = []
    for item in calib_data:
        ids = item["input_ids"]
        mask = item.get("attention_mask")
        if isinstance(ids, torch.Tensor):
            if ids.dim() == 2 and ids.shape[0] == 1:
                ids = ids.squeeze(0)
            ids = ids.cpu().tolist()
        if mask is not None and isinstance(mask, torch.Tensor):
            if mask.dim() == 2 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            mask = mask.cpu().tolist()
        d = {"input_ids": ids}
        if mask is not None:
            d["attention_mask"] = mask
        data_list.append(d)
    return Dataset.from_list(data_list)


def prune_model(model, calibration_dataset, sparsity=0.5):
    """Apply Wanda unstructured pruning in-place."""
    recipe = WandaPruningModifier(
        sparsity=sparsity, mask_structure="0:0", targets="__ALL__"
    )
    oneshot(model=model, dataset=calibration_dataset, recipe=recipe)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectral_comparison(df, model_short, output_dir):
    """Generate spectral comparison plots from the per-layer / per-dataset df."""
    os.makedirs(output_dir, exist_ok=True)
    datasets = df["dataset"].unique()

    # --- 1. Effective rank ratio ---
    _line_plot(
        df, datasets, "effective_rank_pruned", "effective_rank_orig",
        ylabel="Effective Rank Ratio (pruned / original)",
        suptitle=f"Effective Rank Preservation – {model_short}",
        fname=os.path.join(output_dir, f"effective_rank_{model_short}.pdf"),
        marker="o",
    )

    # --- 2. Stable rank ratio ---
    _line_plot(
        df, datasets, "stable_rank_pruned", "stable_rank_orig",
        ylabel="Stable Rank Ratio (pruned / original)",
        suptitle=f"Stable Rank Preservation – {model_short}",
        fname=os.path.join(output_dir, f"stable_rank_{model_short}.pdf"),
        marker="s",
    )

    # --- 3. Spectral norm ratio ---
    _line_plot_col(
        df, datasets, "spectral_norm_ratio",
        ylabel="Spectral Norm Ratio (pruned / original)",
        suptitle=f"Spectral Norm Preservation – {model_short}",
        fname=os.path.join(output_dir, f"spectral_norm_{model_short}.pdf"),
        marker="D",
    )

    # --- 4. Subspace overlap ---
    overlap_cols = [c for c in df.columns if c.startswith("subspace_overlap")]
    if overlap_cols:
        _line_plot_col(
            df, datasets, overlap_cols[0],
            ylabel="Subspace Overlap (top-k)",
            suptitle=f"Top-k Subspace Overlap – {model_short}",
            fname=os.path.join(output_dir, f"subspace_overlap_{model_short}.pdf"),
            marker="^", ylim=(0, 1.05),
        )

    # --- 5. Singular value distributions ---
    for ds in datasets:
        _plot_sv_distributions(df[df["dataset"] == ds], model_short, ds, output_dir)

    log.info(f"Plots saved to {output_dir}")


def _line_plot(df, datasets, num_col, den_col, ylabel, suptitle, fname, marker="o"):
    """Ratio plot (num/den) per layer, one subplot per dataset."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for calib in sub["calibration"].unique():
            s = sub[sub["calibration"] == calib]
            ax.plot(
                s["layer_idx"], s[num_col] / s[den_col],
                label=calib, marker=marker, markersize=3, linewidth=1.2,
            )
        ax.set_xlabel("Layer")
        ax.set_title(ds)
        ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _line_plot_col(df, datasets, col, ylabel, suptitle, fname, marker="o", ylim=None):
    """Single-column line plot per layer."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for calib in sub["calibration"].unique():
            s = sub[sub["calibration"] == calib]
            ax.plot(
                s["layer_idx"], s[col],
                label=calib, marker=marker, markersize=3, linewidth=1.2,
            )
        ax.set_xlabel("Layer")
        ax.set_title(ds)
        ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
        ax.legend(fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _plot_sv_distributions(df, model_short, ds, output_dir):
    """SV distribution (log-scale) for first / mid / last layer."""
    calibrations = df["calibration"].unique()
    layer_indices = sorted(df["layer_idx"].unique())
    if len(layer_indices) < 3:
        picks = layer_indices
    else:
        mid = layer_indices[len(layer_indices) // 2]
        picks = [layer_indices[0], mid, layer_indices[-1]]

    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 5), sharey=False)
    if len(picks) == 1:
        axes = [axes]

    for ax, lidx in zip(axes, picks):
        for calib in calibrations:
            row = df[(df["layer_idx"] == lidx) & (df["calibration"] == calib)]
            if row.empty:
                continue
            sv = row.iloc[0]["singular_values_pruned"]
            if sv is not None and len(sv) > 0:
                ax.semilogy(np.arange(len(sv)), sv, label=calib, alpha=0.8, linewidth=1.2)
        row_any = df[df["layer_idx"] == lidx]
        if not row_any.empty:
            sv_orig = row_any.iloc[0]["singular_values_orig"]
            if sv_orig is not None and len(sv_orig) > 0:
                ax.semilogy(
                    np.arange(len(sv_orig)), sv_orig,
                    label="original", color="black", linestyle="--", linewidth=1.2,
                )
        ax.set_xlabel("Singular value index")
        ax.set_title(f"Layer {lidx}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Singular value (log scale)")
    fig.suptitle(f"Singular Value Spectrum – {model_short} / {ds}", y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"sv_distribution_{model_short}_{ds}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spectral Analysis of Pruned Models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B",
                        help="HuggingFace model name or path")
    parser.add_argument("--datasets", nargs="+",
                        default=["boolq", "winogrande", "hellaswag"],
                        help="Datasets used for calibration")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Pruning sparsity level")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k singular vectors for subspace overlap")
    parser.add_argument("--output_dir", type=str, default="results/spectral_analysis",
                        help="Directory for results")
    parser.add_argument("--calibration_types", nargs="+",
                        default=["random", "unique_tokens"],
                        help="Calibration strategies to compare")
    parser.add_argument("--all_layers", action="store_true",
                        help="Analyse ALL linear layers (slow); default: key projections only")

    args = parser.parse_args()
    safe_model_name = args.model.replace("/", "-")
    os.makedirs(args.output_dir, exist_ok=True)
    svd_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Tokenizer ----
    log.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Sentence transformer (for calibration sampling) ----
    sentence_transformer = SentenceTransformer("all-MiniLM-L12-v2", device="cpu")

    # ---- Pre-tokenize ALL datasets once ----
    log.info("Tokenizing datasets...")
    tokenized_by_ds = {}
    for ds_name in args.datasets:
        raw_dataset = get_dataset(ds_name)
        if raw_dataset is None:
            log.warning(f"Could not load dataset {ds_name}, skipping.")
            continue
        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            dataset = (
                raw_dataset.get("train")
                or raw_dataset.get("test")
                or raw_dataset[list(raw_dataset.keys())[0]]
            )
        else:
            dataset = raw_dataset
        tokenized_by_ds[ds_name] = get_tokenized_data(dataset, tokenizer, ds_name)
        log.info(f"  {ds_name}: {len(tokenized_by_ds[ds_name])} samples tokenized")

    # ---- Load model ONCE, backup state-dict on CPU ----
    log.info(f"Loading model {args.model} (single load)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )

    log.info("Extracting original weights (key projection layers)...")
    key_only = not args.all_layers
    orig_weights = extract_linear_weights(model, key_only=key_only)
    log.info(f"  {len(orig_weights)} layers selected for spectral analysis")

    log.info("Backing up original state-dict to CPU...")
    orig_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- Main loop: dataset x calibration_type ----
    all_rows = []
    n_combos = len(tokenized_by_ds) * len(args.calibration_types)
    combo_idx = 0

    for ds_name, tokenized_data in tokenized_by_ds.items():
        for calib_type in args.calibration_types:
            combo_idx += 1
            log.info(
                f"\n[{combo_idx}/{n_combos}] dataset={ds_name}  calibration={calib_type}"
            )

            # Restore original weights from CPU backup (fast, no disk I/O)
            log.info("Restoring original weights from backup...")
            model.load_state_dict(orig_state_dict)

            # Prepare calibration dataset
            log.info(f"Preparing {calib_type} calibration ({args.nsamples} samples)...")
            calib_ds = _prepare_calibration_dataset(
                [tokenized_data], sentence_transformer, tokenizer,
                args.nsamples, calib_type, safe_model_name, ds_name,
            )

            # Prune in-place
            log.info("Pruning model...")
            prune_model(model, calib_ds, sparsity=args.sparsity)

            # Extract pruned weights (same subset)
            log.info("Extracting pruned weights...")
            pruned_weights = extract_linear_weights(model, key_only=key_only)

            # Compute spectral metrics (GPU-accelerated SVD)
            log.info("Computing spectral metrics on GPU...")
            layer_idx = 0
            for name in sorted(orig_weights.keys()):
                if name not in pruned_weights:
                    continue
                metrics = spectral_metrics_for_layer(
                    orig_weights[name], pruned_weights[name],
                    top_k=args.top_k, device=svd_device,
                )
                all_rows.append({
                    "model": args.model,
                    "dataset": ds_name,
                    "calibration": calib_type,
                    "layer_name": name,
                    "layer_idx": layer_idx,
                    **metrics,
                })
                layer_idx += 1

            del pruned_weights
            torch.cuda.empty_cache()

    # ---- Cleanup model ----
    del model, orig_state_dict, orig_weights
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Save CSV (drop numpy arrays) ----
    df = pd.DataFrame(all_rows)
    sv_cols = ["singular_values_orig", "singular_values_pruned"]
    df_csv = df.drop(columns=sv_cols, errors="ignore")

    csv_path = os.path.join(args.output_dir, f"spectral_{safe_model_name}.csv")
    df_csv.to_csv(csv_path, index=False)
    log.info(f"Results saved to {csv_path}")

    # ---- Plots ----
    model_short = args.model.split("/")[-1]
    plot_spectral_comparison(df, model_short, args.output_dir)
    log.info("Spectral analysis complete.")


if __name__ == "__main__":
    main()
