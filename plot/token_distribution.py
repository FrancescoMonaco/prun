import os
import sys
import csv
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import argparse
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from collections import Counter
from joblib import Parallel, delayed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "source"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "COLA"))
try:
    from data import get_dataset, get_text_from_item
    from similarity_check import prepare_calibration
    from cola.sample_selection import select_samples
except ImportError as e:
        print(f"Could not import modules, {e}")
        sys.exit(1)
        
mpl.rcParams.update(
        {
            #"text.usetex": True,
            #"text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 15,
            "legend.title_fontsize": 11,
            "figure.titlesize": 12,
            "axes.spines.right": False, # Disable top and left spines by default (Tufte style)
            "axes.spines.top": False,
            
        }
    )

sns.set_theme(palette="muted", style="white", font_scale=1.5)


# ── Label display names ──────────────────────────────────────────────────────
LABEL_MAP = {
    "Full Dataset": "Full Dataset",
    "random": "Random",
    "words_dataset": "Zipf",
    "cola": "COLA",
}

def _display_label(raw: str) -> str:
    return LABEL_MAP.get(raw, raw)


# ── CSV cache helpers ────────────────────────────────────────────────────────

def _cache_path(output_dir: str, model_name: str, dataset_name: str, label: str, nsamples: int) -> str:
    safe_model = model_name.replace("/", "_")
    cache_dir = os.path.join(output_dir, ".cache")
    return os.path.join(cache_dir, f"{safe_model}__{dataset_name}__{label}__{nsamples}.csv")


def _save_counts(counts: Counter, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count"])
        for token, count in counts.items():
            writer.writerow([token, count])


def _load_counts(path: str) -> Counter:
    counts = Counter()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["token"]] = int(row["count"])
    return counts


def _tokenize_one(text: str, tokenizer, max_length: int) -> list:
    """Tokenise a single text; returned as a plain list (picklable)."""
    text = text.lower()
    tokens = tokenizer.tokenize(text, truncation=True, max_length=max_length)
    tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]
    cleaned = []
    for t in tokens:
        c = t.replace(' ', '').replace('Ġ', '').replace('▁', '').strip()
        if c:
            cleaned.append(c)
    return cleaned


def get_token_counts(texts, tokenizer, max_length=128, n_jobs: int = -1):
    """Parallel tokenisation → Counter."""
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_tokenize_one)(text, tokenizer, max_length)
        for text in tqdm(texts, desc="Tokenizing")
    )
    counts = Counter()
    for tokens in results:
        counts.update(tokens)
    return counts


def calculate_kl(full_counts, subset_counts):
    # Get all tokens present in either distribution
    all_tokens = sorted(list(set(full_counts.keys()) | set(subset_counts.keys())))
    
    # Convert counts to probability distributions
    p = np.array([full_counts.get(t, 0) for t in all_tokens], dtype=np.float64)
    q = np.array([subset_counts.get(t, 0) for t in all_tokens], dtype=np.float64)
    
    # Normalize
    p_sum = p.sum()
    q_sum = q.sum()
    
    if p_sum == 0 or q_sum == 0:
        return float('inf')
        
    p /= p_sum
    q /= q_sum
    
    # Add epsilon to q to avoid infinite KL if a token in P is missing in Q
    epsilon = 1e-10
    q = q + epsilon
    q /= q.sum()
    
    return entropy(p, q)

def plot_token_distribution_multi(
    datasets_distributions: dict,   # {dataset_name: (all_counts_list, raw_labels_list)}
    output_path: str,
    tokenizer,
):
    """
    One rectangular figure with one subplot per dataset (Zipf's Law log-log).
    A single shared legend sits at the top of the figure.
    X-axis follows Tufte convention: spine spans only the actual rank range.
    """
    dataset_names = list(datasets_distributions.keys())
    n = len(dataset_names)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    # Collect a shared line-handle per display-label across all subplots
    shared_handles: dict = {}

    for ax, ds_name in zip(axes, dataset_names):
        all_counts, raw_labels = datasets_distributions[ds_name]

        for counts, raw_label in zip(all_counts, raw_labels):
            label = _display_label(raw_label)
            freqs = sorted(counts.values(), reverse=True)
            if not freqs:
                continue
            total = sum(freqs)
            freqs_norm = np.array(freqs, dtype=np.float64) / total
            ranks = np.arange(1, len(freqs_norm) + 1)
            (line,) = ax.plot(ranks, freqs_norm, linewidth=1.8, alpha=0.85, label=label)
            if label not in shared_handles:
                shared_handles[label] = line

        # Reference Zipf curve
        max_rank = max(len(c) for c in all_counts)
        zipf_ranks = np.arange(1, max_rank + 1)
        zipf_vals = 1.0 / zipf_ranks
        zipf_vals /= zipf_vals.sum()
        (ref_line,) = ax.plot(
            zipf_ranks, zipf_vals, "k--", linewidth=1.2, alpha=0.55, label="Theoretical Zipf"
        )
        if "Theoretical Zipf" not in shared_handles:
            shared_handles["Theoretical Zipf"] = ref_line

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank (log)")
        if ax is axes[0]:
            ax.set_ylabel("Frequency (log)")
        ax.set_title(ds_name)
        ax.grid(False)

        # Tufte convention: x-spine spans only the actual rank range
        sns.despine(ax=ax, trim=False)
        ax.spines["bottom"].set_bounds(1, max_rank)

    # ── Shared legend at the top ──────────────────────────────────────────────
    handles = list(shared_handles.values())
    labels_list = list(shared_handles.keys())
    fig.legend(
        handles,
        labels_list,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Token Distribution to Show the Distribution")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["boolq", "gsm8k", "winogrande", "hellaswag", "anli_r1"],
        required=False,
        help="Names of datasets to plot (one subplot each, max 3 recommended)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-7b",
        help="Model name for tokenizer and selection",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of samples for calibration data",
    )
    parser.add_argument(
        "--pruning_types",
        nargs="+",
        default=["random", "words_dataset", "cola"],
        help="Selection methods to compare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/token_distribution",
        help="Output directory for plots",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model will be loaded lazily only when a cache miss requires it
    _model_holder = [None]  # mutable container to allow assignment inside nested scope

    def _ensure_model():
        if _model_holder[0] is None:
            print(f"Loading model: {args.model}")
            _model_holder[0] = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
        return _model_holder[0]

    calibration_type_map = {
        "random": "random_sample",
        "most_similar": "prototype",
        "most_dissimilar": "most_different",
        "decoupled": "decoupled",
        "least_perplexity": "least_perplexity",
        "herding": "herding",
        "distribution_matching": "distribution_matching",
        "zipf": "zipf",
        "unique_tokens": "unique_tokens",
        "words_dataset": "words_dataset",
    }

    # ── Process each dataset ───────────────────────────────────────────────────
    datasets_distributions: dict = {}   # {dataset_name: (counts_list, labels_list)}

    for dataset_name in args.datasets:
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")

        raw_dataset = get_dataset(dataset_name)
        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            split = None
            for key in ("train", "test", "validation") + tuple(raw_dataset.keys()):
                candidate = raw_dataset.get(key) if hasattr(raw_dataset, "get") else raw_dataset[key] if key in raw_dataset else None
                if candidate is not None:
                    split = candidate
                    break
            if split is None:
                raise ValueError(f"All splits for dataset '{dataset_name}' are None: {list(raw_dataset.keys())}")
        else:
            split = raw_dataset
        if split is None:
            raise ValueError(f"Could not load any split for dataset '{dataset_name}'")

        all_texts = [get_text_from_item(item, dataset_name) for item in split]
        all_texts_limited = all_texts[:5000] if len(all_texts) > 5000 else all_texts

        full_cache = _cache_path(args.output_dir, args.model, dataset_name, "Full_Dataset", len(all_texts_limited))
        if os.path.exists(full_cache):
            print(f"Loading full dataset counts from cache: {full_cache}")
            full_counts = _load_counts(full_cache)
        else:
            print("Calculating full dataset token distribution...")
            full_counts = get_token_counts(all_texts_limited, tokenizer)
            _save_counts(full_counts, full_cache)

        all_dist = [full_counts]
        ds_labels = ["Full Dataset"]

        # Tokenize candidates once per dataset
        num_candidates = min(len(all_texts), 4080)
        all_tokenized_data = []
        for text in tqdm(all_texts[:num_candidates], desc="Tokenizing candidates"):
            encoded = tokenizer(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
            all_tokenized_data.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            })

        for p_type in args.pruning_types:
            print(f"\nProcessing: {p_type}")

            p_cache = _cache_path(args.output_dir, args.model, dataset_name, p_type, args.nsamples)
            if os.path.exists(p_cache):
                print(f"  Loading from cache: {p_cache}")
                counts = _load_counts(p_cache)
                all_dist.append(counts)
                ds_labels.append(p_type)
                continue

            if p_type == "cola":
                processed_samples = [{"text": t} for t in all_texts[:num_candidates]]
                selected_samples = select_samples(
                    processed_samples,
                    _ensure_model(),
                    tokenizer,
                    num_clusters=args.nsamples,
                    device=device,
                    batch_size=4,
                )
                selected_texts = [s["text"] for s in selected_samples]

            elif p_type in calibration_type_map:
                method = calibration_type_map[p_type]
                calib_data = prepare_calibration(
                    model=_ensure_model(),
                    dataloader=[all_tokenized_data],
                    nsamples=args.nsamples,
                    type=method,
                    distance="flatten",
                    model_name=args.model.replace("/", "_"),
                    dataset_name=[dataset_name],
                    tokenizer=tokenizer,
                )
                selected_texts = [
                    tokenizer.decode(item["input_ids"].tolist(), skip_special_tokens=True)
                    for item in calib_data
                ]
            else:
                print(f"Unknown pruning type: {p_type}")
                continue

            counts = get_token_counts(selected_texts, tokenizer)
            _save_counts(counts, p_cache)
            all_dist.append(counts)
            ds_labels.append(p_type)

        datasets_distributions[dataset_name] = (all_dist, ds_labels)

        # ── KL divergence report ───────────────────────────────────────────────
        print(f"\n--- KL Divergence for {dataset_name} ---")
        for i in range(1, len(ds_labels)):
            kl_val = calculate_kl(all_dist[0], all_dist[i])
            print(f"  {ds_labels[i]:25} | KL: {kl_val:.6f}")

    # ── Single multi-panel figure ──────────────────────────────────────────────
    suffix = "_".join(args.datasets)
    output_path = os.path.join(args.output_dir, f"{suffix}_token_dist.pdf")
    plot_token_distribution_multi(datasets_distributions, output_path, tokenizer)
