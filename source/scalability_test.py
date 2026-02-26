import argparse
import os
import sys
import time
import torch
import pandas as pd
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from data import get_dataset, get_text_from_item
from similarity_check import prepare_calibration

# Add COLA to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "COLA"))
from cola.sample_selection import select_samples
from cola.dataset_processing import tokenize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ── Default Meta-Llama models for COLA (need the actual model for activations) ──
DEFAULT_COLA_MODELS = [
    #"meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",   # text-only (no 11B text exists in Llama 3.x)
    "meta-llama/Llama-3.1-70B",  # Needs multi-GPU
]

FRACTIONS = [0.1, 0.25, 0.5, 1.0]


# ── helpers ──────────────────────────────────────────────────────────────────────

def get_tokenized_data(dataset, tokenizer, dataset_name, max_length=128):
    """Tokenize a HF dataset into a list of dicts with input_ids / attention_mask."""
    texts = [get_text_from_item(item, dataset_name) for item in dataset]
    processed = []
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        for j in range(len(batch_texts)):
            processed.append(
                {
                    "input_ids": encoded["input_ids"][j],
                    "attention_mask": encoded["attention_mask"][j],
                }
            )
    return processed


def load_raw_dataset(dataset_name):
    """Load a dataset and return the first available split."""
    raw = get_dataset(dataset_name)
    if raw is None:
        raise ValueError(f"Could not load dataset {dataset_name}")
    if isinstance(raw, dict) or hasattr(raw, "keys"):
        return (
            raw.get("train")
            or raw.get("test")
            or raw[list(raw.keys())[0]]
        )
    return raw


def prepare_cola_processed_samples(dataset, dataset_name, tokenizer, max_length=2048):
    """
    Build the list[dict] that COLA's select_samples expects:
    each dict must have at least a "text" key (+ optional input_ids / attention_mask).
    """
    samples = []
    for item in dataset:
        text = get_text_from_item(item, dataset_name)
        sample = tokenize_text(text, tokenizer, max_length=max_length)
        samples.append(sample)
    return samples


def load_model_for_cola(model_name, device):
    """
    Load a causal-LM and its tokenizer.
    Uses device_map="auto" so large models are sharded across all visible GPUs.
    """
    log.info(f"Loading model {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",          # auto-shard across all GPUs
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Measure calibration-sample selection time: our method vs COLA."
    )
    parser.add_argument(
        "--cola_models",
        nargs="+",
        default=DEFAULT_COLA_MODELS,
        help="HF model names for COLA (default: Llama 3B, 11B, 70B).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names to evaluate on (e.g. winogrande gsm8k).",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples to select.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max token length for tokenization.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/scalability_results.csv",
        help="Path for the output CSV.",
    )
    args = parser.parse_args()

    results = []  # list of dicts → DataFrame at the end
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ##  OUR METHOD (model-free: uses unique_tokens sampling)
    log.info("=" * 60)
    log.info("Timing OUR method (unique_tokens – no LLM required)")
    log.info("=" * 60)

    # We only need a lightweight sentence-transformer for the embedding step
    # and a tokenizer for unique_tokens; no heavy LLM is required.
    st_model = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    # Use first COLA model's tokenizer (or any) just for tokenization
    our_tokenizer = AutoTokenizer.from_pretrained(args.cola_models[0], trust_remote_code=True)
    if our_tokenizer.pad_token is None:
        our_tokenizer.pad_token = our_tokenizer.eos_token

    for dataset_name in args.datasets:
        raw_dataset = load_raw_dataset(dataset_name)
        total_size = len(raw_dataset)

        for fraction in FRACTIONS:
            num_samples = int(total_size * fraction)
            if num_samples < args.nsamples:
                log.warning(
                    f"Skipping {dataset_name} @ {fraction*100:.0f}%: "
                    f"only {num_samples} samples (< nsamples={args.nsamples})"
                )
                continue

            subset = raw_dataset.select(range(num_samples))
            tokenized = get_tokenized_data(subset, our_tokenizer, dataset_name, max_length=args.max_length)

            log.info(
                f"[OUR] {dataset_name} – {num_samples} samples ({fraction*100:.0f}%)"
            )

            # tokenization time
            tok_start = time.perf_counter()
            _ = get_tokenized_data(subset, our_tokenizer, dataset_name, max_length=args.max_length)
            tok_time = time.perf_counter() - tok_start

            #  selection time 
            sel_start = time.perf_counter()
            _ = prepare_calibration(
                model=st_model,
                dataloader=[tokenized],
                nsamples=args.nsamples,
                type="unique_tokens",
                distance="flatten",
                model_name="scalability_test",
                dataset_name=dataset_name,
                tokenizer=our_tokenizer,
            )
            sel_time = time.perf_counter() - sel_start

            results.append(
                {
                    "method": "ours",
                    "model": "none (model-free)",
                    "dataset": dataset_name,
                    "total_pool": total_size,
                    "fraction": fraction,
                    "n_used": num_samples,
                    "nsamples": args.nsamples,
                    "tokenization_time_s": round(tok_time, 3),
                    "selection_time_s": round(sel_time, 3),
                    "total_time_s": round(tok_time + sel_time, 3),
                }
            )
            log.info(f"  tok={tok_time:.2f}s  sel={sel_time:.2f}s  total={tok_time+sel_time:.2f}s")

    ## 2. COLA METHOD (requires the actual LLM for activations)
    log.info("=" * 60)
    log.info("Timing COLA method (activation-based – needs LLM)")
    log.info("=" * 60)

    for model_name in args.cola_models:
        try:
            model, tokenizer = load_model_for_cola(model_name, device)
        except Exception as e:
            log.error(f"Could not load {model_name}: {e}")
            continue

        safe_model_name = model_name.replace("/", "-")

        for dataset_name in args.datasets:
            raw_dataset = load_raw_dataset(dataset_name)
            total_size = len(raw_dataset)

            for fraction in FRACTIONS:
                num_samples = int(total_size * fraction)
                if num_samples < args.nsamples:
                    log.warning(
                        f"Skipping {dataset_name} @ {fraction*100:.0f}% for {model_name}: "
                        f"only {num_samples} samples (< nsamples={args.nsamples})"
                    )
                    continue

                subset = raw_dataset.select(range(num_samples))

                log.info(
                    f"[COLA] {model_name} | {dataset_name} – "
                    f"{num_samples} samples ({fraction*100:.0f}%)"
                )

                # preprocessing time (COLA stage-2 style) 
                pre_start = time.perf_counter()
                processed_samples = prepare_cola_processed_samples(
                    subset, dataset_name, tokenizer, max_length=args.max_length
                )
                pre_time = time.perf_counter() - pre_start

                # selection time (COLA stage-3: activations + clustering)
                sel_start = time.perf_counter()
                try:
                    _ = select_samples(
                        processed_samples,
                        model,
                        tokenizer,
                        num_clusters=args.nsamples,
                        device=device,
                        batch_size=4,
                    )
                    sel_time = time.perf_counter() - sel_start
                except Exception as e:
                    sel_time = float("nan")
                    log.error(f"  COLA select_samples failed: {e}")

                results.append(
                    {
                        "method": "cola",
                        "model": model_name,
                        "dataset": dataset_name,
                        "total_pool": total_size,
                        "fraction": fraction,
                        "n_used": num_samples,
                        "nsamples": args.nsamples,
                        "tokenization_time_s": round(pre_time, 3),
                        "selection_time_s": round(sel_time, 3) if sel_time == sel_time else sel_time,
                        "total_time_s": round(pre_time + sel_time, 3) if sel_time == sel_time else float("nan"),
                    }
                )
                log.info(f"  pre={pre_time:.2f}s  sel={sel_time:.2f}s")

        # Free GPU memory before loading the next model
        del model
        torch.cuda.empty_cache()

    # 3. Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    log.info(f"Results saved to {args.output}")
    print(df.to_string(index=False))
    
    
    