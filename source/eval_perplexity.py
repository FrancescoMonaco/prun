import argparse
import os
import torch
import torch.nn.functional as F
import pandas as pd
import logging
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot
from sentence_transformers import SentenceTransformer
import sys

# Aggiungi COLA, 2SSP e source al sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "COLA"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "2SSP"))
from src.pruning import two_stage_2ssp

from data import get_dataset, get_text_from_item
from similarity_check import prepare_calibration
from eval import evaluate_model

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)

def get_tokenized_data(dataset, tokenizer, dataset_name, max_length=128):
    processed_dataset = []
    for item in dataset:
        text = get_text_from_item(item, dataset_name)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        processed_dataset.append(
            {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        )
    return processed_dataset

def load_raw_texts(dataset_name: str, max_count: int = 4000) -> list[str]:
    """Load raw texts from a dataset. C4 is loaded via streaming to avoid full download."""
    from datasets import load_dataset

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = []
        for item in ds:
            text = item.get("text", "")
            if text and len(text.strip()) > 0:
                texts.append(text)
            if len(texts) >= max_count:
                break
        return texts

    elif dataset_name.startswith("c4"):
        # Use streaming to avoid downloading the full dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = []
        for item in ds:
            text = item.get("text", "")
            if text and len(text.strip()) > 0:
                texts.append(text)
            if len(texts) >= max_count:
                break
        return texts

    else:
        raw = get_dataset(dataset_name)
        if raw is None:
            raise ValueError(f"Could not load dataset {dataset_name}.")
        if isinstance(raw, dict):
            raw = raw.get("train") or raw.get("test") or raw[list(raw.keys())[0]]
        texts = []
        for item in raw:
            text = get_text_from_item(item, dataset_name)
            if text and len(text.strip()) > 0:
                texts.append(text)
            if len(texts) >= max_count:
                break
        return texts


def tokenize_texts(texts: list[str], tokenizer, max_seq_len: int) -> list[dict]:
    """Tokenize a list of raw texts into input_ids / attention_mask dicts."""
    tokenized = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized.append(
            {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        )
    return tokenized


def build_recipe(exp_type: str, sparsity: float):
    """Return the llmcompressor recipe for a given compression technique."""
    if exp_type == "wanda":
        return WandaPruningModifier(sparsity=sparsity, targets="__ALL__")
    elif exp_type == "gptq":
        return GPTQModifier(targets="Linear", scheme="W4A16")
    elif exp_type == "awq":
        return AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    return None


@torch.no_grad()
def _compute_ppl_gemma(model, chunks: list) -> float:
    """
    BOS-prepend per-window PPL for Gemma2.
    Each chunk is a (1, seq_len) tensor.  A BOS token is prepended and the
    last token is dropped so length stays constant, then NLL is accumulated
    with a token-count-weighted running average.
    """
    import math
    device = next(model.parameters()).device
    bos_token_id = model.config.bos_token_id

    nll_running = torch.tensor(0.0, device=device)
    tokens_total = 0
    skipped = 0

    model.eval()
    for chunk in chunks:
        inputs = chunk.to(device)          # (1, seq_len)
        if inputs.size(1) < 2:
            skipped += 1
            continue
        bos = torch.full((1, 1), int(bos_token_id), dtype=inputs.dtype, device=device)
        inputs = torch.cat([bos, inputs[:, :-1]], dim=1)
        lm_logits = model(inputs).logits.float()
        shift_logits = lm_logits[:, :-1].contiguous()
        shift_labels = inputs[:, 1:]
        n_tok = shift_labels.numel()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        if not torch.isfinite(loss):
            skipped += 1
            continue
        denom = tokens_total + n_tok
        nll_running = (n_tok / denom) * loss + (tokens_total / denom) * nll_running
        tokens_total += n_tok

    if tokens_total == 0:
        return float("inf")
    if skipped:
        log.info(f"Gemma PPL: skipped {skipped} windows")
    return math.exp(nll_running.item())


def compute_ppl_direct(
    model,
    tokenizer,
    dataset_name: str,
    seq_len: int = 2048,
    n_windows: int = 40,
) -> float:
    """
    Compute token-level perplexity using the concatenated sliding-window approach
    (SparseGPT / Wanda protocol).  Loss is evaluated in float32 regardless of
    model dtype to avoid the float16 underflow that causes inf perplexity on C4.
    """
    import math

    log.info(
        f"Computing direct PPL on {dataset_name} "
        f"(seq_len={seq_len}, max_windows={n_windows})..."
    )

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(d["text"] for d in ds if d["text"].strip())
    elif dataset_name.startswith("c4"):
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in ds:
            if item["text"].strip():
                texts.append(item["text"])
            if len(texts) >= 1100:
                break
        text = " ".join(texts)
    elif dataset_name == "pile":
        from pathlib import Path
        pile_path = Path(__file__).parent.parent / "datasets" / "pile"
        pile_texts = []
        if pile_path.exists():
            import glob
            patterns = ["**/*.jsonl", "**/*.json", "**/*.parquet"]
            local_files = []
            for pat in patterns:
                local_files.extend(sorted(pile_path.glob(pat)))
            if local_files:
                log.info(f"Loading pile from local files: {[str(f) for f in local_files[:3]]}...")
                fmt = "parquet" if any(".parquet" in str(f) for f in local_files) else "json"
                ds = load_dataset(fmt, data_files={"train": [str(f) for f in local_files]}, split="train", streaming=True)
                for item in ds:
                    t = item.get("text", "").strip()
                    if t:
                        pile_texts.append(t)
                    if len(pile_texts) >= 1100:
                        break
        if not pile_texts:
            log.info("Pile local files not found, falling back to HuggingFace streaming...")
            ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
            for item in ds:
                t = item.get("text", "").strip()
                if t:
                    pile_texts.append(t)
                if len(pile_texts) >= 1100:
                    break
        text = " ".join(pile_texts)
    else:
        log.warning(f"Direct PPL not supported for dataset: {dataset_name}")
        return float("nan")

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids  # shape: (1, total_tokens)

    total_tokens = input_ids.size(1)
    max_windows = total_tokens // seq_len
    actual_windows = min(n_windows, max_windows)

    if actual_windows == 0:
        log.warning(f"Not enough tokens for even one window in {dataset_name}")
        return float("nan")

    # Gemma2 uses BOS-per-window approach for accurate PPL
    is_gemma = getattr(model.config, "model_type", "") in ("gemma2", "gemma")
    if is_gemma:
        chunks = [
            input_ids[:, i * seq_len : (i + 1) * seq_len]
            for i in range(actual_windows)
        ]
        ppl = _compute_ppl_gemma(model, chunks)
        log.info(f"Gemma PPL on {dataset_name}: {ppl:.2f} ({actual_windows} windows)")
        return ppl

    nlls = []
    model.eval()
    with torch.no_grad():
        for i in range(actual_windows):
            chunk = input_ids[:, i * seq_len : (i + 1) * seq_len].to(model.device)
            outputs = model(chunk, labels=chunk)
            loss = outputs.loss.float()  # float32 to avoid overflow
            if torch.isfinite(loss):
                nlls.append(loss.item())
            else:
                log.warning(
                    f"Non-finite loss at window {i} for {dataset_name}, skipping."
                )

    if not nlls:
        log.error(f"All windows produced non-finite loss for {dataset_name}.")
        return float("inf")

    avg_nll = sum(nlls) / len(nlls)
    ppl = math.exp(avg_nll)
    log.info(
        f"Direct PPL on {dataset_name}: {ppl:.2f} "
        f"(avg NLL={avg_nll:.4f}, {len(nlls)}/{actual_windows} valid windows)"
    )
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate Perplexity across compression techniques")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B", help="Base model name")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "winogrande",
            "arc_challenge",
            "boolq",
            "hellaswag",
            "openbookqa",
            "rte",
        ],
        help="Datasets used to build mixed calibration data",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument(
        "--calibration_type",
        type=str,
        choices=[
            "random_sample",
            "zipf",
            "shuffled_zipf",
            "unique_tokens",
            "random_words",
            "words_dataset",
            "dictionary",
        ],
        default="random_words",
        help="Calibration sampling strategy (selected via prepare_calibration)",
    )
    parser.add_argument("--sparsity", type=float, default=0.25, help="Pruning sparsity (used by Wanda)")
    parser.add_argument("--ppl_tasks", nargs="+", default=["wikitext", "c4", "pile"], help="Tasks for perplexity evaluation")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max seq len for calibration")
    parser.add_argument("--output_csv", type=str, default="results/perplexity_comparison.csv", help="Output file")

    args = parser.parse_args()
    if not args.datasets:
        raise ValueError("Please provide at least one dataset in --datasets.")
    if args.nsamples <= 0:
        raise ValueError("--nsamples must be > 0.")

    # 1. Load Tokenizer (shared across all runs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build tokenized pools for each calibration dataset
    calibration_label = " ".join(args.datasets)
    all_tokenized_datasets = []

    log.info(f"Preparing tokenized pools from datasets: {args.datasets}")

    for d_name in args.datasets:
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            log.warning(f"Dataset {d_name} unavailable, skipping.")
            continue

        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            dataset_split = (
                raw_dataset.get("train")
                or raw_dataset.get("test")
                or raw_dataset[list(raw_dataset.keys())[0]]
            )
        else:
            dataset_split = raw_dataset

        tokenized_data = get_tokenized_data(dataset_split, tokenizer, d_name, args.max_seq_len)
        if not tokenized_data:
            log.warning(f"Tokenized pool for {d_name} is empty, skipping.")
            continue
        all_tokenized_datasets.append(tokenized_data)
        log.info(f"Loaded {len(tokenized_data)} tokenized samples from {d_name}")

    if not all_tokenized_datasets:
        raise RuntimeError("Unable to build calibration set: no texts loaded from --datasets.")

    # 3. Select calibration samples with the shared utility (same as run_experiment pipeline)
    log.info(
        f"Selecting calibration with prepare_calibration "
        f"(type={args.calibration_type}, nsamples={args.nsamples})"
    )

    sentence_transformer = SentenceTransformer("all-MiniLM-L12-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    calib_token = prepare_calibration(
        model=sentence_transformer,
        dataloader=all_tokenized_datasets,
        nsamples=args.nsamples,
        type=args.calibration_type,
        distance="flatten",
        model_name=args.model.replace("/", "-"),
        dataset_name=args.datasets,
        tokenizer=tokenizer,
    )
    if not calib_token:
        raise RuntimeError("prepare_calibration returned no samples.")
    log.info(f"Calibration data prepared with {len(calib_token)} samples")

    # Three compression techniques + baseline
    experiments = [ "wanda", "gptq", "awq", "2ssp"] #"original", "wanda", "gptq", "awq", "2ssp"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    csv_exists = os.path.isfile(args.output_csv)

    for exp_type in experiments:
        log.info(f"\n--- [calibration: {calibration_label}] Evaluating: {exp_type} ---")

        # Fresh model for every experiment
        # Gemma2 requires eager attention for correct perplexity evaluation
        # (SDPA/FlashAttention break sliding-window + global attention alternation)
        extra_kwargs = {}
        if "gemma" in args.model.lower():
            extra_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
            **extra_kwargs
        )

        if exp_type == "2ssp":
            log.info(f"Structured pruning with 2SSP, sparsity {args.sparsity}...")
            all_ids = torch.cat([item["input_ids"].view(-1) for item in calib_token])
            ssp_seq_len = 2048
            num_chunks = all_ids.size(0) // ssp_seq_len
            if num_chunks == 0:
                log.warning(f"Only {all_ids.size(0)} tokens, need >= {ssp_seq_len} for 2SSP. Using all tokens as one sample.")
                calibration_2ssp = [all_ids.unsqueeze(0)]
            else:
                calibration_2ssp = [all_ids[i * ssp_seq_len : (i + 1) * ssp_seq_len].unsqueeze(0) for i in range(num_chunks)]
            log.info(f"2SSP calibration: {len(calibration_2ssp)} samples of length {calibration_2ssp[0].size(1)}")
            model.config.use_cache = False
            result = two_stage_2ssp(model, calibration_2ssp, args.sparsity)
            if result is False:
                log.error("2SSP pruning failed – invalid sparsity parameters")
                del model
                torch.cuda.empty_cache()
                continue
            model = result
        elif exp_type != "original":
            log.info(f"Applying compression technique: {exp_type}...")
            data_list = [
                {
                    "input_ids": item["input_ids"].cpu().tolist(),
                    "attention_mask": item["attention_mask"].cpu().tolist(),
                }
                for item in calib_token
            ]
            calibration_dataset = Dataset.from_list(data_list)
            recipe = build_recipe(exp_type, args.sparsity)
            oneshot(model=model, dataset=calibration_dataset, recipe=recipe)

        rows = []

        for task_name in args.ppl_tasks:
            log.info(f"Running direct PPL evaluation on: {task_name}...")
            ppl = compute_ppl_direct(
                model, tokenizer, task_name, seq_len=args.max_seq_len
            )
            rows.append({
                "experiment":    exp_type,
                "dataset":       calibration_label,
                "task":          task_name,
                "metric":        "word_perplexity",
                "value":         ppl,
                "sparsity":      args.sparsity if exp_type in ("wanda", "2ssp") else 0,
                "model":         args.model,
                "nsamples":      args.nsamples,
            })

        # Append to CSV (write header only if file does not exist yet)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(
                args.output_csv,
                mode="a",
                index=False,
                header=not csv_exists,
            )
            csv_exists = True  # header already written from now on
            log.info(f"Appended {len(rows)} rows to {args.output_csv}")

        del model
        torch.cuda.empty_cache()

    log.info(f"\nAll results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
