import argparse
import os
import sys
import torch
import pandas as pd
import logging
from filelock import FileLock
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "2SSP"))
from src.pruning import two_stage_2ssp

from data import get_dataset, get_text_from_item
from similarity_check import prepare_calibration
from eval import evaluate_model

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)


def get_existing_original_results(output_csv, model_name):
    if not os.path.exists(output_csv):
        return []
    try:
        df = pd.read_csv(output_csv)
        model_df = df[df["model"] == model_name]
        if model_df.empty:
            return []

        subset = model_df[["task", "metric", "original_value"]].drop_duplicates()
        subset = subset[subset["original_value"].notnull()]

        results = []
        for _, row in subset.iterrows():
            results.append(
                {
                    "task": row["task"],
                    "metric": row["metric"],
                    "value": row["original_value"],
                }
            )
        return results
    except Exception as e:
        log.error(f"Error reading existing results: {e}")
        return []


def get_tokenized_data(dataset, tokenizer, dataset_name, max_length=2048):
    """Tokenize a dataset with a given max_length (window size)."""
    texts = [get_text_from_item(item, dataset_name) for item in dataset]
    processed_dataset = []
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
            processed_dataset.append(
                {
                    "input_ids": encoded["input_ids"][j],
                    "attention_mask": encoded["attention_mask"][j],
                }
            )
    return processed_dataset


def process_results(results_dict):
    """Extracts clean metrics from lm_eval output."""
    processed = []
    if "results" in results_dict:
        for task, metrics in results_dict["results"].items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and "stderr" not in metric_name:
                    processed.append(
                        {"task": task, "metric": metric_name, "value": value}
                    )
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="nsamples ablation: sweep number of calibration samples with fixed window length"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k"],
        help="Datasets for calibration",
    )
    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        default=[
            # Commonsense
             "boolq",
            # "rte",
             "hellaswag",
            # "winogrande",
            # "arc_challenge",
            # "arc_easy",
            # "openbookqa",
            # Math (MC, no generation)
            "mmlu_high_school_mathematics",
            # Code (MMLU CS proxy, no generation)
            "mmlu_high_school_computer_science",
        ],
        help="Tasks for evaluation (lm_eval names)",
    )
    parser.add_argument(
        "--compression_type",
        type=str,
        choices=["pruning", "quantization", "awq", "2ssp"],
        default="pruning",
        help="Type of compression to perform",
    )
    parser.add_argument(
        "--pruning_type",
        type=str,
        choices=[
            "most_similar",
            "random",
            "decoupled",
            "most_dissimilar",
            "least_perplexity",
            "herding",
            "distribution_matching",
            "distribution_matching_no_outliers",
            "zipf",
            "shuffled_zipf",
            "unique_tokens",
            "random_words",
            "words_dataset",
            "dictionary",
        ],
        default="unique_tokens",
        help="Calibration selection strategy to use across all nsamples values",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=2048,
        help="Fixed sequence length (window size) for calibration",
    )
    parser.add_argument(
        "--nsamples_list",
        nargs="+",
        type=int,
        default=[16, 128, 512, 1024],
        help="Number of calibration samples to sweep over",
    )
    parser.add_argument("--sparsity", type=float, default=0.25, help="Pruning sparsity")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/nsamples_ablation_results.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--save_models", action="store_true", help="Save compressed models to disk"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/pruned",
        help="Directory to save compressed models",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    calibration_type_map = {
        "most_similar": "prototype",
        "most_dissimilar": "most_different",
        "decoupled": "decoupled",
        "least_perplexity": "least_perplexity",
        "random": "random_sample",
        "herding": "herding",
        "distribution_matching": "distribution_matching",
        "distribution_matching_no_outliers": "distribution_matching_no_outliers",
        "zipf": "zipf",
        "shuffled_zipf": "shuffled_zipf",
        "unique_tokens": "unique_tokens",
        "random_words": "random_words",
        "words_dataset": "words_dataset",
        "dictionary": "dictionary",
    }

    # 1. Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Evaluate original model once
    existing_orig_metrics = get_existing_original_results(args.output_csv, args.model)
    existing_tasks = set(m["task"] for m in existing_orig_metrics)
    tasks_to_eval = [t for t in args.eval_tasks if t not in existing_tasks]

    # bfloat16 has float32 dynamic range: needed for AWQ to avoid inf/NaN in SiLU
    model_dtype = torch.bfloat16 if args.compression_type == "awq" else torch.float16

    model = None
    if tasks_to_eval:
        log.info(
            f"Loading original model {args.model} for initial evaluation on: {tasks_to_eval}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        orig_raw = evaluate_model(args.model, model, tokenizer, tasks_to_eval)
        new_orig_metrics = process_results(orig_raw)
        orig_metrics = (
            [m for m in existing_orig_metrics if m["task"] in args.eval_tasks]
            + new_orig_metrics
        )
    else:
        log.info("All tasks already have original results. Skipping initial evaluation.")
        orig_metrics = [m for m in existing_orig_metrics if m["task"] in args.eval_tasks]

    # Load and tokenize datasets once (window length is fixed)
    log.info(f"Loading and tokenizing datasets with fixed window_length={args.window_length}...")
    safe_model_name = args.model.replace("/", "-")
    calib_name = "_".join(args.datasets)

    all_tokenized_datasets = []
    for d_name in args.datasets:
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            continue
        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            raw_dataset = (
                raw_dataset.get("train")
                or raw_dataset.get("test")
                or raw_dataset[list(raw_dataset.keys())[0]]
            )
        tokenized = get_tokenized_data(raw_dataset, tokenizer, d_name, max_length=args.window_length)
        all_tokenized_datasets.append(tokenized)

    # Lazy-loaded sentence transformer (reused when not using least_perplexity)
    sentence_transformer = None

    # 3. Main loop: sweep over nsamples
    for nsamples in args.nsamples_list:
        log.info(f"\n--- nsamples: {nsamples} ---")

        # Define save path (includes nsamples for caching)
        save_path = os.path.join(
            args.models_dir,
            safe_model_name,
            args.compression_type,
            args.pruning_type,
            str(nsamples),
            str(args.sparsity),
            calib_name,
            f"wlen{args.window_length}",
        )

        pruned_model = None
        use_disk_cache = args.compression_type in ("quantization", "awq")
        if use_disk_cache and os.path.exists(os.path.join(save_path, "config.json")):
            log.info(f"Found existing compressed model at {save_path}. Loading...")
            pruned_model = AutoModelForCausalLM.from_pretrained(
                save_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            log.info(f"No existing model found at {save_path}. Compressing...")

            if model is None:
                log.info(f"Loading original model {args.model}...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=model_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )

            # Select calibration subset for the current nsamples
            calib_model = model if args.pruning_type == "least_perplexity" else None
            if calib_model is None:
                if sentence_transformer is None:
                    sentence_transformer = SentenceTransformer(
                        "all-MiniLM-L12-v2", device=device
                    )
                calib_model = sentence_transformer

            calibration_data_dicts = prepare_calibration(
                model=calib_model,
                dataloader=all_tokenized_datasets,
                nsamples=nsamples,
                type=calibration_type_map[args.pruning_type],
                distance="flatten",
                model_name=safe_model_name,
                dataset_name=calib_name,
                tokenizer=tokenizer,
            )

            # Compress
            if args.compression_type == "2ssp":
                log.info(f"Structured pruning with 2SSP, sparsity {args.sparsity}...")
                # Convert calibration data to 2SSP format: list of (1, seq_len) tensors
                all_ids = torch.cat([item["input_ids"].view(-1) for item in calibration_data_dicts])
                num_chunks = all_ids.size(0) // args.window_length
                if num_chunks == 0:
                    log.warning(f"Only {all_ids.size(0)} tokens, need >= {args.window_length}. Using all tokens as one sample.")
                    calibration_2ssp = [all_ids.unsqueeze(0)]
                else:
                    calibration_2ssp = [all_ids[i * args.window_length : (i + 1) * args.window_length].unsqueeze(0) for i in range(num_chunks)]
                log.info(f"2SSP calibration: {len(calibration_2ssp)} samples of length {calibration_2ssp[0].size(1)}")
                model.config.use_cache = False
                result = two_stage_2ssp(model, calibration_2ssp, args.sparsity)
                if result is False:
                    log.error("2SSP pruning failed – invalid sparsity parameters")
                    del model
                    torch.cuda.empty_cache()
                    model = None
                    continue
                model = result
            else:
                data_list = [
                    {
                        "input_ids": item["input_ids"].cpu().tolist(),
                        "attention_mask": item["attention_mask"].cpu().tolist(),
                    }
                    for item in calibration_data_dicts
                ]
                calibration_dataset = Dataset.from_list(data_list)

                if args.compression_type == "pruning":
                    log.info(f"Pruning model with sparsity {args.sparsity}...")
                    recipe = WandaPruningModifier(
                        sparsity=args.sparsity, mask_structure="0:0", targets="__ALL__"
                    )
                elif args.compression_type == "quantization":
                    log.info("Quantizing model with GPTQ...")
                    recipe = GPTQModifier(targets="Linear", scheme="W4A16")
                elif args.compression_type == "awq":
                    log.info("Quantizing model with AWQ...")
                    recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

                oneshot(model=model, dataset=calibration_dataset, recipe=recipe)
            log.info("Compression complete.")

            if args.compression_type in ("quantization", "awq"):
                # GPTQ/AWQ pack weights into int4: in-memory model can't run inference.
                # Must save to disk and reload.
                log.info(f"Saving compressed model to {save_path}...")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                del model
                torch.cuda.empty_cache()
                model = None
                log.info(f"Reloading compressed model from {save_path} for evaluation...")
                pruned_model = AutoModelForCausalLM.from_pretrained(
                    save_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # Pruning/2SSP modify weights in-place; model is still valid for inference.
                pruned_model = model
                model = None

        # 4. Evaluate compressed model
        log.info(f"Evaluating compressed model (nsamples={nsamples})...")
        pruned_raw = evaluate_model(args.model, pruned_model, tokenizer, args.eval_tasks)
        pruned_metrics = process_results(pruned_raw)

        # 5. Save results to CSV
        log.info(f"Saving results for nsamples={nsamples} to {args.output_csv}...")
        rows = []
        for orig in orig_metrics:
            pruned_val = next(
                (
                    p["value"]
                    for p in pruned_metrics
                    if p["task"] == orig["task"] and p["metric"] == orig["metric"]
                ),
                None,
            )
            rows.append(
                {
                    "model": args.model,
                    "task": orig["task"],
                    "metric": orig["metric"],
                    "original_value": orig["value"],
                    "pruned_value": pruned_val,
                    "compression_type": args.compression_type,
                    "pruning_type": args.pruning_type,
                    "sparsity": args.sparsity,
                    "calibration_datasets": calib_name,
                    "window_length": args.window_length,
                    "nsamples": nsamples,
                }
            )

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

        lock_path = args.output_csv + ".lock"
        lock = FileLock(lock_path)
        with lock:
            df.to_csv(
                args.output_csv,
                mode="a",
                header=not os.path.exists(args.output_csv),
                index=False,
            )

        # Reload original model before next nsamples iteration (compression modifies in-place)
        if nsamples != args.nsamples_list[-1]:
            log.info("Reloading original model for next nsamples value...")
            if model is not None:
                del model
            del pruned_model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

    log.info("All nsamples experiments finished successfully.")


if __name__ == "__main__":
    main()
