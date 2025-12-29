import argparse
import os
import torch
import pandas as pd
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
from sentence_transformers import SentenceTransformer

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

def process_results(results_dict):
    """Extracts clean metrics from lm_eval output."""
    processed = []
    if "results" in results_dict:
        for task, metrics in results_dict["results"].items():
            for metric_name, value in metrics.items():
                # Filter out stderr and non-numeric values
                if isinstance(value, (int, float)) and "stderr" not in metric_name:
                    processed.append({
                        "task": task,
                        "metric": metric_name,
                        "value": value
                    })
    return processed

def main():
    parser = argparse.ArgumentParser(description="Run Pruning Experiment with lm_eval")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path")
    parser.add_argument("--datasets", nargs="+", default=["winogrande"], help="Datasets for calibration")
    parser.add_argument("--eval_tasks", nargs="+", default=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], help="Tasks for evaluation (lm_eval names)")
    parser.add_argument("--pruning_types", nargs="+", choices=["most_similar", "random", "decoupled", "most_dissimilar", "least_perplexity", "herding", "distribution_matching"], default=["most_similar", "random", "decoupled", "most_dissimilar", "least_perplexity", "herding", "distribution_matching"], help="Types of pruning to perform")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Pruning sparsity")
    parser.add_argument("--output_csv", type=str, default="results/experiment_results.csv", help="Output CSV file")
    parser.add_argument("--save_models", action="store_true", help="Save pruned models to disk")
    parser.add_argument("--models_dir", type=str, default="models/pruned", help="Directory to save pruned models")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Evaluate original model once
    log.info(f"Loading original model {args.model} for initial evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    log.info(f"Evaluating original model on: {args.eval_tasks}")
    orig_raw = evaluate_model(args.model, model, tokenizer, args.eval_tasks)
    orig_metrics = process_results(orig_raw)
    
    # Prepare calibration data base (tokenized) once
    log.info("Preparing base tokenized data for calibration...")
    all_tokenized_data = []
    for d_name in args.datasets:
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None: continue
        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            dataset = raw_dataset.get("train") or raw_dataset.get("test") or raw_dataset[list(raw_dataset.keys())[0]]
        else:
            dataset = raw_dataset
        tokenized_data = get_tokenized_data(dataset, tokenizer, d_name)
        all_tokenized_data.extend(tokenized_data)

    calibration_type_map = {
        "most_similar": "prototype",
        "most_dissimilar": "most_different",
        "decoupled": "decoupled",
        "least_perplexity": "least_perplexity",
        "random": "random_sample",
        "herding": "herding",
        "distribution_matching": "distribution_matching",
    }

    # 3. Loop through pruning types
    for p_type in args.pruning_types:
        log.info(f"\n--- Starting process for pruning type: {p_type} ---")
        
        # Define save path
        safe_model_name = args.model.replace("/", "-")
        calib_name = "_".join(args.datasets)
        save_path = os.path.join(args.models_dir, safe_model_name, p_type, str(args.nsamples), str(args.sparsity), calib_name)
        
        pruned_model = None
        if os.path.exists(os.path.join(save_path, "config.json")):
            log.info(f"Found existing pruned model at {save_path}. Loading...")
            pruned_model = AutoModelForCausalLM.from_pretrained(
                save_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
        else:
            log.info(f"No existing model found at {save_path}. Pruning...")
            # Reload original model to ensure clean state for each pruning type
            # (If we already have 'model' from step 2, we can use it for the first p_type, 
            # but it's safer to just reload or manage state carefully)
            # For simplicity and memory, we'll reload.
            
            # If 'model' is already loaded and hasn't been pruned yet, we use it.
            # But after the first 'oneshot', 'model' is modified.
            
            # Prepare calibration for this specific type
            calib_model = model if p_type == "least_perplexity" else SentenceTransformer("all-MiniLM-L12-v2", device=device)
            
            calibration_data_dicts = prepare_calibration(
                model=calib_model,
                dataloader=[all_tokenized_data],
                nsamples=args.nsamples,
                type=calibration_type_map[p_type],
                distance="flatten",
                model_name=safe_model_name,
                dataset_name=calib_name,
                tokenizer=tokenizer,
            )
            
            data_list = [{"input_ids": item["input_ids"].cpu().tolist(), 
                          "attention_mask": item["attention_mask"].cpu().tolist()} 
                         for item in calibration_data_dicts]
            calibration_dataset = Dataset.from_list(data_list)

            # Prune
            log.info(f"Pruning model with sparsity {args.sparsity}...")
            recipe = WandaPruningModifier(sparsity=args.sparsity, mask_structure="0:0", targets="__ALL__")
            oneshot(model=model, dataset=calibration_dataset, recipe=recipe)
            log.info("Pruning complete.")
            
            # if args.save_models:
            #     log.info(f"Saving pruned model to {save_path}...")
            #     os.makedirs(save_path, exist_ok=True)
            #     model.save_pretrained(save_path)
            #     tokenizer.save_pretrained(save_path)
            
            pruned_model = model

        # 4. Evaluate pruned model
        log.info(f"Evaluating pruned model ({p_type})...")
        pruned_raw = evaluate_model(args.model, pruned_model, tokenizer, args.eval_tasks)
        pruned_metrics = process_results(pruned_raw)

        # 5. Save results to CSV
        log.info(f"Saving results for {p_type} to {args.output_csv}...")
        rows = []
        for orig in orig_metrics:
            pruned_val = next((p["value"] for p in pruned_metrics if p["task"] == orig["task"] and p["metric"] == orig["metric"]), None)
            rows.append({
                "model": args.model,
                "task": orig["task"],
                "metric": orig["metric"],
                "original_value": orig["value"],
                "pruned_value": pruned_val,
                "pruning_type": p_type,
                "nsamples": args.nsamples,
                "sparsity": args.sparsity,
                "calibration_datasets": calib_name
            })

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)
        
        # If we are going to the next p_type, we MUST reload the original model
        # because 'model' (which is 'pruned_model') is now modified.
        if p_type != args.pruning_types[-1]:
            log.info("Reloading original model for the next pruning type...")
            del model
            del pruned_model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )

    log.info("All experiments finished successfully.")

if __name__ == "__main__":
    main()
