import argparse
import os
import torch
import pandas as pd
import logging
import copy
from filelock import FileLock
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from data import get_dataset, get_text_from_item
from similarity_check import prepare_calibration
from eval import evaluate_model
from prune import get_tokenized_data

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)

def get_embeddings(texts, st_model, device="cuda"):
    return st_model.encode(texts, convert_to_tensor=True, device=device)

def main():
    parser = argparse.ArgumentParser(description="Run Pruning Experiment with Intersection Analysis")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-8B", help="Model name or path"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=[            
            "boolq",
            "rte",
            "hellaswag",
            "winogrande",
            "arc_challenge",
            "arc_easy",
            "openbookqa",
            "ds1000",
            "race",
            "mawps"
            ], help="Datasets for calibration"
    )
    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        default=[
            "boolq",
            "arc_challenge",
            "winogrande",
            
        ],
        help="Tasks for evaluation (lm_eval names)",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="Pruning sparsity"
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of samples for each technique"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/intersection_test.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--step_size", type=int, default=5, help="Step size for adding tail samples"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    # Save original state dict to reset model after each pruning
    log.info("Saving original model state...")
    original_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    st_model = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    all_results = []

    for d_name in args.datasets:
        log.info(f"=== Processing calibration dataset: {d_name} ===")
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            continue
        
        # Handle different dataset structures
        if isinstance(raw_dataset, dict):
            if "train" in raw_dataset:
                dataset = raw_dataset["train"]
            elif "validation" in raw_dataset:
                dataset = raw_dataset["validation"]
            else:
                dataset = list(raw_dataset.values())[0]
        else:
            dataset = raw_dataset
            
        tokenized = get_tokenized_data(dataset, tokenizer, d_name, return_tensors=True)
        current_tokenized_data = [Dataset.from_list(tokenized)]

        # 1. Get calibration data for random and unique_tokens
        log.info(f"Sampling for {d_name}: Random")
        rand_calib = prepare_calibration(
            model=st_model,
            dataloader=current_tokenized_data,
            nsamples=args.nsamples,
            type="random_sample",
            tokenizer=tokenizer,
            dataset_name=d_name,
            model_name=args.model.replace("/", "-")
        )
        
        log.info(f"Sampling for {d_name}: Unique Tokens")
        unique_calib = prepare_calibration(
            model=st_model,
            dataloader=current_tokenized_data,
            nsamples=args.nsamples,
            type="unique_tokens",
            tokenizer=tokenizer,
            dataset_name=d_name,
            model_name=args.model.replace("/", "-")
        )

        # Convert to lists
        rand_list = [rand_calib[i] for i in range(len(rand_calib))]
        unique_list = [unique_calib[i] for i in range(len(unique_calib))]

        # 2. Find intersection using sentence embeddings
        log.info(f"Calculating intersection for {d_name}...")
        
        rand_texts = [item["text"] for item in rand_list]
        unique_texts = [item["text"] for item in unique_list]
        
        rand_embs = get_embeddings(rand_texts, st_model, device=device)
        unique_embs = get_embeddings(unique_texts, st_model, device=device)
        
        center_rand = rand_embs.mean(0)
        dists_rand = torch.norm(rand_embs - center_rand, dim=1)
        dists_unique = torch.norm(unique_embs - center_rand, dim=1)
        
        # Threshold for "intersection"
        threshold = dists_rand.mean() + dists_rand.std()
        
        intersection_indices = (dists_unique <= threshold).nonzero(as_tuple=True)[0].tolist()
        tail_indices = (dists_unique > threshold).nonzero(as_tuple=True)[0].tolist()
        
        tail_indices = sorted(tail_indices, key=lambda i: dists_unique[i].item(), reverse=True)

        log.info(f"[{d_name}] Intersection samples: {len(intersection_indices)}")
        log.info(f"[{d_name}] Tail samples: {len(tail_indices)}")

        intersection_samples = [unique_list[i] for i in intersection_indices]
        tail_samples = [unique_list[i] for i in tail_indices]
        # For comparison, take random samples from rand_list. 
        random_extra_samples = list(rand_list)

        def run_experiment(calib_samples, label):
            log.info(f"--- Dataset: {d_name} | Label: {label} (N={len(calib_samples)}) ---")
            
            # Reset model
            model.load_state_dict(original_state_dict)
            
            # Prepare calibration dataset
            data_list = []
            for item in calib_samples:
                ids = item["input_ids"]
                mask = item.get("attention_mask")
                if isinstance(ids, torch.Tensor): ids = ids.cpu().numpy().tolist()
                if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy().tolist()
                    
                data_list.append({"input_ids": ids, "attention_mask": mask})
            calib_dataset = Dataset.from_list(data_list)
            
            # Prune
            recipe = WandaPruningModifier(sparsity=args.sparsity, mask_structure="0:0", targets="__ALL__")
            oneshot(model=model, dataset=calib_dataset, recipe=recipe)
            
            # Evaluate on the target tasks
            eval_res = evaluate_model(args.model, model, tokenizer, args.eval_tasks)
            
            # Store metrics
            results_rows = []
            if "results" in eval_res:
                for task, metrics in eval_res["results"].items():
                    row = {
                        "model": args.model,
                        "calibration_dataset": d_name,
                        "target_task": task,
                        "label": label,
                        "n_samples": len(calib_samples),
                        "sparsity": args.sparsity,
                    }
                    for m, v in metrics.items():
                        if isinstance(v, (int, float)) and "stderr" not in m:
                            row[f"{m}"] = v
                    results_rows.append(row)
            return results_rows

        # 3. Prune and evaluate on intersection data
        if intersection_samples:
            all_results.extend(run_experiment(intersection_samples, "intersection"))
        else:
            log.warning(f"No intersection samples found for {d_name}!")

        # 4. Add tail samples in steps
        current_calib_tail = list(intersection_samples)
        for i in range(0, len(tail_samples), args.step_size):
            chunk = tail_samples[i : i + args.step_size]
            current_calib_tail.extend(chunk)
            label = f"tail_add_{len(current_calib_tail)}"
            all_results.extend(run_experiment(current_calib_tail, label))
            
        # 5. Add random samples in steps (Comparison)
        current_calib_rand = list(intersection_samples)
        for i in range(0, min(len(random_extra_samples), len(tail_samples)), args.step_size):
            chunk = random_extra_samples[i : i + args.step_size]
            current_calib_rand.extend(chunk)
            label = f"random_add_{len(current_calib_rand)}"
            all_results.extend(run_experiment(current_calib_rand, label))

        # Save results for THIS dataset incrementally
        if all_results:
            os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
            df = pd.DataFrame(all_results)
            with FileLock(args.output_csv + ".lock"):
                df.to_csv(
                    args.output_csv,
                    mode="a",
                    header=not os.path.exists(args.output_csv),
                    index=False,
                )
            log.info(f"Results for {d_name} saved to {args.output_csv}")
            all_results = [] # Clear for next dataset

    log.info(f"Experiment finished. All results are in {args.output_csv}")

if __name__ == "__main__":
    main()