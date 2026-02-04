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
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class WrappedGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            # Wanda usually averages over total number of tokens
            tmp = inp.shape[0] * inp.shape[1]
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        # scaler_row is the running average of squared activations
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

def get_last_layers_wanda_info(model, calib_samples, n_layers=3, device="cuda", top_k=0.05):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        return {}

    total_layers = len(layers)
    target_layer_indices = list(range(max(0, total_layers - n_layers), total_layers))
    
    wrapped_modules = {}
    handles = []
    
    def get_hook(name):
        def hook(module, inp, out):
            if name not in wrapped_modules:
                wrapped_modules[name] = WrappedGPT(module)
            wrapped_modules[name].add_batch(inp[0].data, out.data)
        return hook

    for idx in target_layer_indices:
        layer = layers[idx]
        subset = find_layers(layer)
        for name, module in subset.items():
            full_name = f"layer_{idx}_{name}"
            handles.append(module.register_forward_hook(get_hook(full_name)))

    model.eval()
    with torch.no_grad():
        for item in calib_samples:
            input_ids = item["input_ids"]
            attention_mask = item.get("attention_mask")
            if not isinstance(input_ids, torch.Tensor):
                input_ids_t = torch.tensor(input_ids).to(device)
            else:
                input_ids_t = input_ids.to(device)
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask_t = torch.tensor(attention_mask).to(device)
                else:
                    attention_mask_t = attention_mask.to(device)
            else:
                attention_mask_t = None
            
            if len(input_ids_t.shape) == 1:
                input_ids_t = input_ids_t.unsqueeze(0)
            if attention_mask_t is not None and len(attention_mask_t.shape) == 1:
                attention_mask_t = attention_mask_t.unsqueeze(0)
                
            model(input_ids_t, attention_mask=attention_mask_t)

    for h in handles:
        h.remove()

    results = {}
    distributions = {}
    for idx in target_layer_indices:
        layer_scores = []
        layer_activations = []
        for full_name, wrapped in wrapped_modules.items():
            if full_name.startswith(f"layer_{idx}_"):
                # Wanda Score: |W| * sqrt(S)
                W_metric = torch.abs(wrapped.layer.weight.data) * torch.sqrt(wrapped.scaler_row.reshape((1, -1)))
                layer_scores.append(W_metric.view(-1))
                # Activations: sqrt(S)
                layer_activations.append(torch.sqrt(wrapped.scaler_row).view(-1))
        
        if layer_scores:
            all_layer_scores = torch.cat(layer_scores)
            
            # Use activations for ridge plots as they are more sensitive to calibration data
            all_layer_acts = torch.cat(layer_activations)
            sample_size = 10000
            if len(all_layer_acts) > sample_size:
                indices = torch.randperm(len(all_layer_acts))[:sample_size]
                distributions[f"layer_{idx}"] = all_layer_acts[indices].cpu().numpy()
            else:
                distributions[f"layer_{idx}"] = all_layer_acts.cpu().numpy()

            # If top_k is >= 1, treat as absolute number, else as ratio
            k = int(top_k) if top_k >= 1 else int(len(all_layer_acts) * top_k)
            k = max(1, min(k, len(all_layer_acts)))
            
            top_values, _ = torch.topk(all_layer_acts, k)
            results[f"mean_act_layer_{idx}"] = top_values.mean().item()
    
    if results:
        results["mean_act_last_layers_avg"] = sum(results.values()) / len(results)
    
    return results, distributions

def get_layer_wanda_metrics(model, calib_samples, n_layers=3, device="cuda"):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        return {}

    total_layers = len(layers)
    target_layer_indices = list(range(max(0, total_layers - n_layers), total_layers))
    
    wrapped_modules = {}
    handles = []
    
    def get_hook(name):
        def hook(module, inp, out):
            if name not in wrapped_modules:
                wrapped_modules[name] = WrappedGPT(module)
            wrapped_modules[name].add_batch(inp[0].data, out.data)
        return hook

    for idx in target_layer_indices:
        layer = layers[idx]
        subset = find_layers(layer)
        for name, module in subset.items():
            full_name = f"layer_{idx}_{name}"
            handles.append(module.register_forward_hook(get_hook(full_name)))

    model.eval()
    with torch.no_grad():
        for item in calib_samples:
            input_ids = item["input_ids"]
            attention_mask = item.get("attention_mask")
            if not isinstance(input_ids, torch.Tensor):
                input_ids_t = torch.tensor(input_ids).to(device)
            else:
                input_ids_t = input_ids.to(device)
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask_t = torch.tensor(attention_mask).to(device)
                else:
                    attention_mask_t = attention_mask.to(device)
            else:
                attention_mask_t = None
            
            if len(input_ids_t.shape) == 1:
                input_ids_t = input_ids_t.unsqueeze(0)
            if attention_mask_t is not None and len(attention_mask_t.shape) == 1:
                attention_mask_t = attention_mask_t.unsqueeze(0)
                
            model(input_ids_t, attention_mask=attention_mask_t)

    for h in handles:
        h.remove()

    metrics = {}
    for idx in target_layer_indices:
        for full_name, wrapped in wrapped_modules.items():
            if full_name.startswith(f"layer_{idx}_"):
                W_metric = torch.abs(wrapped.layer.weight.data) * torch.sqrt(wrapped.scaler_row.reshape((1, -1)))
                metrics[full_name] = W_metric.cpu()
    return metrics

def compute_mask_overlap(metrics1, metrics2, sparsity):
    overlaps = {}
    for name in metrics1:
        if name not in metrics2: continue
        
        m1 = metrics1[name] # (rows, cols)
        m2 = metrics2[name] # (rows, cols)
        
        rows, cols = m1.shape
        k = int(cols * (1 - sparsity))
        if k == 0: k = 1
        
        # Get topk per row (as Wanda prunes per row)
        _, top_idx1 = torch.topk(m1, k, dim=1)
        _, top_idx2 = torch.topk(m2, k, dim=1)
        
        # Efficiently compute overlap using bitmask or flattened indexed
        row_offsets = torch.arange(rows, device=m1.device).unsqueeze(1) * cols
        flat_idx1 = (top_idx1 + row_offsets).view(-1)
        flat_idx2 = (top_idx2 + row_offsets).view(-1)
        
        overlap_count = torch.isin(flat_idx1, flat_idx2).sum().item()
        overlaps[name] = overlap_count / (rows * k)
    return overlaps

def plot_mask_overlap(overlaps, dataset_name, output_dir="plots/mask_overlap"):
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for name, overlap in overlaps.items():
        parts = name.split("_")
        # Handle cases where module name might have underscores
        layer_idx = parts[1]
        module_name = "_".join(parts[2:])
        data.append({"Layer": layer_idx, "Module": module_name, "Overlap": overlap})
    
    df = pd.DataFrame(data)
    df["Layer"] = df["Layer"].astype(int)
    # Pivot for heatmap
    pivot_df = df.pivot(index="Layer", columns="Module", values="Overlap").sort_index()
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Pruning Mask Overlap (Random vs Unique Tokens) - {dataset_name}")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"overlap_{dataset_name}.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_ridge_activations(distributions_list, dataset_name, output_dir="plots/activations_ridge"):
    """
    distributions_list: list of dicts {'label': ..., 'layer': ..., 'values': np.array}
    """
    if not distributions_list:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(distributions_list)
    
    layers = df['layer'].unique()
    for layer in layers:
        layer_df = df[df['layer'] == layer].copy()
        
        expanded_data = []
        for _, row in layer_df.iterrows():
            for val in row['values']:
                expanded_data.append({'Label': row['label'], 'Activation': val})
        
        plot_df = pd.DataFrame(expanded_data)
        
        # Filter for plotting (optional, if values are too many)
        # plot_df = plot_df.groupby('Label').sample(n=2000, replace=True)

        # Initialize the FacetGrid object
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        labels = plot_df['Label'].unique()
        pal = sns.cubehelix_palette(len(labels), rot=-.25, light=.7)
        g = sns.FacetGrid(plot_df, row="Label", hue="Label", aspect=15, height=.5, palette=pal)

        # Draw the densities
        g.map(sns.kdeplot, "Activation", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, "Activation", clip_on=False, color="w", lw=2, bw_adjust=.5)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        def label_axes(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label_axes, "Activation")
        g.figure.subplots_adjust(hspace=-.25)
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        plt.suptitle(f"Activation Distribution Evolution - {dataset_name} - Layer {layer}", y=0.98)
        
        save_path = os.path.join(output_dir, f"ridge_{dataset_name}_layer_{layer}.png")
        plt.savefig(save_path)
        plt.close()
        sns.set_theme() # Reset

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
            # "rte",
            # "hellaswag",
            "winogrande",
            "arc_challenge",
            # "arc_easy",
            # "openbookqa",
            # "ds1000",
            # "race",
            # "mawps"
            #"boolq gsm8k arc_challenge"
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
        default="results/intersection_test_with_activations_randomwords.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--step_size", type=int, default=5, help="Step size for adding tail samples"
    )
    parser.add_argument(
        "--top_k", type=float, default=0.05, help="Ratio of top Activation scores to average"
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
        
        # Reset model to original state at the beginning of each dataset to avoid carry-over pruning
        model.load_state_dict(original_state_dict)

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
            type="random_words",
            tokenizer=tokenizer,
            dataset_name=d_name,
            model_name=args.model.replace("/", "-")
        )

        # Convert to lists
        rand_list = [rand_calib[i] for i in range(len(rand_calib))]
        unique_list = [unique_calib[i] for i in range(len(unique_calib))]

        # Calculate mask overlap between Random and Unique tokens 
        log.info(f"Calculating mask overlap for {d_name}...")
        metrics_rand = get_layer_wanda_metrics(model, rand_list, n_layers=3, device=device)
        metrics_unique = get_layer_wanda_metrics(model, unique_list, n_layers=3, device=device)
        
        mask_overlaps = compute_mask_overlap(metrics_rand, metrics_unique, args.sparsity)
        plot_path = plot_mask_overlap(mask_overlaps, f"{d_name}_sparsity_{args.sparsity}")
        log.info(f"Mask overlap plot saved to {plot_path}")
        
        mean_overlap = sum(mask_overlaps.values()) / len(mask_overlaps)
        log.info(f"[{d_name}] Mean mask overlap (last layers): {mean_overlap:.4f}")

        # List to store data for ridge plots
        all_ridge_data = []

        # 2. Find intersection using sentence embeddings
        log.info(f"Calculating intersection for {d_name}...")
        
        def extract_text(item, tokenizer):
            if "text" in item:
                return item["text"]
            # Fallback if text is missing (e.g. from cache or weird dataset behavior)
            return tokenizer.decode(item["input_ids"], skip_special_tokens=True)

        rand_texts = [extract_text(item, tokenizer) for item in rand_list]
        unique_texts = [extract_text(item, tokenizer) for item in unique_list]
        
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

            # Get Activation metrics for last layers before pruning
            log.info(f"Calculating Activation metrics for last layers...")
            act_metrics, dists = get_last_layers_wanda_info(model, calib_samples, n_layers=3, device=device, top_k=args.top_k)
            
            # Store distributions for ridge plot
            for layer_key, values in dists.items():
                all_ridge_data.append({
                    "label": label,
                    "layer": layer_key,
                    "values": values
                })
            
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
                        "mean_mask_overlap": mean_overlap,
                    }
                    # Add Activation metrics to row
                    row.update(act_metrics)
                    
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

        # Generate Ridge Plots for the current dataset
        log.info(f"Generating Activation Ridge Plots for {d_name}...")
        plot_ridge_activations(all_ridge_data, d_name)

    log.info(f"Experiment finished. All results are in {args.output_csv}")

if __name__ == "__main__":
    main()