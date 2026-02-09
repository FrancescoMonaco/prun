import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.stats import ks_2samp, kurtosis, skew
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from data import get_dataset
from similarity_check import prepare_calibration, random_words_sampling
from prune import get_tokenized_data


# Set up logging
FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)

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
        self.max_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        
        # Update running mean of squared norms (L2 proxy)
        # self.scaler_row = E[x^2]
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row += torch.norm(inp.type(torch.float32), p=2, dim=1) ** 2 / (self.nsamples + tmp)
        
        # Update max activation per feature
        # inp is [features, tokens], so we look for max along dim 1
        current_max, _ = torch.max(torch.abs(inp), dim=1)
        self.max_row = torch.maximum(self.max_row, current_max)
        
        self.nsamples += tmp

def generate_shuffled_zipf_samples(tokenizer, nsamples=128, seq_len=128, model=None, dataloader=None, dataset_name=None):
    """
    Generates scrambled versions of Zipf-selected samples.
    """
    # First, get the standard Zipf samples
    zipf_calib = prepare_calibration(
        model=model,
        dataloader=dataloader,
        nsamples=nsamples,
        type="zipf",
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        model_name="temp_model" # dummy name
    )
    
    shuffled_samples = []
    
    for i in range(len(zipf_calib)):
        item = zipf_calib[i]
        input_ids = item["input_ids"]
        
        # Convert to list if tensor
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
            
        # Decode to text
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Scramble words
        words = text.split()
        np.random.shuffle(words)
        shuffled_text = " ".join(words)
        
        # Re-encode
        encoded = tokenizer(shuffled_text, truncation=True, max_length=seq_len, return_tensors="pt")
        
        if encoded.input_ids.shape[1] == 0:
            continue
            
        shuffled_samples.append({
            "input_ids": encoded.input_ids[0],
            "attention_mask": encoded.attention_mask[0]
        })
        
    return shuffled_samples

def get_layer_stats(model, calib_samples, n_layers=3, device="cuda"):
    """
    Collects detailed activation statistics for the last n_layers.
    Returns:
        results: dict {layer_name: {stat_name: np.array}}
    """
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
    
    for full_name, wrapped in wrapped_modules.items():
        stats = {}
        
        # 1. RMS Activation (L2-like)
        rms_acts = torch.sqrt(wrapped.scaler_row).cpu()
        stats["activation_rms"] = rms_acts
        
        # 2. Max Activation (L-inf)
        max_acts = wrapped.max_row.cpu()
        stats["activation_max"] = max_acts
        
        # 3. Wanda Metric: |W| * RMS
        # Weights: [out_features, in_features]
        # Activations: [in_features]
        W = torch.abs(wrapped.layer.weight.data).cpu()
        
        # We calculate the Wanda metric importance for each input channel.
        # This is usually sum(|W_col|) * act_col for structured pruning,
        # or just |W_ij| * act_j for unstructured.
        # Since we want to prune weights, let's look at the distribution of "pruning scores" 
        # for ALL weights in the layer.
        
        # Option A: Flattened Wanda scores (too big? 4096*11008 ~ 45M)
        # 45M floats is ~180MB. Per layer. It might fit in RAM but computing KS on it is slow.
        # Let's sample a subset of weights (e.g. 1 Million) to compare distributions.
        
        # wanda_score_matrix = W * rms_acts.unsqueeze(0) # [out, in]
        # flattened = wanda_score_matrix.view(-1)
        
        # To avoid OOM, perform this without full expansion if possible, or just sample.
        # Let's do random sampling of indices.
        out_f, in_f = W.shape
        num_samples = min(1000000, out_f * in_f)
        
        # Random indices
        row_idx = torch.randint(0, out_f, (num_samples,))
        col_idx = torch.randint(0, in_f, (num_samples,))
        
        w_sample = W[row_idx, col_idx]
        act_sample = rms_acts[col_idx]
        wanda_scores_sample = w_sample * act_sample
        
        stats["wanda_score_sample"] = wanda_scores_sample
        
        results[full_name] = stats

    return results

def main():
    parser = argparse.ArgumentParser(description="Compare Random vs Zipf Sampling Activation Distributions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name or path")
    parser.add_argument("--datasets", nargs="+", default=["boolq", "winogrande", "arc_challenge"], help="Datasets to test")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--output_csv", type=str, default="results/pvalues_detailed.csv", help="Output CSV file")
    parser.add_argument("--zipf_type", type=str, default="words_dataset", choices=["zipf", "unique_tokens", "words_dataset"], help="Type of zipf-like sampling to use")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    st_model = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    
    results_list = []

    for d_name in args.datasets:
        log.info(f"=== Processing dataset: {d_name} ===")
        
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            log.warning(f"Could not load dataset {d_name}, skipping.")
            continue

        # Handle different dataset structures
        if isinstance(raw_dataset, dict):
            if "train" in raw_dataset:
                dataset = raw_dataset["train"]
            elif "validation" in raw_dataset:
                dataset = raw_dataset["validation"]
            elif "test" in raw_dataset:
                dataset = raw_dataset["test"]
            else:
                dataset = list(raw_dataset.values())[0]
        else:
            dataset = raw_dataset

        tokenized = get_tokenized_data(dataset, tokenizer, d_name, return_tensors=True)
        current_tokenized_data = [Dataset.from_list(tokenized)]

        # 1. Random Sampling
        log.info(f"{d_name}: Generating Random samples...")
        rand_calib = prepare_calibration(
            model=st_model,
            dataloader=current_tokenized_data,
            nsamples=args.nsamples,
            type="random_sample",
            tokenizer=tokenizer,
            dataset_name=d_name,
            model_name=args.model.replace("/", "-")
        )
        rand_list = [rand_calib[i] for i in range(len(rand_calib))]

        # 2. Zipf/Unique Sampling
        log.info(f"{d_name}: Generating {args.zipf_type} samples...")
        zipf_calib = prepare_calibration(
            model=st_model,
            dataloader=current_tokenized_data,
            nsamples=args.nsamples,
            type=args.zipf_type,
            tokenizer=tokenizer,
            dataset_name=d_name,
            model_name=args.model.replace("/", "-")
        )
        zipf_list = [zipf_calib[i] for i in range(len(zipf_calib))]

        # 3. Random Words Sampling (Baseline Hypothesis)
        log.info(f"{d_name}: Generating Random Words samples...")
        random_words_list = random_words_sampling(nsamples=args.nsamples, tokenizer=tokenizer, sentence_length=128)

        # 4. Shuffled Zipf Sampling (Structure Hypothesis)
        log.info(f"{d_name}: Generating Shuffled Zipf samples...")
        shuffled_zipf_list = generate_shuffled_zipf_samples(
            tokenizer, nsamples=args.nsamples, seq_len=128,
            model=st_model, dataloader=current_tokenized_data, dataset_name=d_name
        )
        
        log.info(f"{d_name}: Computing stats for dataset words")
        datasets_words_list = prepare_calibration(
            model=st_model,
            dataloader=current_tokenized_data,
            nsamples=args.nsamples,
            type="words_dataset",
            tokenizer=tokenizer,
            dataset_name=d_name,
            model_name=args.model.replace("/", "-")
        )

        # 5. Get Statistics
        log.info(f"{d_name}: Computing stats for Random...")
        stats_rand = get_layer_stats(model, rand_list, n_layers=3, device=device)
        
        log.info(f"{d_name}: Computing stats for {args.zipf_type}...")
        stats_zipf = get_layer_stats(model, zipf_list, n_layers=3, device=device)

        log.info(f"{d_name}: Computing stats for Random Words...")
        stats_random_words = get_layer_stats(model, random_words_list, n_layers=3, device=device)
        
        log.info(f"{d_name}: Computing stats for Shuffled Zipf...")
        stats_shuffled = get_layer_stats(model, shuffled_zipf_list, n_layers=3, device=device)
        
        log.info(f"{d_name}: Computing stats for dataset words")
        stats_dataset_words = get_layer_stats(model, datasets_words_list, n_layers=3, device=device)

        # 6. Compare Pairs
        # We want to compare: 
        #  - Random vs Zipf
        #  - Random Words vs Zipf
        #  - Zipf vs Shuffled Zipf (Does structure matter?)
        
        comparisons = [
            ("random", stats_rand, "zipf", stats_zipf),
            ("random_words", stats_random_words, "zipf", stats_zipf),
            ("shuffled_zipf", stats_shuffled, "zipf", stats_zipf),
            ("random", stats_rand, "dataset_words", stats_dataset_words),
            ("random_words", stats_random_words, "dataset_words", stats_dataset_words)
        ]

        for method1_name, stats1, method2_name, stats2 in comparisons:
            for layer_name in stats1.keys():
                if layer_name in stats2:
                    layer_stats1 = stats1[layer_name]
                    layer_stats2 = stats2[layer_name]
                    
                    for metric_name in layer_stats1.keys():
                        data1 = layer_stats1[metric_name].numpy()
                        data2 = layer_stats2[metric_name].numpy()
                        
                        # KS Test
                        ks_stat, ks_pval = ks_2samp(data1, data2)
                        
                        # Descriptive Stats
                        mean1 = np.mean(data1)
                        mean2 = np.mean(data2)
                        mean_diff = mean2 - mean1
                        
                        max1 = np.max(data1)
                        max2 = np.max(data2)
                        max_diff = max2 - max1
                        
                        min1 = np.min(data1)
                        min2 = np.min(data2)
                        
                        skew1 = skew(data1)
                        skew2 = skew(data2)
                        kurt1 = kurtosis(data1)
                        kurt2 = kurtosis(data2)
                        
                        res_entry = {
                            "model": args.model,
                            "dataset": d_name,
                            "layer": layer_name,
                            "metric": metric_name,
                            "method_1": method1_name,
                            "method_2": method2_name,
                            "ks_statistic": ks_stat,
                            "ks_pvalue": ks_pval,
                            "mean_1": mean1,
                            "mean_2": mean2,
                            "mean_diff": mean_diff,
                            "max_1": max1,
                            "max_2": max2,
                            "min_1": min1,
                            "min_2": min2,
                            "skew_1": skew1,
                            "skew_2": skew2,
                            "kurtosis_1": kurt1,
                            "kurtosis_2": kurt2
                        }
                        results_list.append(res_entry)
                        
                        log.info(f"[{d_name}][{layer_name}][{metric_name}][{method1_name} vs {method2_name}] KS p={ks_pval:.2e}, Mean Diff={mean_diff:.2e}")
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(results_list)
    with FileLock(f"{args.output_csv}.lock"):   
        df.to_csv(args.output_csv,
                mode='a',
                header= not os.path.exists(args.output_csv),
                index=False)
    log.info(f"Detailed results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
