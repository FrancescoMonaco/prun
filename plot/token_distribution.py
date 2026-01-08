import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "source"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "COLA"))
try:
    from data import get_dataset, get_text_from_item
    from similarity_check import prepare_calibration
    from cola.sample_selection import select_samples
except ImportError as e:
        print(f"Could not import modules, {e}")
        sys.exit(1)

def get_token_counts(texts, tokenizer, max_length=128):
    counts = Counter()
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        # Filter out pad tokens if needed, but here we want the distribution of actual tokens
        # Filter out special tokens for a cleaner distribution if desired
        counts.update(tokens)
    return counts

def plot_token_distribution(all_counts, labels, output_path, tokenizer, top_n=30):
    plt.figure(figsize=(15, 8))
    
    # We want to compare the frequency (percentage) to make it fair
    data = []
    
    # Use the union of top tokens from the FULL DATASET to compare subsets against it
    full_counts = all_counts[0]
    top_tokens = [t for t, c in full_counts.most_common(top_n)]
    
    for counts, label in zip(all_counts, labels):
        total = sum(counts.values())
        for token_id in top_tokens:
            freq = (counts[token_id] / total) * 100 if total > 0 else 0
            token_text = f"'{tokenizer.decode([token_id])}' ({token_id})"
            # Escape some special characters for plot
            token_text = token_text.replace("\n", "\\n").replace("\r", "\\r")
            data.append({
                "Token": token_text,
                "Frequency (%)": freq,
                "Method": label
            })
    
    df = pd.DataFrame(data)
    
    sns.barplot(data=df, x="Token", y="Frequency (%)", hue="Method")
    plt.title(f"Token Distribution Comparison (Top {top_n} Tokens of Full Dataset)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Token Distribution to Show the Distribution")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to plot token distribution for",
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
        help="Number of samples for subsets",
    )
    parser.add_argument(
        "--pruning_types",
        nargs="+",
        default=["cola", "random", "most_similar", "distribution_matching", "perplexity"],
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
    
    # Load tokenizer and model
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model only if needed for selection methods
    model = None
    if any(p in args.pruning_types for p in ["cola", "most_similar", "most_dissimilar", "decoupled", "least_perplexity"]):
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )

    # 1. Load full dataset
    print(f"Loading dataset: {args.dataset}")
    raw_dataset = get_dataset(args.dataset)
    if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
        dataset = raw_dataset.get("train") or raw_dataset.get("test") or raw_dataset[list(raw_dataset.keys())[0]]
    else:
        dataset = raw_dataset
        
    all_texts = [get_text_from_item(item, args.dataset) for item in dataset]
    # Limit full dataset for faster tokenization if it's too large
    if len(all_texts) > 5000:
        print(f"Limiting full dataset Analysis to 5000 samples for speed.")
        all_texts_limited = all_texts[:5000]
    else:
        all_texts_limited = all_texts
        
    # 2. Get full dataset token counts
    print("Calculating full dataset token distribution...")
    full_counts = get_token_counts(all_texts_limited, tokenizer)
    
    all_distributions = [full_counts]
    labels = ["Full Dataset"]
    
    # 3. For each pruning type, get calibration data and count tokens
    
    # Prepare tokenized and candidates for selection methods
    all_tokenized_data = []
    print("Preparing tokenized data for selection methods...")
    num_candidates = min(len(all_texts), 4080)
    for text in tqdm(all_texts[:num_candidates], desc="Tokenizing candidates"): # Limit candidates as in eval_cola.py
        encoded = tokenizer(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        all_tokenized_data.append({
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        })
    
    calibration_type_map = {
        "most_similar": "prototype",
        "most_dissimilar": "most_different",
        "decoupled": "decoupled",
        "least_perplexity": "perplexity",
        "herding": "herding",
        "distribution_matching": "distribution_matching",
    }

    for p_type in args.pruning_types:
        print(f"\nProcessing selection method: {p_type}")
        
        if p_type == "cola":
            processed_samples = [{"text": t} for t in all_texts[:num_candidates]] 
            selected_samples = select_samples(
                processed_samples, 
                model, 
                tokenizer, 
                num_clusters=args.nsamples,
                device=device,
                batch_size=4 
            )
            selected_texts = [s["text"] for s in selected_samples]
            counts = get_token_counts(selected_texts, tokenizer)
            all_distributions.append(counts)
            labels.append(p_type)
            
        elif p_type == "random":
            import random
            indices = random.sample(range(len(all_texts)), min(args.nsamples, len(all_texts)))
            selected_texts = [all_texts[i] for i in indices]
            counts = get_token_counts(selected_texts, tokenizer)
            all_distributions.append(counts)
            labels.append(p_type)
            
        elif p_type in calibration_type_map:
            method = calibration_type_map[p_type]
            calib_data = prepare_calibration(
                all_tokenized_data, 
                args.nsamples, 
                method=method,
                model_name=args.model
            )
            counts = Counter()
            for item in calib_data:
                # Filter out padding tokens for counts
                input_ids = item["input_ids"].tolist()
                if tokenizer.pad_token_id is not None:
                    input_ids = [tid for tid in input_ids if tid != tokenizer.pad_token_id]
                counts.update(input_ids)
            all_distributions.append(counts)
            labels.append(p_type)
        else:
            print(f"Unknown pruning type: {p_type}")
            
    # 4. Plot
    output_path = os.path.join(args.output_dir, f"{args.dataset}_token_dist.png")
    plot_token_distribution(all_distributions, labels, output_path, tokenizer)
