import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import nltk
from scipy.stats import entropy
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
        # Lowercase the text to treat 'The' and 'the' the same
        text = text.lower()
        tokens = tokenizer.tokenize(text, truncation=True, max_length=max_length)
        
        # Filter out special tokens for a cleaner distribution
        tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]        
        
        # Standardize tokens by removing subword markers (e.g., ' ' for Gemma/Llama or 'Ġ' for GPT-style)
        # and stripping whitespace. This ensures ' the' and 'the' are counted together.
        processed_tokens = []
        for t in tokens:
            clean_t = t.replace(' ', '').replace('Ġ', '').strip()
            if clean_t:
                processed_tokens.append(clean_t)
        
        counts.update(processed_tokens)
    return counts

def get_pos_counts(texts):
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    
    counts = Counter()
    for text in tqdm(texts, desc="POS Tagging"):
        try:
            # Lowercase for consistent POS tagging frequency
            tokens = nltk.word_tokenize(text.lower())
            tags = nltk.pos_tag(tokens)
            # Map detailed tags to simpler categories if desired, or keep them
            # Common tags: NN (Noun), VB (Verb), JJ (Adjective), RB (Adverb)
            # We can group them:
            simplified_tags = []
            for word, tag in tags:
                if tag.startswith('NN'): simplified_tags.append('Noun')
                elif tag.startswith('VB'): simplified_tags.append('Verb')
                elif tag.startswith('JJ'): simplified_tags.append('Adj')
                elif tag.startswith('RB'): simplified_tags.append('Adv')
                elif tag.startswith('PRP'): simplified_tags.append('Pron')
                elif tag.startswith('IN'): simplified_tags.append('Prep')
                elif tag.startswith('DT'): simplified_tags.append('Det')
                else: simplified_tags.append('Other')
            counts.update(simplified_tags)
        except Exception as e:
            continue
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

def plot_pos_distribution(all_pos_counts, labels, output_path):
    plt.figure(figsize=(15, 8))
    data = []
    
    for counts, label in zip(all_pos_counts, labels):
        total = sum(counts.values())
        for pos, count in counts.items():
            freq = (count / total) * 100 if total > 0 else 0
            data.append({
                "POS Tag": pos,
                "Frequency (%)": freq,
                "Method": label
            })
    
    df = pd.DataFrame(data)
    sns.barplot(data=df, x="POS Tag", y="Frequency (%)", hue="Method")
    plt.title("Part-of-Speech Distribution Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"POS distribution plot saved to {output_path}")

def plot_token_distribution(all_counts, labels, output_path, tokenizer, top_n=30):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    # --- Subplot 1: Bar chart of top tokens ---
    data = []
    full_counts = all_counts[0]
    top_tokens = [t for t, c in full_counts.most_common(top_n)]
    
    for counts, label in zip(all_counts, labels):
        total = sum(counts.values())
        for token_text in top_tokens:
            freq = (counts[token_text] / total) * 100 if total > 0 else 0
            # Escape some special characters for plot
            cleaned_display = str(token_text).replace("\n", "\\n").replace("\r", "\\r")
            data.append({
                "Token": f"'{cleaned_display}'",
                "Frequency (%)": freq,
                "Method": label
            })
    
    df = pd.DataFrame(data)
    
    sns.barplot(data=df, x="Token", y="Frequency (%)", hue="Method", ax=ax1)
    ax1.set_title(f"Top {top_n} Tokens Frequency Comparison")
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlabel("Token")
        
    # --- Subplot 2: Zipf's Law Comparison (Log-Log) ---
    for counts, label in zip(all_counts, labels):
        freqs = sorted(counts.values(), reverse=True)
        if not freqs:
            continue
        total = sum(freqs)
        # Normalize to probability
        freqs_norm = np.array(freqs) / total
        ranks = np.arange(1, len(freqs_norm) + 1)
        ax2.plot(ranks, freqs_norm, label=label, linewidth=2, alpha=0.8)
    
    # Reference Zipf curve: f(r) = 1/r
    max_rank = max([len(c) for c in all_counts])
    if max_rank > 0:
        zipf_ranks = np.arange(1, max_rank + 1)
        zipf_vals = 1.0 / zipf_ranks
        zipf_vals /= zipf_vals.sum()
        ax2.plot(zipf_ranks, zipf_vals, "k--", label="Theoretical Zipf (s=1)", alpha=0.6)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Rank (log)")
    ax2.set_ylabel("Frequency (log)")
    ax2.set_title("Token Frequency vs Rank (Zipf's Law)")
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
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
        default=512,
        help="Number of samples for calibration data",
    )
    parser.add_argument(
        "--pruning_types",
        nargs="+",
        default=["random", "most_similar", "distribution_matching", "least_perplexity", "zipf", "unique_tokens", "words_dataset"],
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
    print("Calculating full dataset token and POS distribution...")
    full_counts = get_token_counts(all_texts_limited, tokenizer)
    full_pos_counts = get_pos_counts(all_texts_limited)
    
    all_distributions = [full_counts]
    all_pos_distributions = [full_pos_counts]
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
            pos_counts = get_pos_counts(selected_texts)
            all_distributions.append(counts)
            all_pos_distributions.append(pos_counts)
            labels.append(p_type)
            
        elif p_type in calibration_type_map:
            method = calibration_type_map[p_type]
            calib_data = prepare_calibration(
                model=model,
                dataloader=[all_tokenized_data],
                nsamples=args.nsamples,
                type=method,
                distance="flatten",
                model_name=args.model.replace("/", "_"),
                dataset_name=[args.dataset],
                tokenizer=tokenizer,
            )
            
            selected_texts = []
            for item in calib_data:
                input_ids = item["input_ids"].tolist()
                text = tokenizer.decode(input_ids, skip_special_tokens=True)
                selected_texts.append(text)
            
            counts = get_token_counts(selected_texts, tokenizer)
            pos_counts = get_pos_counts(selected_texts)
            all_distributions.append(counts)
            all_pos_distributions.append(pos_counts)
            labels.append(p_type)
        else:
            print(f"Unknown pruning type: {p_type}")
            
    # 4. Plot
    output_path = os.path.join(args.output_dir, f"{args.dataset}_token_dist.png")
    plot_token_distribution(all_distributions, labels, output_path, tokenizer, top_n=30)
    
    pos_output_path = os.path.join(args.output_dir, f"{args.dataset}_pos_dist.png")
    plot_pos_distribution(all_pos_distributions, labels, pos_output_path)

    # 5. Calculate and print KL Divergence for tokens
    print("\n--- KL Divergence (D_KL(Full || Subset)) for Token Distribution ---")
    full_dist = all_distributions[0]
    for i in range(1, len(labels)):
        kl_val = calculate_kl(full_dist, all_distributions[i])
        print(f"Method: {labels[i]:25} | KL Divergence: {kl_val:.6f}")
