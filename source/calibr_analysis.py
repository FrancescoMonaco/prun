import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from prune import get_tokenized_data
from similarity_check import prepare_calibration, embedd_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from data import get_dataset
import os
import argparse

def get_llm_metrics(dataloader, model, device="cuda", top_k=5):
    model.eval()
    all_activations = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            else:
                input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
                attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)
            
            if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1: attention_mask = attention_mask.unsqueeze(0)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # 1. Activations (Hidden States)
            hidden = outputs.hidden_states[-1]
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_embeddings = torch.sum(hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            all_activations.append(mean_pooled.cpu())

            # 2. Confidence (Sum of Top-K Softmax Probabilities)
            logits = outputs.logits[:, :-1, :]
            mask_shift = attention_mask[:, 1:]
            
            probs = F.softmax(logits, dim=-1)
            # Consider the sum of top-k probabilities to better reflect model certainty 
            # in a beam-search-like evaluation.
            top_k_probs, _ = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
            sum_top_k = top_k_probs.sum(dim=-1) # [batch, seq-1]
            
            # Average confidence per sequence
            seq_conf = (sum_top_k * mask_shift).sum(dim=1) / torch.clamp(mask_shift.sum(dim=1), min=1)
            all_confidences.append(seq_conf.cpu())
            
    return torch.cat(all_activations, dim=0), torch.cat(all_confidences, dim=0)

def plot_enhanced_distributions(dist_dict, original_norms, dataset_name, output_dir, type_name="Activations"):
     
     
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. KDE Plot delle Norme
    def safe_kde(data, label, color=None, linestyle="-", ax=None, fill=False, alpha=0.2):
        if np.var(data) < 1e-9:
            sns.kdeplot(data + np.random.normal(0, 1e-6, size=len(data)), label=f"{label} (Zero Var)", color=color, linestyle=linestyle, ax=ax, fill=fill, alpha=alpha, warn_singular=False)
        else:
            sns.kdeplot(data, label=label, color=color, linestyle=linestyle, ax=ax, fill=fill, alpha=alpha)

    safe_kde(original_norms, "Full Population", color="black", linestyle="--", ax=ax1)
    for tech, norms in dist_dict.items():
        safe_kde(norms, tech, ax=ax1, fill=True)
    
    ax1.set_title(f"Impact of Sampling on {type_name} Norms - {dataset_name}", fontsize=15)
    ax1.set_xlabel(f"L2 Norm of {type_name}")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative Distribution (CDF) - Mostra dove le tecniche "pescano" (code vs centro)
    for tech, norms in dist_dict.items():
        sns.ecdfplot(norms, label=tech, ax=ax2)
    sns.ecdfplot(original_norms, color="black", linestyle="--", ax=ax2)
    
    ax2.set_title(f"Cumulative Distribution of {type_name} (Tail Coverage)", fontsize=12)
    ax2.set_xlabel("L2 Norm")
    ax2.set_ylabel("Cumulative Prob.")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{type_name.lower()}_dist_{dataset_name}.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Calibration Analysis")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="LLM to use for activations")
    parser.add_argument("--st_model_name", type=str, default="all-MiniLM-L12-v2", help="ST model used for sampling")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K probabilities to sum for confidence")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = args.model_name
    st_model_name = args.st_model_name
    model_id = model_name.replace("/", "_")
    datasets_list = ["winogrande", "svamp", "boolq", "commonsense_qa", "race", "arc_challenge"]
    sampling_techniques = ["most_similar", "distribution_matching", "zipf", "unique_tokens", "random_sample"]
    nsamples = args.nsamples

    print(f"Loading LLM {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Sentence Transformer {st_model_name}...")
    st_model = SentenceTransformer(st_model_name, device=device)

    results = []
    plot_dir = f"plots/calibration_analysis/{model_id}"
    results_dir = f"results/calibration_analysis/{model_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    for ds_name in datasets_list:
        print(f"\n--- Deep Analysis: {ds_name} ---")
        raw_dataset = get_dataset(ds_name)
        if raw_dataset is None: continue
        
        dataset = raw_dataset["train"] if "train" in raw_dataset else (raw_dataset["test"] if "test" in raw_dataset else raw_dataset[list(raw_dataset.keys())[0]])
        
        # Sottocampionamento popolazione per confronto
        pop_indices = np.random.choice(len(dataset), min(2000, len(dataset)), replace=False)
        pop_subset = dataset.select(pop_indices)
        tokenized_pop = get_tokenized_data(pop_subset, tokenizer, ds_name, return_tensors=True)
        
        # 1. Calcolo norme LLM (Activations & Confidence)
        print("Computing LLM metrics (forward pass)...")
        pop_loader = torch.utils.data.DataLoader(tokenized_pop, batch_size=16)
        pop_act_emb, pop_conf = get_llm_metrics(pop_loader, model, device=device, top_k=args.top_k)
        pop_act_norms = torch.norm(pop_act_emb.float(), p=2, dim=1).cpu().numpy()
        pop_conf_vals = pop_conf.cpu().numpy()

        # 2. Calcolo norme ST (Embeddings)
        print("Computing ST embedding norms...")
        pop_st_emb = embedd_data(tokenized_pop, st_model, device=device)
        pop_st_norms = torch.norm(pop_st_emb.float(), p=2, dim=1).cpu().numpy()

        act_norms_to_plot = {}
        st_norms_to_plot = {}
        conf_to_plot = {}

        # Baseline Random
        print("Sampling: random...")
        random_samples = prepare_calibration(model=model, dataloader=[tokenized_pop], nsamples=nsamples, type="random_sample", tokenizer=tokenizer, dataset_name=ds_name)
        random_loader = torch.utils.data.DataLoader(random_samples, batch_size=16)
                
        # Random LLM Metrics
        random_act_emb, random_conf = get_llm_metrics(random_loader, model, device=device, top_k=args.top_k)
        random_act_norms = torch.norm(random_act_emb.float(), p=2, dim=1).cpu().numpy()
        random_conf_vals = random_conf.cpu().numpy()
        
        act_norms_to_plot["random"] = random_act_norms
        conf_to_plot["random"] = random_conf_vals

        # Random ST
        random_st_emb = embedd_data(random_samples, st_model, device=device)
        random_st_norms = torch.norm(random_st_emb.float(), p=2, dim=1).cpu().numpy()
        st_norms_to_plot["random"] = random_st_norms

        for tech in sampling_techniques:
            print(f"Sampling: {tech}...")
            calib_type = "prototype" if tech == "most_similar" else tech
            try:
                tech_samples = prepare_calibration(model=model, dataloader=[tokenized_pop], nsamples=nsamples, type=calib_type, tokenizer=tokenizer, dataset_name=ds_name)
                tech_loader = torch.utils.data.DataLoader(tech_samples, batch_size=16)
                
                # Tech LLM Metrics
                tech_act_emb, tech_conf = get_llm_metrics(tech_loader, model, device=device, top_k=args.top_k)
                tech_act_norms = torch.norm(tech_act_emb.float(), p=2, dim=1).cpu().numpy()
                tech_conf_vals = tech_conf.cpu().numpy()
                
                act_norms_to_plot[tech] = tech_act_norms
                conf_to_plot[tech] = tech_conf_vals

                # Tech ST
                tech_st_emb = embedd_data(tech_samples, st_model, device=device)
                tech_st_norms = torch.norm(tech_st_emb.float(), p=2, dim=1).cpu().numpy()
                st_norms_to_plot[tech] = tech_st_norms

                # Wasserstein vs Random
                wd_act = wasserstein_distance(random_act_norms, tech_act_norms)
                wd_st = wasserstein_distance(random_st_norms, tech_st_norms)
                wd_conf = wasserstein_distance(random_conf_vals, tech_conf_vals)
                
                results.append({
                    "dataset": ds_name, 
                    "technique": tech, 
                    "wd_act_vs_random": wd_act, 
                    "wd_st_vs_random": wd_st,
                    "wd_conf_vs_random": wd_conf,
                    "mean_act_norm": np.mean(tech_act_norms),
                    "mean_st_norm": np.mean(tech_st_norms),
                    "mean_confidence": np.mean(tech_conf_vals)
                })
                print(f"[{tech}] WD-Act: {wd_act:.4f} | Conf: {np.mean(tech_conf_vals):.4f}")
            except Exception as e:
                print(f"Error with {tech}: {e}")

        # Plot All
        plot_enhanced_distributions(act_norms_to_plot, pop_act_norms, ds_name, plot_dir, type_name="Activations")
        plot_enhanced_distributions(st_norms_to_plot, pop_st_norms, ds_name, plot_dir, type_name="Embeddings")
        plot_enhanced_distributions(conf_to_plot, pop_conf_vals, ds_name, plot_dir, type_name="Confidence")

    results_path = os.path.join(results_dir, "informed_norms.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nDone. Results saved in {results_path}")
    print(f"Plots with CDF and KDE in {plot_dir}")
