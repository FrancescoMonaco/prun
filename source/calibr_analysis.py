import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from prune import get_tokenized_data
from similarity_check import prepare_calibration, embedd_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import get_dataset
import os

def compute_mmd(x, y, gamma=None):
    """
    Calcola il Maximum Mean Discrepancy tra due set di embedding usando un kernel RBF.
    Più è basso il valore, più le distribuzioni sono simili.
    """
    if gamma is None:
        gamma = 1.0 / x.size(-1)
    
    def rbf_kernel(A, B):
        # Calcola le distanze euclidee al quadrato (N, M)
        dist = torch.cdist(A, B).pow(2)
        return torch.exp(-gamma * dist)

    # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]
    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)

    return (k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()).item()

def analyze_embedding_density(original_emb, sample_emb, k=10):
    """
    Analizza come i campioni si posizionano rispetto alla densità della popolazione originale.
    Restituisce statistiche sulla distanza dai k-vicini originali.
    """
    # Distanza di ogni punto del campione da tutti i punti originali (M_sample, N_original)
    dists = torch.cdist(sample_emb, original_emb)
    
    # k-vicini più prossimi nell'originale per ogni punto del campione
    k_dists, _ = torch.topk(dists, k=k, largest=False, dim=1)
    
    # La distanza media dai vicini indica se il punto è in una zona ad alta o bassa densità
    avg_k_dist = k_dists.mean(dim=1)
    
    return {
        "nn_dist_mean": avg_k_dist.mean().item(),
        "nn_dist_std": avg_k_dist.std().item(),
        "nn_dist_median": avg_k_dist.median().item(),
        "nn_dist_max": avg_k_dist.max().item(),
        "nn_dist_min": avg_k_dist.min().item()
    }

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-1.7B"
    datasets_list = ["winogrande", "svamp", "boolq", "commonsense_qa", "race"]
    sampling_techniques = ["random", "most_similar", "distribution_matching", "zipf"]
    nsamples = 128

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    results = []

    if not os.path.exists("results"):
        os.makedirs("results")

    for ds_name in datasets_list:
        print(f"\n--- Analyzing Dataset: {ds_name} ---")
        try:
            raw_dataset = get_dataset(ds_name)
        except Exception as e:
            print(f"Error loading dataset {ds_name}: {e}")
            continue
        
        # Gestione split dataset (train o test o il primo disponibile)
        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            if "train" in raw_dataset.keys():
                dataset = raw_dataset["train"]
            elif "test" in raw_dataset.keys():
                dataset = raw_dataset["test"]
            else:
                dataset = raw_dataset[list(raw_dataset.keys())[0]]
        else:
            dataset = raw_dataset
            
        print(f"Tokenizing {ds_name}...")
        tokenized_full = get_tokenized_data(dataset, tokenizer, ds_name, return_tensors=True)
        
        pop_subset = tokenized_full

        print(f"Computing embeddings for original population ({len(pop_subset)} samples)...")
        original_embeddings = embedd_data(pop_subset, model, device="cuda")
        if original_embeddings.dim() == 3:
            original_embeddings = original_embeddings.mean(dim=1)
        # Normalizzazione per coerenza
        original_embeddings = F.normalize(original_embeddings.to(torch.float32), p=2, dim=1)

        for tech in sampling_techniques:
            print(f"Sampling with technique: {tech}...")
            
            # Mapping nomi tecniche verso prepare_calibration
            calib_type = tech
            if tech == "random": calib_type = "random_sample"
            if tech == "most_similar": calib_type = "prototype"

            try:
                calibration_samples = prepare_calibration(
                    model=model,
                    dataloader=[tokenized_full],
                    nsamples=nsamples,
                    type=calib_type,
                    tokenizer=tokenizer,
                    dataset_name=ds_name
                )

                sample_embeddings = embedd_data(calibration_samples, model, device="cuda")
                if sample_embeddings.dim() == 3:
                    sample_embeddings = sample_embeddings.mean(dim=1)
                sample_embeddings = F.normalize(sample_embeddings.to(torch.float32), p=2, dim=1)

                # 1. Calcolo MMD
                mmd_val = compute_mmd(original_embeddings, sample_embeddings)
                
                # 2. Analisi della Densità Locale (Distanza dai vicini originali)
                density_stats = analyze_embedding_density(original_embeddings, sample_embeddings)

                res = {
                    "dataset": ds_name,
                    "technique": tech,
                    "mmd": mmd_val,
                    **density_stats
                }
                results.append(res)
                print(f"[{tech}] MMD: {mmd_val:.6f} | Avg_NN_Dist: {density_stats['nn_dist_mean']:.4f}")
            except Exception as e:
                print(f"Error processing {tech} on {ds_name}: {e}")

    # Salvataggio risultati
    df = pd.DataFrame(results)
    output_path = "results/calibration_embedding_distribution.csv"
    df.to_csv(output_path, index=False)
    print(f"\nAnalysis complete. Results saved to {output_path}")
