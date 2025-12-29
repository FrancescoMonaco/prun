from prune import get_tokenized_data
from similarity_check import prepare_calibration, embedd_data
from scipy.stats import wasserstein_distance
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import torch.nn.functional as F

from data import get_dataset

def compute_coverage_metrics(original_embeddings, sample_embeddings):
    """
    Computes metrics to evaluate how well the sample_embeddings cover the original_embeddings.
    Uses Cosine Distance and processes in batches to avoid OOM.
    """
    N = original_embeddings.shape[0]
    M = sample_embeddings.shape[0]
    
    # Flatten if 3D (batch, seq_len, embed_dim)
    if original_embeddings.dim() == 3:
        original_embeddings = original_embeddings.view(N, -1)
    if sample_embeddings.dim() == 3:
        sample_embeddings = sample_embeddings.view(M, -1)
        
    # Normalize sample embeddings once
    sample_norm = F.normalize(sample_embeddings.to(torch.float32), p=2, dim=-1).to("cuda")
    
    min_dists = []
    batch_size = 100
    for i in range(0, N, batch_size):
        batch = original_embeddings[i:i+batch_size].to(torch.float32).to("cuda")
        batch_norm = F.normalize(batch, p=2, dim=-1)
        
        # Cosine similarity (batch_size, M)
        cos_sim = torch.matmul(batch_norm, sample_norm.t())
        # Cosine distance = 1 - similarity
        cos_dist = 1 - cos_sim
        
        min_dist, _ = torch.min(cos_dist, dim=1)
        min_dists.append(min_dist.cpu())
        
    min_dists = torch.cat(min_dists)
    
    mnnd = torch.mean(min_dists).item()
    maxnnd = torch.max(min_dists).item()
    
    # Centroid distance
    orig_centroid = torch.mean(original_embeddings.to(torch.float32), dim=0, keepdim=True)
    sample_centroid = torch.mean(sample_embeddings.to(torch.float32), dim=0, keepdim=True)
    
    orig_centroid_norm = F.normalize(orig_centroid, p=2, dim=-1).to("cuda")
    sample_centroid_norm = F.normalize(sample_centroid, p=2, dim=-1).to("cuda")
    
    centroid_dist = (1 - F.cosine_similarity(orig_centroid_norm, sample_centroid_norm)).item()
    
    return mnnd, maxnnd, centroid_dist

if __name__ == "__main__":
    datasets = ["winogrande", "svamp", "boolq", "commonsense_qa", "race"]
    multiple_datasets = [["winogrande", "svamp"], ["boolq", "commonsense_qa"]]
    model_name = "Qwen/Qwen3-1.7B"
    sentence_transformer_model_name = (
        "google/embeddinggemma-300m"  # "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    with open("results/sampling_analysis.csv", "a+") as f:
        f.write(
            "dataset,pruning_type,wasserstein,mnnd,maxnnd,centroid_dist\n"
        )

        for dataset_name in datasets:
            # f.write(f"{dataset_name},") # Removed this as we write per pruning type now
            raw_dataset = get_dataset(dataset_name)
            if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
                if "train" in raw_dataset.keys():
                    dataset = raw_dataset["train"]
                elif "test" in raw_dataset.keys():
                    dataset = raw_dataset["test"]
                else:
                    dataset = raw_dataset[list(raw_dataset.keys())[0]]
            else:
                dataset = raw_dataset
            # Truncate data
            # if len(dataset) > 5000:
            #     dataset = dataset.select(range(5000))
            tokenized_data = get_tokenized_data(
                dataset, tokenizer, dataset_name, return_tensors=True
            )

            # Compute original distribution (Distances to Centroid)
            print(f"Computing original embeddings for {dataset_name}...", flush=True)
            original_embeddings_model = embedd_data(
                tokenized_data, model, device="cuda"
            )
            
            # original_sentence_embeddings = embedd_data(
            #     tokenized_data,
            #     SentenceTransformer(sentence_transformer_model_name, device="cuda"),
            #     device="cuda",
            # )
            
            original_dists = (
                torch.norm(original_embeddings_model, dim=-1)
                .view(-1)
                .cpu()
                .float()
                .numpy()
            )

            for pruning_type in [
                "most_similar",
                "decoupled",
                "most_dissimilar",
                "least_perplexity",
                "random",
                "herding",
                "distribution_matching",
            ]:
                # Map pruning_type to prepare_calibration type
                calibration_type = "prototype"
                if pruning_type == "most_similar":
                    calibration_type = "prototype"
                elif pruning_type == "most_dissimilar":
                    calibration_type = "most_different"
                elif pruning_type == "decoupled":
                    calibration_type = "decoupled"
                elif pruning_type == "least_perplexity":
                    calibration_type = "least_perplexity"
                elif pruning_type == "random":
                    calibration_type = "random_sample"
                elif pruning_type == "herding":
                    calibration_type = "herding"
                elif pruning_type == "distribution_matching":
                    calibration_type = "distribution_matching"

                # Prepare the calibration sample
                calibration = prepare_calibration(
                    model=model,
                    dataloader=[tokenized_data],
                    nsamples=128,
                    type=calibration_type,
                    distance="flatten",
                    save_calibration_distribution=False,
                    model_name=model_name.replace("/", "-"),
                    dataset_name=dataset_name,
                    tokenizer=tokenizer,
                    return_distribution=False,
                )

                # Compute sample distribution
                sample_embeddings = embedd_data(calibration, model, device="cuda")
                
                # Compare the distribution of the sample and the original dataset (Wasserstein on norms)
                sample_dists = (
                    torch.norm(sample_embeddings, dim=-1).view(-1).cpu().float().numpy()
                )
                w_distance = wasserstein_distance(original_dists, sample_dists)
                
                # Compute coverage metrics
                mnnd, maxnnd, c_dist = compute_coverage_metrics(original_embeddings_model, sample_embeddings)

                f.write(f"{dataset_name},{pruning_type},{w_distance},{mnnd},{maxnnd},{c_dist}\n")
                f.flush()
            # Removed f.write("\n") as we write per line now

        for dataset_group in multiple_datasets:
            dataset_names = dataset_group
            group_name = '_'.join(dataset_names)
            combined_tokenized_data = []
            for dataset_name in dataset_names:
                raw_dataset = get_dataset(dataset_name)
                if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
                    if "train" in raw_dataset.keys():
                        dataset = raw_dataset["train"]
                    elif "test" in raw_dataset.keys():
                        dataset = raw_dataset["test"]
                    else:
                        dataset = raw_dataset[list(raw_dataset.keys())[0]]
                else:
                    dataset = raw_dataset
                # Truncate data
                if len(dataset) > 5000:
                    dataset = dataset.select(range(5000))
                tokenized_data = get_tokenized_data(
                    dataset, tokenizer, dataset_name, return_tensors=True
                )
                combined_tokenized_data.extend(tokenized_data)
            # Compute original distribution (Distances to Centroid)
            print(
                f"Computing original embeddings for {group_name}...",
                flush=True,
            )
            original_embeddings = embedd_data(
                combined_tokenized_data, model, device="cuda"
            )
            
            original_dists = (
                torch.norm(original_embeddings, dim=-1).view(-1).cpu().float().numpy()
            )

            for pruning_type in [
                "most_similar",
                "decoupled",
                "most_dissimilar",
                "least_perplexity",
                "random",
                "herding",
                "distribution_matching",
            ]:
                # Map pruning_type to prepare_calibration type
                calibration_type = "prototype"
                if pruning_type == "most_similar":
                    calibration_type = "prototype"
                elif pruning_type == "most_dissimilar":
                    calibration_type = "most_different"
                elif pruning_type == "decoupled":
                    calibration_type = "decoupled"
                elif pruning_type == "least_perplexity":
                    calibration_type = "least_perplexity"
                elif pruning_type == "random":
                    calibration_type = "random_sample"
                elif pruning_type == "herding":
                    calibration_type = "herding"
                elif pruning_type == "distribution_matching":
                    calibration_type = "distribution_matching"

                # Prepare the calibration sample
                calibration = prepare_calibration(
                    model=model,
                    dataloader=[combined_tokenized_data],
                    nsamples=128,
                    type=calibration_type,
                    distance="flatten",
                    save_calibration_distribution=False,
                    model_name=model_name.replace("/", "-"),
                    dataset_name=group_name,
                    tokenizer=tokenizer,
                    return_distribution=False,
                )
                # Compute sample distribution
                sample_embeddings = embedd_data(calibration, model, device="cuda")
                
                # Compare the distribution of the sample and the original dataset (Wasserstein on norms)
                sample_dists = (
                    torch.norm(sample_embeddings, dim=-1).view(-1).cpu().float().numpy()
                )
                w_distance = wasserstein_distance(original_dists, sample_dists)
                
                # Compute coverage metrics
                mnnd, maxnnd, c_dist = compute_coverage_metrics(original_embeddings, sample_embeddings)

                f.write(f"{group_name},{pruning_type},{w_distance},{mnnd},{maxnnd},{c_dist}\n")
                f.flush()
