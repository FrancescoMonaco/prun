from prune import get_tokenized_data
from similarity_check import prepare_calibration, embedd_data
from scipy.stats import wasserstein_distance
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

from data import get_dataset

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
            "dataset,most_similar,decoupled,most_dissimilar,least_perplexity,random\n"
        )

        for dataset_name in datasets:
            f.write(f"{dataset_name},")
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

            # Compute original distribution (Distances to Centroid)
            print(f"Computing original embeddings for {dataset_name}...", flush=True)
            original_embeddings_model = embedd_data(
                tokenized_data, model, device="cuda"
            )
            original_sentence_embeddings = embedd_data(
                tokenized_data,
                SentenceTransformer(sentence_transformer_model_name, device="cuda"),
                device="cuda",
            )
            # centroid = torch.mean(original_embeddings, dim=0)
            # # Cosine distance = 1 - cosine similarity
            # original_dists = 1 - torch.nn.functional.cosine_similarity(original_embeddings, centroid.unsqueeze(0))
            # original_dists = original_dists.cpu().numpy()
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

                # Prepare the calibration sample
                calibration = prepare_calibration(
                    model=model,
                    dataloader=[tokenized_data],
                    nsamples=256,
                    type=calibration_type,
                    distance="flatten",
                    save_calibration_distribution=False,
                    model_name=model_name.replace("/", "-"),
                    dataset_name=dataset_name,
                    tokenizer=tokenizer,
                    return_distribution=False,
                )

                # Compute sample distribution (Distances to Centroid)
                sample_embeddings = embedd_data(calibration, model, device="cuda")
                # sample_dists = 1 - torch.nn.functional.cosine_similarity(sample_embeddings, centroid.unsqueeze(0))
                # sample_dists = sample_dists.cpu().numpy()
                # # Remove from both distributions the percentile extremes to avoid outliers
                # lower_percentile = 1
                # upper_percentile = 99
                # lower_orig = np.percentile(original_dists, lower_percentile)
                # upper_orig = np.percentile(original_dists, upper_percentile)
                # lower_sample = np.percentile(sample_dists, lower_percentile)
                # upper_sample = np.percentile(sample_dists, upper_percentile)
                # original_dists = original_dists[(original_dists >= lower_orig) & (original_dists <= upper_orig)]
                # sample_dists = sample_dists[(sample_dists >= lower_sample) & (sample_dists <= upper_sample)]
                # Compare the distribution of the sample and the original dataset
                sample_dists = (
                    torch.norm(sample_embeddings, dim=-1).view(-1).cpu().float().numpy()
                )
                distance = wasserstein_distance(original_dists, sample_dists)

                f.write(f"{distance},")
                f.flush()
            # Close the line
            f.write("\n")

        for dataset_group in multiple_datasets:
            dataset_names = dataset_group
            f.write(f"{'_'.join(dataset_names)},")
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
                f"Computing original embeddings for {'_'.join(dataset_names)}...",
                flush=True,
            )
            original_embeddings = embedd_data(
                combined_tokenized_data, model, device="cuda"
            )
            # centroid = torch.mean(original_embeddings, dim=0)
            # # Cosine distance = 1 - cosine similarity
            # original_dists = 1 - torch.nn.functional.cosine_similarity(original_embeddings, centroid.unsqueeze(0))
            # original_dists = original_dists.cpu().numpy()
            original_dists = (
                torch.norm(original_embeddings, dim=-1).view(-1).cpu().float().numpy()
            )

            for pruning_type in [
                "most_similar",
                "decoupled",
                "most_dissimilar",
                "least_perplexity",
                "random",
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

                # Prepare the calibration sample
                calibration = prepare_calibration(
                    model=model,
                    dataloader=[combined_tokenized_data],
                    nsamples=128,
                    type=calibration_type,
                    distance="flatten",
                    save_calibration_distribution=False,
                    model_name=model_name.replace("/", "-"),
                    dataset_name="_".join(dataset_names),
                    tokenizer=tokenizer,
                    return_distribution=False,
                )
                # !!!Compute sample distribution (Distances to Centroid)
                sample_embeddings = embedd_data(calibration, model, device="cuda")
                # sample_dists = 1 - torch.nn.functional.cosine_similarity(sample_embeddings, centroid.unsqueeze(0))
                # sample_dists = sample_dists.cpu().numpy()
                # # Remove from both distributions the percentile extremes to avoid outliers
                # lower_percentile = 1
                # upper_percentile = 99
                # lower_orig = np.percentile(original_dists, lower_percentile)
                # upper_orig = np.percentile(original_dists, upper_percentile)
                # lower_sample = np.percentile(sample_dists, lower_percentile)
                # upper_sample = np.percentile(sample_dists, upper_percentile)
                # original_dists = original_dists[(original_dists >= lower_orig) & (original_dists <= upper_orig)]
                # sample_dists = sample_dists[(sample_dists >= lower_sample) & (sample_dists <= upper_sample)]
                # Compare the distribution of the sample and the original dataset

                # !!!Distibutions of norms
                sample_dists = (
                    torch.norm(sample_embeddings, dim=-1).view(-1).cpu().float().numpy()
                )
                # original_dists = torch.norm(original_embeddings, dim=1).cpu().numpy()
                distance = wasserstein_distance(original_dists, sample_dists)
                f.write(f"{distance},")
                f.flush()
            # Close the line
            f.write("\n")
