"""
Part of the code is adapted from: https://github.com/muraronicola/Muraro-Nicola-Master-Thesis
"""

import os
import torch
import torch.nn.functional as F
from joblib import Memory

# import wandb
from sentence_transformers import SentenceTransformer


import nltk

# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np

# Setup cache directory
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
memory = Memory(cache_dir, verbose=0)


def embedd_data(dataset, model, device="cuda:0", batch_size=32):
    # Permetti di usare un sentence embedder opzionale
    sentence_embedder = None
    if hasattr(model, "encode"):
        # Assume che sia un sentence embedder tipo SentenceTransformer
        sentence_embedder = model

    if sentence_embedder is not None:
        # dataset: lista di tuple (input_ids,) o dict con 'input_ids'
        texts = []
        for item in dataset:
            if isinstance(item, dict):
                t = item.get("sentence", item.get("text", ""))
                if t == "":
                    t = item.get("question", item.get("prompt", item.get("code", "")))
            elif isinstance(item, (list, tuple)):
                # Se hai solo input_ids, non puoi decodificare senza tokenizer
                t = None
            else:
                t = None
            if t is not None:
                texts.append(t)
        # Usa il sentence embedder
        print("Embedding data with sentence embedder...", flush=True)
        embeddings = sentence_embedder.encode(
            texts, batch_size=batch_size, device=device, convert_to_tensor=True
        )
        print("Embedding done, shape = {}".format(embeddings.shape), flush=True)
        return embeddings

    # Altrimenti usa l'embedding layer del modello
    model.eval()
    embedding_layer = model.get_input_embeddings()
    embedding_layer.to(device)

    def collate_fn(batch):
        input_ids_list = []
        for item in batch:
            if isinstance(item, dict):
                t = item["input_ids"]
            elif isinstance(item, (list, tuple)):
                t = item[0]
            else:
                t = item
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            if t.dim() == 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            input_ids_list.append(t)
        return torch.stack(input_ids_list)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    all_embeddings = []
    with torch.no_grad():
        print("Embedding data...", flush=True)
        for batch_input_ids in data_loader:
            batch_input_ids = batch_input_ids.to(device)
            embedding = embedding_layer(batch_input_ids)
            all_embeddings.append(embedding.cpu())
        if all_embeddings:
            final_embeddings = torch.cat(all_embeddings, dim=0)
            print(
                "Embedding done, shape = {}".format(final_embeddings.shape), flush=True
            )
            return final_embeddings
        else:
            return torch.tensor([])


def cosine_similarity_vectorized(data):
    # Normalize the data along the vector dimension
    data_normalized = F.normalize(data.to(torch.float32), p=2, dim=1)

    similarity_matrix = torch.zeros(
        (data.shape[0], data.shape[0]), dtype=torch.float32, device=data.device
    )

    for i in range(0, data.shape[0], 512):
        end_i = min(i + 512, data.shape[0])
        batch = data_normalized[i:end_i]
        with torch.amp.autocast(
            "cuda",
        ):
            chunk = torch.matmul(batch, data_normalized.T)
        similarity_matrix[i:end_i] = chunk

    return similarity_matrix


def get_cosine_similarity(last_hidden_state_array_torch, distance="flatten"):
    if distance == "flatten":
        Matrix_embeddings = last_hidden_state_array_torch.view(
            last_hidden_state_array_torch.shape[0], -1
        )  # flatten the sequence length -> (batch_size, embedding_dim * sequence_length)
    elif distance == "mean":
        Matrix_embeddings = torch.mean(
            last_hidden_state_array_torch, dim=1
        )  # mean over the sequence length (batch_size, embedding_dim)

    cosine_similarit_matrix = cosine_similarity_vectorized(Matrix_embeddings)

    return cosine_similarit_matrix


def random_sample(dataloader, sample_per_dataset):
    calibration_data = []
    for dataset in dataloader:
        if len(dataset) > sample_per_dataset:
            indices = torch.randperm(len(dataset))[:sample_per_dataset]
            sampled_data = [dataset[i] for i in indices]
        else:
            sampled_data = dataset

        calibration_data.append(sampled_data)

    return torch.utils.data.ConcatDataset(calibration_data)


def use_embedding_for_sampling(
    dataloader,
    model,
    sample_per_dataset,
    distance,
    type="prototype",
    save_calibration_distribution=False,
    model_name="model",
    dataset_name="dataset",
    return_distribution=False,
):
    calibration_data = []
    original_distributions = []
    sample_distributions = []

    filename = "./out/cd/cd_{}_{}_{}.pt"

    for indice, dataset in enumerate(dataloader):
        print("\nDataset {}".format(indice), flush=True)
        last_hidden_state_array_torch = embedd_data(dataset, model, device="cuda:0")
        # print("embedding done", flush=True)
        cosine_similarity_matrix = get_cosine_similarity(
            last_hidden_state_array_torch, distance=distance
        )
        # print("cosine similarity done", flush=True)

        # Find the most similar embedding (prototipe)
        mean_cosine_similarity = torch.mean(cosine_similarity_matrix, dim=1)
        # print("mean done", flush=True)

        if return_distribution:
            original_distributions.append(mean_cosine_similarity.cpu())

        if save_calibration_distribution:
            torch.save(
                cosine_similarity_matrix.to(torch.float16),
                filename.format(model_name, dataset_name[indice], distance),
            )

        if type == "prototype" or type == "decoupled":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True)
        elif type == "most_different":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False)
        elif type == "distribution_matching":
            # Match the distribution of mean similarities (quantiles)
            sorted_indices_all = torch.argsort(mean_cosine_similarity)
            idx_indices = torch.linspace(
                0, len(sorted_indices_all) - 1, sample_per_dataset
            ).long()
            sorted_indices = sorted_indices_all[idx_indices]
        elif type == "herding":
            # Flatten embeddings if needed
            if last_hidden_state_array_torch.dim() == 3:
                emb = torch.mean(last_hidden_state_array_torch, dim=1)
            else:
                emb = last_hidden_state_array_torch
            sorted_indices = herding(emb, sample_per_dataset)

        if type == "decoupled":
            selected_indices = []
            selected_input_ids = []
            ngram_n = 3
            ngram_threshold = 0.5

            for idx in sorted_indices:
                if len(selected_indices) >= sample_per_dataset:
                    break

                item = dataset[idx]
                if isinstance(item, dict):
                    input_ids = item["input_ids"]
                elif isinstance(item, (list, tuple)):
                    input_ids = item[0]
                else:
                    input_ids = item

                if isinstance(input_ids, torch.Tensor):
                    input_ids_list = input_ids.cpu().tolist()
                else:
                    input_ids_list = input_ids

                is_similar = False
                candidate_ngrams = set(nltk.ngrams(input_ids_list, ngram_n))

                if len(candidate_ngrams) > 0:
                    for selected_ids in selected_input_ids:
                        selected_ngrams = set(nltk.ngrams(selected_ids, ngram_n))
                        if len(selected_ngrams) == 0:
                            continue

                        intersection = len(
                            candidate_ngrams.intersection(selected_ngrams)
                        )
                        union = len(candidate_ngrams.union(selected_ngrams))
                        jaccard = intersection / union

                        if jaccard > ngram_threshold:
                            is_similar = True
                            break

                if not is_similar:
                    selected_indices.append(idx)
                    selected_input_ids.append(input_ids_list)
                    calibration_data.append(dataset[idx])

            if return_distribution:
                if selected_indices:
                    sample_distributions.append(
                        mean_cosine_similarity[torch.stack(selected_indices)].cpu()
                    )
                else:
                    sample_distributions.append(torch.tensor([]))

        else:
            # print("ordering done", flush=True)
            for i in range(sample_per_dataset):
                calibration_data.append(dataset[sorted_indices[i]])

            if return_distribution:
                sample_distributions.append(
                    mean_cosine_similarity[sorted_indices[:sample_per_dataset]].cpu()
                )

    if return_distribution:
        return (
            calibration_data,
            torch.cat(original_distributions),
            torch.cat(sample_distributions),
        )
    return calibration_data


def get_intersection_over_union(data_torch_tokenized, count_number_occurrence=False):
    result_matrix = torch.zeros((len(data_torch_tokenized), len(data_torch_tokenized)))

    unique_list = []
    count_unique_list = []

    for i in range(len(data_torch_tokenized)):
        unique, counts = torch.unique(data_torch_tokenized[i], return_counts=True)
        unique_list.append(unique)
        count_unique_list.append(counts)

    for i in range(len(data_torch_tokenized)):
        for j in range(i + 1, len(data_torch_tokenized)):
            # print(f"Comparing sample {i} with sample {j}")

            unique_a, count_unique_a = unique_list[i], count_unique_list[i]
            unique_b, count_unique_b = unique_list[j], count_unique_list[j]

            common_elements = torch.isin(unique_a, unique_b, assume_unique=True)

            if count_number_occurrence:
                intersection_number = 0
                for index, is_common in enumerate(common_elements):
                    if is_common:
                        element = unique_a[index]
                        index_element_a = (unique_a == element).nonzero(as_tuple=False)
                        index_element_b = (unique_b == element).nonzero(as_tuple=False)

                        this_int = min(
                            count_unique_a[index_element_a].item(),
                            count_unique_b[index_element_b].item(),
                        )
                        intersection_number += this_int

                union_number = 2048 + 2048
            else:
                intersection_number = torch.sum(common_elements).item()
                union_number = unique_a.numel() + unique_b.numel() - intersection_number

            iou = intersection_number / union_number
            result_matrix[i, j] = iou
            result_matrix[j, i] = iou

    return result_matrix


def use_embedding_for_sampling_iou(
    dataloader,
    model,
    sample_per_dataset,
    count_number_occurrence,
    type="prototype_iou",
    save_calibration_distribution=False,
    model_name="model",
    dataset_name="dataset",
):
    calibration_data = []

    for indice, dataset in enumerate(dataloader):
        print("\nDataset {}".format(indice), flush=True)
        data_0 = []
        for d in dataset:
            if isinstance(d, dict):
                data_0.append(d["input_ids"])
            elif isinstance(d, (list, tuple)):
                data_0.append(d[0])
            else:
                data_0.append(d)
        iou_similarity_matrix = get_intersection_over_union(
            data_0, count_number_occurrence=count_number_occurrence
        )

        # Find the most similar embedding (prototipe)
        mean_cosine_similarity = torch.mean(iou_similarity_matrix, dim=1)

        # print(f"Dataset {indice} - Mean IoU: {mean_cosine_similarity.mean().item()}")  # Debugging output
        # print(torch.sort(mean_cosine_similarity, descending=True)[:100])  # Debugging output

        if type == "prototype_iou":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True)
        elif type == "most_different_iou":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False)

        for i in range(sample_per_dataset):
            calibration_data.append(dataset[sorted_indices[i]])

    return calibration_data


def get_similarity_matrix_from_model(data, model):
    device = model.device

    # Stack data
    # data is list of tensors.
    # If they are (1, seq), cat(dim=0) -> (N, seq)
    input_ids = torch.cat(data, dim=0).to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Last hidden state
    hidden_states = outputs.hidden_states[-1]

    # Mean pooling
    embeddings = torch.mean(hidden_states, dim=1)

    # Normalize
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Similarity
    similarity_matrix = torch.matmul(embeddings, embeddings.T)

    return similarity_matrix.cpu()


def use_embedding_for_sampling_st(
    dataloader,
    model,
    sample_per_dataset,
    tokenizer,
    type="prototype_st",
    save_calibration_distribution=False,
    model_name="model",
    dataset_name="dataset",
):
    calibration_data = []

    for indice, dataset in enumerate(dataloader):
        data_0 = []
        for d in dataset:
            if isinstance(d, dict):
                data_0.append(d["input_ids"])
            elif isinstance(d, (list, tuple)):
                data_0.append(d[0])
            else:
                data_0.append(d)
        st_similarity_matrix = get_similarity_matrix_from_model(data_0, model)

        # Find the most similar embedding (prototipe)
        mean_cosine_similarity = torch.mean(st_similarity_matrix, dim=1)
        # print(f"Dataset {indice} - Mean cosine similarity: {mean_cosine_similarity.mean().item()}")  # Debugging output
        # print(torch.sort(mean_cosine_similarity, descending=True)[:100])  # Debugging output

        if type == "prototype_st":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True)
        elif type == "most_different_st":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False)

        # TODO here check the cosine similarity across the selected samples, to see if they are diverse enough
        # or if they are similar to each other

        for i in range(sample_per_dataset):
            calibration_data.append(dataset[sorted_indices[i]])

    return calibration_data


def least_perplexity_sampling(
    dataloader,
    model,
    sample_per_dataset,
    tokenizer,
    return_distribution=False,
):
    calibration_data = []
    original_distributions = []
    sample_distributions = []

    for indice, dataset in enumerate(dataloader):
        perplexities = []

        for item in dataset:
            if isinstance(item, dict):
                input_ids = item["input_ids"]
            elif isinstance(item, (list, tuple)):
                input_ids = item[0]
            else:
                input_ids = item

            input_ids = input_ids.unsqueeze(0).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

        perplexities_tensor = torch.tensor(perplexities)

        if return_distribution:
            original_distributions.append(perplexities_tensor)

        sorted_indices = torch.argsort(perplexities_tensor, descending=False)

        for i in range(sample_per_dataset):
            calibration_data.append(dataset[sorted_indices[i]])

        if return_distribution:
            sample_distributions.append(
                perplexities_tensor[sorted_indices[:sample_per_dataset]]
            )

    if return_distribution:
        return (
            calibration_data,
            torch.cat(original_distributions),
            torch.cat(sample_distributions),
        )
    return calibration_data


def k_center_greedy(embeddings, n_samples):
    """
    Select n_samples from embeddings that maximize coverage (K-Center Greedy).
    embeddings: (N, D) tensor
    n_samples: number of samples to pick
    returns: indices of picked samples
    """
    if n_samples >= embeddings.shape[0]:
        return torch.arange(embeddings.shape[0])

    device = embeddings.device
    embeddings = embeddings.to(device)
    N = embeddings.shape[0]

    selected_indices = [0]
    # Use squared Euclidean distance for efficiency
    # (N, D) - (1, D) -> (N, D)
    diff = embeddings - embeddings[0]
    min_distances = torch.sum(diff * diff, dim=1)

    for _ in range(1, n_samples):
        # Pick the point furthest from its nearest center
        new_idx = torch.argmax(min_distances).item()
        selected_indices.append(new_idx)

        diff = embeddings - embeddings[new_idx]
        new_distances = torch.sum(diff * diff, dim=1)
        min_distances = torch.min(min_distances, new_distances)

    return torch.tensor(selected_indices)


def herding(embeddings, n_samples):
    """
    Select n_samples from embeddings using Kernel Herding to match the mean embedding.
    embeddings: (N, D) tensor
    n_samples: number of samples to pick
    returns: indices of picked samples
    """
    if n_samples >= embeddings.shape[0]:
        return torch.arange(embeddings.shape[0])

    device = embeddings.device
    embeddings = embeddings.to(device)
    # Normalize to work with cosine-like space
    embeddings = F.normalize(embeddings, p=2, dim=1)
    mu = torch.mean(embeddings, dim=0)

    selected_indices = []
    current_sum = torch.zeros_like(mu)

    # Mask to avoid picking the same index
    mask = torch.ones(embeddings.shape[0], device=device, dtype=torch.bool)

    for t in range(1, n_samples + 1):
        # We want to pick x such that (current_sum + x) / t is close to mu
        # argmin || (current_sum + x) - t*mu ||^2
        target = t * mu - current_sum

        # Find x in embeddings that is closest to target
        diff = embeddings - target
        distances = torch.sum(diff * diff, dim=1)
        distances[~mask] = float("inf")

        best_idx = torch.argmin(distances).item()

        selected_indices.append(best_idx)
        current_sum += embeddings[best_idx]
        mask[best_idx] = False

    return torch.tensor(selected_indices)


@memory.cache(ignore=["model", "dataloader", "tokenizer"])
def prepare_calibration(
    model,
    dataloader,
    nsamples=128,
    type="concat",
    distance="flatten",
    save_calibration_distribution=False,
    model_name="model",
    dataset_name="dataset",
    count_number_occurrence=False,
    tokenizer=None,
    return_distribution=False,
):
    """
    Prepare the calibration data by concatenating the datasets and limiting the number of samples.
    """
    # If multiple datasets, we sample nsamples from each and then use coreset to pick nsamples from the union
    if len(dataloader) > 1 and type != "concat":
        initial_sample_per_dataset = nsamples
    else:
        initial_sample_per_dataset = nsamples // len(dataloader)

    original_dist = None
    sample_dist = None

    if type == "concat":
        calibration_data = torch.utils.data.ConcatDataset(dataloader)
    elif type == "random_sample":
        calibration_data = random_sample(dataloader, initial_sample_per_dataset)
    elif (
        type == "prototype"
        or type == "most_different"
        or type == "decoupled"
        or type == "distribution_matching"
        or type == "herding"
    ):  # uses cosine similarity
        result = use_embedding_for_sampling(
            dataloader,
            model,
            initial_sample_per_dataset,
            distance,
            type=type,
            save_calibration_distribution=save_calibration_distribution,
            model_name=model_name,
            dataset_name=dataset_name,
            return_distribution=return_distribution,
        )
        if return_distribution:
            calibration_data, original_dist, sample_dist = result
        else:
            calibration_data = result
    elif (
        type == "prototype_iou" or type == "most_different_iou"
    ):  # uses intersection over union
        calibration_data = use_embedding_for_sampling_iou(
            dataloader,
            model,
            initial_sample_per_dataset,
            count_number_occurrence,
            type=type,
            save_calibration_distribution=save_calibration_distribution,
            model_name=model_name,
            dataset_name=dataset_name,
        )
    elif (
        type == "prototype_st" or type == "most_different_st"
    ):  # uses sentence transformers
        calibration_data = use_embedding_for_sampling_st(
            dataloader,
            model,
            initial_sample_per_dataset,
            tokenizer,
            type=type,
            save_calibration_distribution=save_calibration_distribution,
            model_name=model_name,
            dataset_name=dataset_name,
        )
    elif type == "least_perplexity":
        result = least_perplexity_sampling(
            dataloader,
            model,
            initial_sample_per_dataset,
            tokenizer,
            return_distribution=return_distribution,
        )
        if return_distribution:
            calibration_data, original_dist, sample_dist = result
        else:
            calibration_data = result

    # Coreset resampling if we have multiple datasets
    if len(dataloader) > 1 and type != "concat":
        coreset_method = (
            "Herding"
            if type in ["herding", "distribution_matching"]
            else "K-Center Greedy"
        )
        print(
            f"Resampling from {len(calibration_data)} candidates to {nsamples} using coreset ({coreset_method})...",
            flush=True,
        )

        # Convert ConcatDataset to list if necessary
        if isinstance(calibration_data, torch.utils.data.ConcatDataset):
            pool = [calibration_data[i] for i in range(len(calibration_data))]
        else:
            pool = calibration_data

        # Embed the pool
        pool_embeddings = embedd_data(pool, model, device="cuda:0")

        # If embeddings are (N, L, D), mean pool them
        if pool_embeddings.dim() == 3:
            pool_embeddings = torch.mean(pool_embeddings, dim=1)

        if type in ["herding", "distribution_matching"]:
            selected_indices = herding(pool_embeddings, nsamples)
        else:
            selected_indices = k_center_greedy(pool_embeddings, nsamples)

        calibration_data = [pool[i] for i in selected_indices]

        if return_distribution and sample_dist is not None:
            sample_dist = sample_dist[selected_indices]

    print(f"Calibration data prepared with {len(calibration_data)} samples.")

    if return_distribution:
        return calibration_data, original_dist, sample_dist
    return calibration_data


if __name__ == "__main__":
    #    matplotlib.use('WebAgg')  # Use a non-interactive backend
    # For example use winogrande, mawps
    # Esempio di caricamento dataset HuggingFace compatibile
    from datasets import load_dataset

    dataset = load_dataset("mu-nlpc/calc-mawps", split="train")
    try:
        dataset_len = len(dataset)
    except Exception:
        dataset_len = getattr(dataset, "num_rows", None)
    print(f"Loaded dataset with {dataset_len} samples.")

    # Usa SentenceTransformer come sentence embedder
    model_name = "all-MiniLM-L6-v2"
    sentence_embedder = SentenceTransformer(model_name, device="cuda")

    # Preprocess dataset: estrai testo
    texts = []
    for item in dataset:
        if isinstance(item, dict):
            text = item.get("sentence", "") or item.get("text", "")
            if text == "":
                text = (
                    item.get("question", "")
                    or item.get("prompt", "")
                    or item.get("code", "")
                )
            texts.append(text)
        else:
            # Se non è dict, prova a convertirlo o ignora
            texts.append(str(item))
    print(f"Preprocessed {len(texts)} samples.")

    # Embedding con sentence embedder
    embeddings = embedd_data(texts, sentence_embedder, device="cuda", batch_size=64)
    print("Embedding completed.")

    # Compute cosine similarity matrix
    cosine_similarity_matrix = get_cosine_similarity(embeddings, distance="flatten")
    print("Cosine similarity matrix computed.")

    # Seleziona 128 campioni per la calibrazione
    calibration_data = use_embedding_for_sampling(
        [texts],
        sentence_embedder,
        sample_per_dataset=128,
        distance="flatten",
        type="prototype",
        save_calibration_distribution=False,
        model_name="all-MiniLM-L6-v2",
        dataset_name="mawps",
    )
    print(f"Calibration data selected with {len(calibration_data)} samples.")

    # Calcola similarità tra coppie
    threshold = 0.90
    for i in range(len(calibration_data)):
        for j in range(i + 1, len(calibration_data)):
            cos_sim = cosine_similarity_matrix[i, j].item()
            if cos_sim > threshold:
                print(f"Cosine similarity between sample {i} and sample {j}: {cos_sim}")
                print(f"Sample {i}: {calibration_data[i]}")
                print(f"Sample {j}: {calibration_data[j]}")
    # Calcola overlap n-gram
    n = 3
    ngram_overlap_matrix = np.zeros((len(calibration_data), len(calibration_data)))
    ngram_overlap_counts = []
    for i in range(len(calibration_data)):
        ngrams_i = set(nltk.ngrams(calibration_data[i].split(), n))
        for j in range(i + 1, len(calibration_data)):
            ngrams_j = set(nltk.ngrams(calibration_data[j].split(), n))
            overlap = ngrams_i.intersection(ngrams_j)
            ngram_overlap_counts.append(len(overlap))
            ngram_overlap_matrix[i, j] = len(overlap)
            ngram_overlap_matrix[j, i] = len(overlap)
            print(
                f"N-gram overlap (n={n}) between sample {i} and sample {j}: {len(overlap)}"
            )
            print(f"Sample {i} n-grams: {ngrams_i}")
            print(f"Sample {j} n-grams: {ngrams_j}")
    # Media sulla triangolare superiore
    upper_triangle_indices = np.triu_indices(len(calibration_data), k=1)
    mean_cosine_similarity = (
        cosine_similarity_matrix[upper_triangle_indices].mean().item()
    )
    print(
        f"Mean cosine similarity among selected calibration samples: {mean_cosine_similarity}"
    )
