"""
Part of the code is adapted from: https://github.com/muraronicola/Muraro-Nicola-Master-Thesis
"""

import os
import torch
import torch.nn.functional as F
from joblib import Memory
import collections
from tqdm import tqdm

# import wandb
from sentence_transformers import SentenceTransformer


import nltk

# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np

# Setup cache directory
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
memory = Memory(cache_dir, verbose=0)


def embedd_data(dataset, model, device="cuda:0", batch_size=128, tokenizer=None):
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
                # If still empty and item has input_ids, decode back to text
                if t == "" and "input_ids" in item and tokenizer is not None:
                    ids = item["input_ids"]
                    if isinstance(ids, torch.Tensor):
                        ids = ids.tolist()
                    t = tokenizer.decode(ids, skip_special_tokens=True)
            elif isinstance(item, (list, tuple)):
                # Se hai solo input_ids, prova a decodificare con tokenizer
                if tokenizer is not None:
                    ids = item[0] if len(item) > 0 else item
                    if isinstance(ids, torch.Tensor):
                        ids = ids.tolist()
                    t = tokenizer.decode(ids, skip_special_tokens=True)
                else:
                    t = None
            else:
                t = None
            if t is not None and t != "":
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

    with torch.amp.autocast("cuda"):
        for i in range(0, data.shape[0], 2048):
            end_i = min(i + 2048, data.shape[0])
            batch = data_normalized[i:end_i]
            chunk = torch.matmul(batch, data_normalized.T)
            similarity_matrix[i:end_i] = chunk

    return similarity_matrix


def get_cosine_similarity(last_hidden_state_array_torch, distance="flatten"):
    if last_hidden_state_array_torch.numel() == 0:
        print("WARNING: get_cosine_similarity received empty tensor, returning empty matrix.", flush=True)
        return torch.zeros((0, 0), dtype=torch.float32)

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
            indices = torch.randperm(len(dataset))[:sample_per_dataset].tolist()
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
    tokenizer=None,
):
    calibration_data = []
    original_distributions = []
    sample_distributions = []

    filename = "./out/cd/cd_{}_{}_{}.pt"

    for indice, dataset in enumerate(dataloader):
        print("\nDataset {}".format(indice), flush=True)
        last_hidden_state_array_torch = embedd_data(dataset, model, device="cuda:0", tokenizer=tokenizer)
        if last_hidden_state_array_torch.numel() == 0:
            print(f"WARNING: Dataset {indice} produced empty embeddings, skipping.", flush=True)
            continue
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
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True).tolist()
        elif type == "most_different":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False).tolist()
        elif type == "distribution_matching":
            # Match the distribution of mean similarities (quantiles)
            sorted_indices_all = torch.argsort(mean_cosine_similarity)
            idx_indices = torch.linspace(
                0, len(sorted_indices_all) - 1, sample_per_dataset
            ).long()
            sorted_indices = sorted_indices_all[idx_indices].tolist()
        elif type == "distribution_matching_no_outliers":
            # Match the distribution but exclude the extreme 5% tails
            sorted_indices_all = torch.argsort(mean_cosine_similarity)
            N = len(sorted_indices_all)
            start_idx = int(0.05 * N)
            end_idx = int(0.95 * N)
            idx_indices = torch.linspace(start_idx, end_idx, sample_per_dataset).long()
            sorted_indices = sorted_indices_all[idx_indices].tolist()
        elif type == "herding":
            # Flatten embeddings if needed
            if last_hidden_state_array_torch.dim() == 3:
                emb = torch.mean(last_hidden_state_array_torch, dim=1)
            else:
                emb = last_hidden_state_array_torch
            sorted_indices = herding(emb, sample_per_dataset).tolist()

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
                    # mean_cosine_similarity is a tensor, we can index it with a list of ints
                    sample_distributions.append(
                        mean_cosine_similarity[selected_indices].cpu()
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
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True).tolist()
        elif type == "most_different_iou":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False).tolist()

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
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True).tolist()
        elif type == "most_different_st":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False).tolist()

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

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    ppl_batch_size = 8

    for indice, dataset in enumerate(dataloader):
        # Collect all input_ids first
        all_input_ids = []
        for item in dataset:
            if isinstance(item, dict):
                input_ids = item["input_ids"]
            elif isinstance(item, (list, tuple)):
                input_ids = item[0]
            else:
                input_ids = item
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            all_input_ids.append(input_ids)

        # Batch forward passes for perplexity
        perplexities = []
        with torch.no_grad():
            for i in range(0, len(all_input_ids), ppl_batch_size):
                batch_ids = torch.stack(all_input_ids[i:i+ppl_batch_size]).to(model.device)
                outputs = model(batch_ids)
                logits = outputs.logits
                # Shift for causal LM loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_ids[:, 1:].contiguous()
                per_token_loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(batch_ids.size(0), -1)
                per_sample_loss = per_token_loss.mean(dim=1)
                per_sample_ppl = torch.exp(per_sample_loss)
                perplexities.extend(per_sample_ppl.cpu().tolist())

        perplexities_tensor = torch.tensor(perplexities)

        if return_distribution:
            original_distributions.append(perplexities_tensor)

        sorted_indices = torch.argsort(perplexities_tensor, descending=False).tolist()

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


def zipf_sampling(dataloader, sample_per_dataset, tokenizer=None, shuffle=False):
    """
    Selects samples that maximize the number of rare words (Zipfian distribution).
    Optionally scrambles the word order if shuffle=True.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for Zipf sampling")

    calibration_data = []
    
    # 1. Collect all words frequencies in a fast way
    word_counts = collections.Counter()
    all_items = []  # Store items to access later
    
    # Pre-process word sets for each sentence to avoid re-splitting later
    sentence_word_sets = [] 

    print("Collecting word statistics for Zipf sampling...", flush=True)
    
    # Iterate over all datasets in the dataloader list
    for dataset in dataloader:
        for item in dataset:
            input_ids = item["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.view(-1).tolist()
                
            text = tokenizer.decode(input_ids, skip_special_tokens=True).lower()
            
            # Simple tokenization by splitting
            words = text.replace('.', ' ').replace(',', ' ').split()
            
            # If shuffle is requested, shuffle the words in the list and re-tokenize
            if shuffle:
                np.random.shuffle(words)
                shuffled_text = " ".join(words)
                
                # Re-encode to get input_ids back for the 'item'
                # Truncate to standard length
                encoded = tokenizer(shuffled_text, truncation=True, max_length=128, return_tensors="pt")
                if encoded.input_ids.shape[1] == 0: 
                    continue
                
                # Create a new item with shuffled content
                new_item = {
                    "input_ids": encoded.input_ids[0].cpu(),
                    "attention_mask": encoded.attention_mask[0].cpu()
                }
                all_items.append(new_item)
            else:
                all_items.append(item)
            
            # Important: Update word counts even if shuffled (counts are invariant to shuffle)
            word_counts.update(words)
            
            # Store unique words for this sentence
            sentence_word_sets.append(set(words))
            
    # 2. Identify "rare" words
    # Calculate Rareness Score for each word: 1 / frequency
    word_scores = {w: 1.0 / count for w, count in word_counts.items()}
    
    # 3. Score sentences
    sentence_scores = []
    for words_set in sentence_word_sets:
        # Score = Sum of scores of unique words in the sentence
        # We use unique words so repeating a rare word doesn't boost score artificially
        score = sum(word_scores.get(w, 0) for w in words_set)
        sentence_scores.append(score)
    
    # 4. Select top samples
    # Get indices of top `sample_per_dataset` scores
    # If we have fewer items than requested, take all
    n_select = min(len(all_items), sample_per_dataset)
    top_indices = np.argsort(sentence_scores)[-n_select:]
    
    for idx in top_indices:
        calibration_data.append(all_items[idx])
        
    print(f"Zipf Sampling (shuffle={shuffle}): Selected {len(calibration_data)} samples.")
    return calibration_data

def old_zipf_sampling_disabled(dataloader, sample_per_dataset, tokenizer=None):
    """
    Original Zipf sampling logic (Tail Coverage + Greedy Distribution Matching).
    Renamed to avoid conflict with new Shuffled-aware Zipf sampling.
    """
    calibration_data = []

    for indice, dataset in enumerate(dataloader):
        print(f"Old Zipf sampling for Dataset {indice}", flush=True)

        all_items = []
        for item in dataset:
            if isinstance(item, dict):
                all_items.append(item)
            else:
                all_items.append({"input_ids": item})

        # 1. Pre-calculate normalization map
        id_to_norm = {}
        if tokenizer:
            all_unique_ids = set()
            for item in all_items:
                ids = item["input_ids"]
                if isinstance(ids, torch.Tensor):
                    ids = ids.view(-1).tolist()
                all_unique_ids.update(ids)

            for tid in all_unique_ids:
                if tid in tokenizer.all_special_ids:
                    id_to_norm[tid] = None
                    continue
                t_str = (
                    tokenizer.decode([tid], skip_special_tokens=True)
                    .lower()
                    .replace(" ", "")
                    .replace("Ġ", "")
                    .strip()
                )
                id_to_norm[tid] = t_str if t_str else None

        global_counts = collections.Counter()
        sentence_tokens = []

        for item in all_items:
            ids = item["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.view(-1).tolist()

            if tokenizer:
                ids_to_use = [
                    id_to_norm[tid] for tid in ids if id_to_norm.get(tid) is not None
                ]
            else:
                ids_to_use = [tid for tid in ids if tid != 0]

            global_counts.update(ids_to_use)
            sentence_tokens.append(ids_to_use)


        total_tokens = sum(global_counts.values())
        if total_tokens == 0:
            print(f"Warning: No tokens found for dataset {indice}")
            continue

        target_probs = {t: c / total_tokens for t, c in global_counts.items()}
        # Tokens sorted by rarity (ascending counts)
        unique_tokens = sorted(global_counts.keys(), key=lambda x: global_counts[x])

        # 3. Map tokens to sentences for tail coverage
        token_to_sentences = collections.defaultdict(list)
        for i, s_tokens in enumerate(sentence_tokens):
            for t in set(s_tokens):
                token_to_sentences[t].append(i)

        selected_indices = []
        selected_indices_set = set()
        nsamples = sample_per_dataset

        # Step 1: Tail coverage (the "long" part)
        # Pick one sentence for each of the rarest tokens until we reach half budget
        for t in unique_tokens:
            if len(selected_indices) >= nsamples // 2:
                break
            candidates = token_to_sentences[t]
            for idx in candidates:
                if idx not in selected_indices_set:
                    selected_indices.append(idx)
                    selected_indices_set.add(idx)
                    break

        # Step 2: Greedy matching for the rest of the budget
        current_counts = collections.Counter()
        for idx in selected_indices:
            current_counts.update(sentence_tokens[idx])

        while len(selected_indices) < nsamples:
            remaining_indices = [
                i for i in range(len(all_items)) if i not in selected_indices_set
            ]
            if not remaining_indices:
                break

            total_selected_tokens = sum(current_counts.values())

            # Sample a pool if candidates are many
            pool_size = 500
            if len(remaining_indices) > pool_size:
                pool = np.random.choice(remaining_indices, pool_size, replace=False)
            else:
                pool = remaining_indices

            best_idx = -1
            best_score = -float("inf")

            for idx in pool:
                # Heuristic: score = sum_{t in s} (P_target(t) - P_current(t)) * counts_in_s(t)
                s_tokens = sentence_tokens[idx]
                s_counts = collections.Counter(s_tokens)
                score = 0
                for t, count in s_counts.items():
                    target_p = target_probs.get(t, 0)
                    current_p = (
                        current_counts[t] / total_selected_tokens
                        if total_selected_tokens > 0
                        else 0
                    )
                    score += (target_p - current_p) * count

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1:
                best_idx = remaining_indices[0]

            selected_indices.append(best_idx)
            selected_indices_set.add(best_idx)
            current_counts.update(sentence_tokens[best_idx])

        for i in selected_indices:
            calibration_data.append(dataset[i])

    return calibration_data

def unique_tokens_sampling(dataloader, sample_per_dataset, tokenizer=None):
    """
    Select samples from each dataset that maximize the total number of unique tokens.
    Tokens are normalized (lowercase, no subword markers) to count concepts.
    """
    calibration_data = []

    for indice, dataset in enumerate(dataloader):
        print(f"\nUnique tokens sampling for Dataset {indice}", flush=True)

        all_items = []
        for item in dataset:
            if isinstance(item, dict):
                all_items.append(item)
            else:
                all_items.append({"input_ids": item})

        # 1. Pre-calculate normalization map
        id_to_norm = {}
        if tokenizer:
            print("Pre-calculating token normalization map...", flush=True)
            all_unique_ids = set()
            for item in all_items:
                ids = item["input_ids"]
                if isinstance(ids, torch.Tensor):
                    ids = ids.view(-1).tolist()
                all_unique_ids.update(ids)

            for tid in all_unique_ids:
                if tid in tokenizer.all_special_ids:
                    id_to_norm[tid] = None
                    continue
                t_str = (
                    tokenizer.decode([tid], skip_special_tokens=True)
                    .lower()
                    .replace(" ", "")
                    .replace("Ġ", "")
                    .strip()
                )
                id_to_norm[tid] = t_str if t_str else None

        # 2. Extract unique normalized concepts for each sentence
        sentence_concepts = []
        for item in all_items:
            ids = item["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.view(-1).tolist()

            if tokenizer:
                concepts = {
                    id_to_norm[tid] for tid in ids if id_to_norm.get(tid) is not None
                }
            else:
                concepts = {tid for tid in ids if tid != 0}
            sentence_concepts.append(concepts)

        # 3. Greedy selection to maximize coverage
        selected_indices = []
        covered_concepts = set()
        nsamples = sample_per_dataset

        # Maintain a set of eligible indices to avoid O(N) rebuild each iteration
        eligible_set = set(range(len(all_items)))

        pbar = tqdm(total=nsamples, desc="Selecting samples")
        while len(selected_indices) < nsamples:
            best_idx = -1
            max_new = -1

            if not eligible_set:
                break

            # If the dataset is very large, subsample the search pool to speed up
            if len(eligible_set) > 2000:
                search_pool = np.random.choice(list(eligible_set), 2000, replace=False)
            else:
                search_pool = eligible_set

            for idx in search_pool:
                new_concepts = sentence_concepts[idx] - covered_concepts
                count = len(new_concepts)
                if count > max_new:
                    max_new = count
                    best_idx = idx
                elif count == max_new and best_idx != -1:
                    # Tie-break: pick the one with more total concepts
                    if len(sentence_concepts[idx]) > len(sentence_concepts[best_idx]):
                        best_idx = idx

            if best_idx == -1:
                break

            selected_indices.append(best_idx)
            eligible_set.discard(best_idx)
            covered_concepts.update(sentence_concepts[best_idx])
            pbar.update(1)
        pbar.close()

        for i in selected_indices:
            calibration_data.append(dataset[i])

    return calibration_data


def random_words_sampling(nsamples, tokenizer, sentence_length=128):
    """
    Generates samples consisting of random English words.
    """
    import random
    from nltk.corpus import words

    try:
        word_list = words.words()
    except LookupError:
        import nltk

        nltk.download("words")
        word_list = words.words()

    calibration_data = []
    print(f"Generating {nsamples} samples of random words...", flush=True)
    for _ in range(nsamples):
        sentence = " ".join(random.choices(word_list, k=sentence_length))
        encoded = tokenizer(
            sentence,
            truncation=True,
            max_length=sentence_length,
            padding="max_length",
            return_tensors="pt",
        )
        calibration_data.append(
            {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        )
    return calibration_data


def words_dataset_sampling(dataloader, nsamples, tokenizer, sentence_length=128):
    """
    Collects all words from the dataset, maintain their original frequencies,
    and generates new samples by combining them (Zipf-aware sampling).
    """
    import random

    all_tokens = []
    print("Collecting words from dataset for Zipf-aware sampling...", flush=True)

    for dataset in dataloader:
        for item in dataset:
            if isinstance(item, dict):
                input_ids = item["input_ids"]
            elif isinstance(item, (list, tuple)):
                input_ids = item[0]
            else:
                input_ids = item

            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.view(-1).tolist()

            # Decode the sequence to get the text
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            # Split into words (simplistic splitting)
            words = text.lower().split()
            all_tokens.extend(words)

    if not all_tokens:
        print("Warning: No words found in dataset. Returning empty calibration data.")
        return []

    # Shuffle all tokens to create a random bag-of-words pool that matches the original distribution
    random.shuffle(all_tokens)
    num_tokens = len(all_tokens)

    print(
        f"Collected {num_tokens} tokens. Generating {nsamples} samples...",
        flush=True,
    )

    calibration_data = []
    token_idx = 0

    for _ in range(nsamples):
        # Pick words for the next sentence by cycling through the token pool
        sentence_words = []
        for _ in range(sentence_length):
            sentence_words.append(all_tokens[token_idx % num_tokens])
            token_idx += 1

        sentence = " ".join(sentence_words)
        encoded = tokenizer(
            sentence,
            truncation=True,
            max_length=sentence_length,
            padding="max_length",
            return_tensors="pt",
        )
        calibration_data.append(
            {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        )

    return calibration_data


def dictionary_sampling(nsamples, tokenizer, sentence_length=128):
    """
    Generates samples consisting of the whole dictionary words.
    Ignores nsamples usually, returning the full dictionary chunked.
    """
    from nltk.corpus import words
    try:
        word_list = words.words()
    except LookupError:
        import nltk
        nltk.download("words")
        word_list = words.words()

    print(f"Loading dictionary with {len(word_list)} words...", flush=True)
    full_text = " ".join(word_list)
    
    # Tokenize the full dictionary text
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"][0] # (N,)
    attention_mask = encoded["attention_mask"][0] # (N,)
    
    total_tokens = input_ids.size(0)
    print(f"Total tokens in dictionary: {total_tokens}", flush=True)
    
    calibration_data = []
    
    for i in range(0, total_tokens, sentence_length):
        chunk_ids = input_ids[i : i + sentence_length]
        chunk_mask = attention_mask[i : i + sentence_length]
        
        # Pad if needed
        if chunk_ids.size(0) < sentence_length:
            pad_len = sentence_length - chunk_ids.size(0)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            # Create padding tensor and concatenate
            padding = torch.full((pad_len,), pad_id, dtype=chunk_ids.dtype, device=chunk_ids.device)
            chunk_ids = torch.cat([chunk_ids, padding])
            
            padding_mask = torch.zeros((pad_len,), dtype=chunk_mask.dtype, device=chunk_mask.device)
            chunk_mask = torch.cat([chunk_mask, padding_mask])
            
        calibration_data.append(
            {
                "input_ids": chunk_ids,
                "attention_mask": chunk_mask,
            }
        )
        
    print(f"Created {len(calibration_data)} dictionary samples (ignoring nsamples={nsamples})", flush=True)
    return calibration_data


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
    if type == "most_similar": 
        type = "prototype"  # alias

    if type == "concat":
        calibration_data = torch.utils.data.ConcatDataset(dataloader)
    elif type == "random_sample":
        calibration_data = random_sample(dataloader, initial_sample_per_dataset)
    elif (
        type == "prototype"
        or type == "most_different"
        or type == "decoupled"
        or type == "distribution_matching"
        or type == "distribution_matching_no_outliers"
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
            tokenizer=tokenizer,
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
    elif type == "zipf":
        calibration_data = zipf_sampling(
            dataloader, initial_sample_per_dataset, tokenizer=tokenizer
        )
    elif type == "shuffled_zipf":
        calibration_data = zipf_sampling(
            dataloader, initial_sample_per_dataset, tokenizer=tokenizer, shuffle=True
        )
    elif type == "unique_tokens":
        calibration_data = unique_tokens_sampling(
            dataloader, initial_sample_per_dataset, tokenizer=tokenizer
        )
    elif type == "unique_tokens_shuffled":
        calibration_data = unique_tokens_sampling(
            dataloader, initial_sample_per_dataset, tokenizer=tokenizer, shuffle=True
        )
    elif type == "random_words":
        calibration_data = random_words_sampling(nsamples, tokenizer)
    elif type == "words_dataset":
        calibration_data = words_dataset_sampling(dataloader, nsamples, tokenizer)
    elif type == "dictionary":
        calibration_data = dictionary_sampling(nsamples, tokenizer)

    # Coreset resampling if we have multiple datasets
    if len(dataloader) > 1 and type != "concat":
        coreset_method = (
            "Herding"
            if type
            in ["herding", "distribution_matching", "distribution_matching_no_outliers"]
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
        pool_embeddings = embedd_data(pool, model, device="cuda:0", tokenizer=tokenizer)

        # If embeddings are (N, L, D), mean pool them
        if pool_embeddings.dim() == 3:
            pool_embeddings = torch.mean(pool_embeddings, dim=1)

        if type in [
            "herding",
            "distribution_matching",
            "distribution_matching_no_outliers",
        ]:
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
