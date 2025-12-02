"""
Part of the code is adapted from: https://github.com/muraronicola/Muraro-Nicola-Master-Thesis
"""

import torch
import torch.nn.functional as F

# import wandb
from transformers import AutoTokenizer
from data import get_dataset
import nltk

# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np


def embedd_data(dataset, model, device="cuda:0", batch_size=32):
    model.eval()

    embedding_layer = model.get_input_embeddings()
    # embedding_layer.to(device) # It should already be on device if model is

    # Define a collate function to handle different input types
    def collate_fn(batch):
        # batch is a list of items from dataset
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

            # Ensure shape is (Seq_Len,)
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

    # Compute cosine similarity matrix using matrix multiplication
    similarity_matrix = torch.matmul(data_normalized, data_normalized.T)
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
):
    calibration_data = []

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

        if save_calibration_distribution:
            torch.save(
                cosine_similarity_matrix.to(torch.float16),
                filename.format(model_name, dataset_name[indice], distance),
            )

        if type == "prototype":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=True)
        elif type == "most_different":
            sorted_indices = torch.argsort(mean_cosine_similarity, descending=False)

        # print("ordering done", flush=True)
        for i in range(sample_per_dataset):
            calibration_data.append(dataset[sorted_indices[i]])

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
        data_0 = [d[0] for d in dataset]
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
        data_0 = [d[0] for d in dataset]
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
):
    """
    Prepare the calibration data by concatenating the datasets and limiting the number of samples.
    """
    sample_per_dataset = nsamples // len(dataloader)

    if type == "concat":
        calibration_data = torch.utils.data.ConcatDataset(dataloader)
    if type == "random_sample":
        calibration_data = random_sample(dataloader, sample_per_dataset)
    elif type == "prototype" or type == "most_different":  # uses cosine similarity
        calibration_data = use_embedding_for_sampling(
            dataloader,
            model,
            sample_per_dataset,
            distance,
            type=type,
            save_calibration_distribution=save_calibration_distribution,
            model_name=model_name,
            dataset_name=dataset_name,
        )
    elif (
        type == "prototype_iou" or type == "most_different_iou"
    ):  # uses intersection over union
        calibration_data = use_embedding_for_sampling_iou(
            dataloader,
            model,
            sample_per_dataset,
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
            sample_per_dataset,
            tokenizer,
            type=type,
            save_calibration_distribution=save_calibration_distribution,
            model_name=model_name,
            dataset_name=dataset_name,
        )

    print(f"Calibration data prepared with {len(calibration_data)} samples.")

    return calibration_data


if __name__ == "__main__":
    #    matplotlib.use('WebAgg')  # Use a non-interactive backend
    # For example use winogrande, mawps
    dataset = get_dataset("mawps", split="train")
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Now let's use a small model Qwen/Qwen3-1.7B there is also 0.6B, 4B and 8nB
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-1.7B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to("cuda:0")

    # Preprocess dataset
    print("Preprocessing dataset...")
    processed_dataset = []
    # Limit to a small number for testing
    subset_size = 2**9
    for i in range(min(len(dataset), subset_size)):
        item = dataset[i]
        text = item.get("sentence", item.get("text", ""))  # Handle winogrande or others
        # If no sentence use 'question' or 'prompt' or 'code'
        if text == "":
            text = item.get("question", item.get("prompt", item.get("code", "")))
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        processed_dataset.append((encoded["input_ids"],))

    dataset = processed_dataset
    print(f"Preprocessed {len(dataset)} samples.")

    embedd_data(dataset, model, device="cuda:0")
    print("Embedding completed.")

    # Compute the cosine similarity matrix
    last_hidden_state_array_torch = embedd_data(dataset, model, device="cuda:0")
    cosine_similarity_matrix = get_cosine_similarity(
        last_hidden_state_array_torch, distance="flatten"
    )
    print("Cosine similarity matrix computed.")

    # Pick 128 samples for calibration, use prototype for most similar or most_different for you know...
    calibration_data = use_embedding_for_sampling(
        [dataset],
        model,
        sample_per_dataset=128,
        distance="flatten",
        type="prototype",
        save_calibration_distribution=False,
        model_name="qwen-1.8b",
        dataset_name="winogrande",
    )
    print(f"Calibration data selected with {len(calibration_data)} samples.")

    # Compare between pairs to see if cosine similarity is high
    threshold = 0.90
    for i in range(calibration_data.__len__()):
        for j in range(i + 1, calibration_data.__len__()):
            cos_sim = cosine_similarity_matrix[i, j].item()
            if cos_sim > threshold:
                print(f"Cosine similarity between sample {i} and sample {j}: {cos_sim}")
                # Print the sentences
                text_i = tokenizer.decode(
                    calibration_data[i][0][0], skip_special_tokens=True
                )
                text_j = tokenizer.decode(
                    calibration_data[j][0][0], skip_special_tokens=True
                )
                print(f"Sample {i}: {text_i}")
                print(f"Sample {j}: {text_j}")
    # Measure the n-gram overlap among the calibration samples using nltk
    n = 3  # Trigram
    ngram_overlap_matrix = np.zeros(
        (calibration_data.__len__(), calibration_data.__len__())
    )
    ngram_overlap_counts = []
    for i in range(calibration_data.__len__()):
        text_i = tokenizer.decode(calibration_data[i][0][0], skip_special_tokens=True)
        ngrams_i = set(nltk.ngrams(text_i.split(), n))
        for j in range(i + 1, calibration_data.__len__()):
            text_j = tokenizer.decode(
                calibration_data[j][0][0], skip_special_tokens=True
            )
            ngrams_j = set(nltk.ngrams(text_j.split(), n))
            overlap = ngrams_i.intersection(ngrams_j)
            ngram_overlap_counts.append(len(overlap))
            ngram_overlap_matrix[i, j] = len(overlap)
            ngram_overlap_matrix[j, i] = len(overlap)
            print(
                f"N-gram overlap (n={n}) between sample {i} and sample {j}: {len(overlap)}"
            )
            print(f"Sample {i} n-grams: {ngrams_i}")
            print(f"Sample {j} n-grams: {ngrams_j}")
    # Plot histogram of n-gram overlaps
    # plt.figure(figsize=(10, 6))
    # plt.heatmap(ngram_overlap_matrix, annot=True, fmt="d", cmap="YlGnBu")
    # plt.xlabel('Sample Index')
    # plt.ylabel('Sample Index')
    # plt.title(f'N-gram (n={n}) Overlap Counts among Calibration Samples')
    # plt.show()
    # Mean over the upper triangle
    upper_triangle_indices = torch.triu_indices(
        calibration_data.__len__(), calibration_data.__len__(), offset=1
    )
    mean_cosine_similarity = (
        cosine_similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
        .mean()
        .item()
    )
    print(
        f"Mean cosine similarity among selected calibration samples: {mean_cosine_similarity}"
    )
