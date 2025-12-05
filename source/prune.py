from similarity_check import (
    prepare_calibration,
)
from wanda_analysis import WandaAnalysis
from eval import evaluate_model
import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from data import get_dataset, get_text_from_item
from datasets import Dataset

from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
import logging
from prettytable import PrettyTable
import argparse

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)


# --- MODIFIED: get_tokenized_data ---
# This function is now used for *both* similarity sampling and evaluation.
# 1. It no longer returns PyTorch tensors (return_tensors="pt" removed).
# 2. It returns a dictionary with 'input_ids' and 'attention_mask' as Python lists.
# 3. It no longer does max_length padding (this is deferred to the Data Collator).
def get_tokenized_data(
    dataset, tokenizer, dataset_name, max_length=128, return_tensors=False
):
    processed_dataset = []
    for item in dataset:
        text = get_text_from_item(item, dataset_name)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            # return_tensors="pt" is now optional/controlled by argument
        )

        # NOTE: We return the raw dict if we don't need tensors yet (for DataCollator)
        if return_tensors:
            # For similarity check, we might still need tensors in the loop
            encoded = tokenizer.pad(
                encoded,
                padding="max_length",
                return_tensors="pt",
                max_length=max_length,
            )
            processed_dataset.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                }
            )
        else:
            # For the DataLoader, return list/arrays, let the Collator do the rest
            processed_dataset.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                }
            )

    return processed_dataset


if __name__ == "__main__":
    # TODO Add as argument the datasets to use and the model to prune
    # Pruning type from command line
    parser = argparse.ArgumentParser(description="Wanda Pruning Script")
    parser.add_argument(
        "--pruning_type",
        type=str,
        choices=[
            "most_similar",
            "random",
            "most_similar_decoupled",
            "most_dissimilar",
            "least_perplexity",
        ],
        default="most_similar",
        help="Type of pruning to perform: 'most_similar', 'random', 'most_similar_decoupled', 'most_dissimilar', or 'least_perplexity'",
    )
    args = parser.parse_args()
    pruning_type = args.pruning_type

    # Load model and tokenizer
    model_name = "Qwen/Qwen3-1.7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loaded model {model_name} for embedding extraction.")

    dataset_name = "gsm8k"
    # Use train or test if the dataset doesn't have train
    dataset = get_dataset(
        dataset_name,
        split="train" if "train" in get_dataset(dataset_name).keys() else "test",
    )
    log.info(f"Loaded dataset {dataset_name} with {len(dataset)} samples.")
    # Preprocess dataset for similarity check
    subset_size = 5000
    if len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))

    # Evaluate the unpruned model first
    # eval_dataset = get_dataset(dataset_name, split="test")
    # tokenized_eval_data = get_tokenized_data(
    #     eval_dataset, tokenizer, dataset_name, max_length=2048, return_tensors=False
    # )
    # eval_hf_dataset = Dataset.from_list(tokenized_eval_data)
    # data_collator = DataCollatorWithPadding(
    #     tokenizer=tokenizer, padding="max_length", max_length=2048
    # )
    # dataloader = DataLoader(
    #     eval_hf_dataset,
    #     batch_size=8,
    #     collate_fn=data_collator,
    #     shuffle=False,
    # )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # avg_loss_original, perplexity_original = evaluate_model(model, dataloader, device, max_length=2048)

    # We now pass return_tensors=True to get_tokenized_data for the sampling part,
    # as the similarity check probably expects tensors.
    tokenized_dataset = get_tokenized_data(
        dataset, tokenizer, dataset_name, return_tensors=True
    )

    # Map pruning_type to prepare_calibration type
    calibration_type = "prototype"
    if pruning_type == "most_similar":
        calibration_type = "prototype"
    elif pruning_type == "most_dissimilar":
        calibration_type = "most_different"
    elif pruning_type == "most_similar_decoupled":
        calibration_type = "decoupled"
    elif pruning_type == "least_perplexity":
        calibration_type = "least_perplexity"
    elif pruning_type == "random":
        calibration_type = "random_sample"

    # Pick the calibration data from the dataset
    calibration_data_dicts = prepare_calibration(
        model=model,
        dataloader=[tokenized_dataset],
        nsamples=128,
        type=calibration_type,
        distance="flatten",
        save_calibration_distribution=False,
        model_name=model_name.replace("/", "-"),
        dataset_name=dataset_name,
        tokenizer=tokenizer,
    )
    log.info(f"Calibration data prepared: {len(calibration_data_dicts)} samples.")

    # Convert calibration data to format expected by oneshot
    # Assuming calibration_data_dicts returns dicts like {'input_ids': tensor(seq_len), ...}
    # We need to extract the Python list/array of IDs from the tensor for Dataset.from_list
    data_list = []
    for item in calibration_data_dicts:
        input_ids = item["input_ids"]
        attention_mask = item.get("attention_mask")

        # Convert tensor to list of integers
        if isinstance(input_ids, torch.Tensor):
            # Ensure it's on CPU and convert to list
            if input_ids.dim() == 2 and input_ids.shape[0] == 1:
                input_ids = input_ids.squeeze(0)
            input_ids = input_ids.cpu().numpy().tolist()

            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                if attention_mask.dim() == 2 and attention_mask.shape[0] == 1:
                    attention_mask = attention_mask.squeeze(0)
                attention_mask = attention_mask.cpu().numpy().tolist()

        data_dict = {"input_ids": input_ids}
        if attention_mask is not None:
            data_dict["attention_mask"] = attention_mask

        data_list.append(data_dict)

    calibration_dataset = Dataset.from_list(data_list)

    # Collect Wanda statistics before pruning
    wanda_analyzer = WandaAnalysis(model, pruning_type=pruning_type)
    wanda_analyzer.collect(
        DataLoader(
            calibration_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, max_length=128),
        )
    )
    wanda_analyzer.compute_scores()
    wanda_analyzer.plot(
        save_path=f"results/wanda_{pruning_type}_{dataset_name}.pdf", max_layers=20
    )
    # TODO continue and save the pruned model's weights
    exit(0)
    # Define Wanda recipe
    recipe = WandaPruningModifier(sparsity=0.5, mask_structure="0:0", targets="__ALL__")

    log.info("Starting Wanda pruning...")
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        # max_seq_length=128,
        # num_calibration_samples=len(calibration_data)
    )

    log.info("Pruning finished.")

    # Evaluate the pruned model
    log.info("Evaluating the pruned model...")
    eval_dataset = get_dataset(dataset_name, split="test")

    # Prepare the tokenized dataset for the DataLoader
    # NOTE: We pass return_tensors=False so the data collator handles the tensor conversion
    tokenized_eval_data = get_tokenized_data(
        eval_dataset, tokenizer, dataset_name, max_length=512, return_tensors=False
    )

    # Convert list of dicts to Hugging Face Dataset object for better integration
    eval_hf_dataset = Dataset.from_list(tokenized_eval_data)

    # Initialize the Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512)

    dataloader = DataLoader(
        eval_hf_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=data_collator,  # <-- FIX for TypeError
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Evaluate the model
    avg_loss, perplexity = evaluate_model(model, dataloader, device, max_length=512)
    log.info("Pruned model evaluation completed.")
    # Summarize results in a table
    table = PrettyTable()
    table.field_names = ["Model", "Avg Loss", "Perplexity", "Pruning", "Calibration"]
    # table.add_row([model_name, f"{avg_loss_original:.4f}", f"{perplexity_original:.2f}", "Original", "N/A"])
    table.add_row(
        [
            model_name,
            f"{avg_loss:.4f}",
            f"{perplexity:.2f}",
            "Wanda 0.5",
            "to