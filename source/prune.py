from similarity_check import (
    use_embedding_for_sampling,
)
from eval import evaluate_model
import torch
from torch.utils.data import DataLoader

# The DataCollatorWithPadding must be correctly imported
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from data import get_dataset
from datasets import Dataset

# import matplotlib.pyplot as plt
# import matplotlib
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
import logging

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
log = logging.getLogger(__name__)


# --- MODIFIED: get_tokenized_data ---
# This function is now used for *both* similarity sampling and evaluation.
# 1. It no longer returns PyTorch tensors (return_tensors="pt" removed).
# 2. It returns a dictionary with 'input_ids' and 'attention_mask' as Python lists.
# 3. It no longer does max_length padding (this is deferred to the Data Collator).
def get_tokenized_data(dataset, tokenizer, max_length=128, return_tensors=False):
    processed_dataset = []
    for item in dataset:
        # Adjust key based on dataset. Winogrande has 'sentence'.
        text = item.get("sentence", item.get("text", ""))
        if not text and "prompt" in item:
            text = item["prompt"]

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            # return_tensors="pt" is now optional/controlled by argument
        )

        # NOTE: We return the raw dict if we don't need tensors yet (for DataCollator)
        if return_tensors:
            # For similarity check, we might still need tensors in the loop
            encoded = tokenizer.pad(encoded, return_tensors="pt", max_length=max_length)
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
    # ... (model loading and initialization remains the same)
    model_name = "Qwen/Qwen3-1.7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loaded model {model_name} for embedding extraction.")

    dataset_name = "winogrande"
    dataset = get_dataset(dataset_name, split="train")
    log.info(f"Loaded dataset {dataset_name} with {len(dataset)} samples.")

    # Preprocess dataset for similarity check
    subset_size = 2000
    if len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))

    # --- MODIFIED: tokenized_dataset for sampling/pruning ---
    # We now pass return_tensors=True to get_tokenized_data for the sampling part,
    # as the similarity check probably expects tensors.
    tokenized_dataset = get_tokenized_data(dataset, tokenizer, return_tensors=True)

    # Pick the calibration data from the dataset
    calibration_data_dicts = (
        use_embedding_for_sampling(  # Renamed from _tuples to _dicts
            [tokenized_dataset],
            model,
            sample_per_dataset=128,
            distance="flatten",
            type="prototype",
            save_calibration_distribution=False,
            model_name=model_name.replace("/", "-"),
            dataset_name=dataset_name,
        )
    )
    log.info(f"Calibration data prepared: {len(calibration_data_dicts)} samples.")

    # Convert calibration data to format expected by oneshot
    # Assuming calibration_data_dicts returns dicts like {'input_ids': tensor(1, 128), 'attention_mask': tensor(1, 128)}
    # We need to extract the Python list/array of IDs from the tensor for Dataset.from_list
    data_list = []
    for item in calibration_data_dicts:
        # item['input_ids'] is expected to be a tensor here from use_embedding_for_sampling
        input_ids = item["input_ids"]
        attention_mask = item.get("attention_mask")

        # Convert tensor (1, seq_len) to list of integers
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.squeeze(0).cpu().numpy().tolist()
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.squeeze(0).cpu().numpy().tolist()

        data_dict = {"input_ids": input_ids}
        if attention_mask is not None:
            data_dict["attention_mask"] = attention_mask

        data_list.append(data_dict)

    calibration_dataset = Dataset.from_list(data_list)

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
        eval_dataset, tokenizer, return_tensors=False
    )

    # Convert list of dicts to Hugging Face Dataset object for better integration
    eval_hf_dataset = Dataset.from_list(tokenized_eval_data)

    # Initialize the Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=128
    )

    dataloader = DataLoader(
        eval_hf_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=data_collator,  # <-- FIX for TypeError
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Evaluate the model
    avg_loss, perplexity = evaluate_model(model, dataloader, device, max_length=128)
    log.info(f"Evaluation completed. Avg Loss: {avg_loss}, Perplexity: {perplexity}")

    # Send wandb info online
    # wandb.finish()
