import transformers
from similarity_check import (
    use_embedding_for_sampling,
)
from eval import evaluate_model
import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from data import get_dataset
from datasets import Dataset

import matplotlib.pyplot as plt
# import matplotlib
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
import logging
from prettytable import PrettyTable
from pydantic import PrivateAttr
from typing import Dict, List
import numpy as np

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)


def get_input_norms(inp, module):
    # Logic adapted from wanda_sparsify.py to extract column norms per sample
    # inp shape: (batch, seq_len, hidden) or similar
    
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    
    # Handle batch dimension by iterating
    batch_norms = []
    
    batch_size = inp.shape[0]
    for i in range(batch_size):
        sample_inp = inp[i] # (seq_len, hidden)
        
        if isinstance(module, (torch.nn.Linear, transformers.Conv1D)):
            # Linear: (seq, hidden) -> (hidden, seq)
            # We want norm over seq dimension (dim=1 after transpose)
            # resulting in (hidden,) vector
            if len(sample_inp.shape) == 2:
                sample_inp = sample_inp.t()
            elif len(sample_inp.shape) == 1:
                 sample_inp = sample_inp.unsqueeze(1)

        # For Conv2d, logic is more complex, but Qwen uses Linear.
        # Assuming Linear for now as per Qwen architecture.
        
        sample_inp = sample_inp.float()
        # Norm over the sequence length (dim=1)
        norms = torch.norm(sample_inp, p=2, dim=1)
        batch_norms.append(norms.cpu())
        
    return batch_norms


class AnalysisWandaModifier(WandaPruningModifier):
    _layer_norms: Dict[str, List[torch.Tensor]] = PrivateAttr(default_factory=dict)

    def calibrate_module(self, module, args, _output):
        super().calibrate_module(module, args, _output)
        
        inp = args[0]
        norms_list = get_input_norms(inp, module)
        
        # Use module name as key if possible, or module object
        # self._module_names maps module -> name
        if module in self._module_names:
            name = self._module_names[module]
        else:
            name = str(module)
            
        if name not in self._layer_norms:
            self._layer_norms[name] = []
        
        self._layer_norms[name].extend(norms_list)

    def plot_stats(self, model, save_path="wanda_analysis.png"):
        log.info(f"Generating analysis plots to {save_path}...")
        
        # Filter layers that have stats
        layers = sorted(self._layer_norms.keys())
        if not layers:
            log.warning("No stats collected to plot.")
            return

        num_layers = len(layers)
        # 2 plots per layer (Mean, Std)
        cols = 4
        rows = (num_layers * 2 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), layout='constrained')
        axes = axes.flatten()
        
        for i, layer_name in enumerate(layers):
            norms_list = self._layer_norms[layer_name]
            # Stack norms: (num_samples, hidden_dim)
            norms_stack = torch.stack(norms_list)
            
            # Compute Mean and Std of norms across samples
            mean_norms = torch.mean(norms_stack, dim=0)
            std_norms = torch.std(norms_stack, dim=0)
            
            # Get the weight matrix
            # We need to find the module in the model
            module = None
            for name, mod in model.named_modules():
                if name == layer_name:
                    module = mod
                    break
            
            if module is None:
                continue
                
            W = module.weight.data.cpu().float()
            if isinstance(module, transformers.Conv1D):
                W = W.t()
            
            W_abs = torch.abs(W)
            
            # Compute Mean and Std of W_metric
            # W_metric = |W| * norms
            # Mean = |W| * Mean(norms)
            # Std = |W| * Std(norms)
            
            # Broadcast multiply: (out, in) * (in,)
            mean_W_metric = W_abs * mean_norms.unsqueeze(0)
            std_W_metric = W_abs * std_norms.unsqueeze(0)
            
            # Plot Mean
            ax_mean = axes[i * 2]
            im_mean = ax_mean.imshow(mean_W_metric.numpy(), aspect='auto', cmap='viridis')
            ax_mean.set_title(f"{layer_name} Mean W_metric")
            fig.colorbar(im_mean, ax=ax_mean)
            
            # Plot Std
            ax_std = axes[i * 2 + 1]
            im_std = ax_std.imshow(std_W_metric.numpy(), aspect='auto', cmap='magma')
            ax_std.set_title(f"{layer_name} Std W_metric")
            fig.colorbar(im_std, ax=ax_std)
            
        # Hide unused subplots
        for j in range(num_layers * 2, len(axes)):
            axes[j].axis('off')
            
        plt.savefig(save_path, format='pdf', dpi=400)
        plt.close()
        log.info("Analysis plots saved.")


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
            encoded = tokenizer.pad(
                encoded, padding="max_length", return_tensors="pt", max_length=max_length
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
    # Load model and tokenizer
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
    
    # Evaluate the unpruned model first
    eval_dataset = get_dataset(dataset_name, split="test")
    tokenized_eval_data = get_tokenized_data(
        eval_dataset, tokenizer, max_length=2048, return_tensors=False
    )
    eval_hf_dataset = Dataset.from_list(tokenized_eval_data)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=2048
    )
    dataloader = DataLoader(
        eval_hf_dataset,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_loss_original, perplexity_original = evaluate_model(model, dataloader, device, max_length=2048)

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
    # Assuming calibration_data_dicts returns dicts like {'input_ids': tensor(seq_len), ...}
    # We need to extract the Python list/array of IDs from the tensor for Dataset.from_list
    data_list = []
    for item in calibration_data_dicts:
        # item['input_ids'] is expected to be a tensor here from use_embedding_for_sampling
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

    # Define Wanda recipe
    # recipe = WandaPruningModifier(sparsity=0.5, mask_structure="0:0", targets="__ALL__")
    recipe = AnalysisWandaModifier(sparsity=0.5, mask_structure="0:0", targets="__ALL__")

    log.info("Starting Wanda pruning...")
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        # max_seq_length=128,
        # num_calibration_samples=len(calibration_data)
    )
    
    # Generate plots
    recipe.plot_stats(model, save_path="wanda_analysis.png")
    
    log.info("Pruning finished.")

    # Evaluate the pruned model
    log.info("Evaluating the pruned model...")
    eval_dataset = get_dataset(dataset_name, split="test")

    # Prepare the tokenized dataset for the DataLoader
    # NOTE: We pass return_tensors=False so the data collator handles the tensor conversion
    tokenized_eval_data = get_tokenized_data(
        eval_dataset, tokenizer, max_length=2048, return_tensors=False
    )

    # Convert list of dicts to Hugging Face Dataset object for better integration
    eval_hf_dataset = Dataset.from_list(tokenized_eval_data)

    # Initialize the Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=2048
    )

    dataloader = DataLoader(
        eval_hf_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=data_collator,  # <-- FIX for TypeError
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Evaluate the model
    avg_loss, perplexity = evaluate_model(model, dataloader, device, max_length=2048)
    log.info(f"Pruned model evaluation completed.")
    # Summarize results in a table
    table = PrettyTable()
    table.field_names = ["Model", "Avg Loss", "Perplexity", "Pruning", "Calibration"]
    table.add_row([model_name, f"{avg_loss_original:.4f}", f"{perplexity_original:.2f}", "Original", "N/A"])
    table.add_row([model_name, f"{avg_loss:.4f}", f"{perplexity:.2f}", "Wanda 0.5", "top 128 cosine"])
    print(table)
    # Send wandb info online
    # wandb.finish()
