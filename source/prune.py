from similarity_check import *
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import get_dataset
from datasets import Dataset
import nltk
# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
import logging

FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
log = logging.getLogger(__name__)

def get_tokenized_data(dataset, tokenizer, max_length=128):
    processed_dataset = []
    for i in range(len(dataset)):
        item = dataset[i]
        # Adjust key based on dataset. Winogrande has 'sentence'.
        text = item.get('sentence', item.get('text', '')) 
        if not text and 'prompt' in item:
             text = item['prompt']
             
        encoded = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        processed_dataset.append((encoded['input_ids'],))
    return processed_dataset

if __name__ == "__main__":
    # Initialize wandb, keep offline 
    wandb.init(project="llm-pruning-wanda", name="wanda-pruning-example") #, mode="offline")
    # Set the model name and dataset
    model_name = "Qwen/Qwen1.5-1.8B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    log.info(f"Loaded model {model_name} for embedding extraction.")
    
    dataset_name = 'winogrande'
    dataset = get_dataset(dataset_name, split='train')
    log.info(f"Loaded dataset {dataset_name} with {len(dataset)} samples.")
    
    # Preprocess dataset for similarity check
    # We limit the pool to select from to save time, e.g. 1000 samples
    subset_size = 2000
    if len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))
        
    tokenized_dataset = get_tokenized_data(dataset, tokenizer)
    
    # Pick the calibration data from the dataset
    # use_embedding_for_sampling expects a list of datasets
    calibration_data_tuples = use_embedding_for_sampling(
        [tokenized_dataset], 
        model, 
        sample_per_dataset=128, 
        distance='flatten', 
        type='prototype', 
        save_calibration_distribution=False, 
        model_name=model_name.replace("/", "-"), 
        dataset_name=dataset_name
    )
    log.info(f"Calibration data prepared: {len(calibration_data_tuples)} samples.")
    
    # Convert calibration data to format expected by oneshot (list of dicts or just tensors)
    # oneshot with custom data usually expects an iterable that yields the inputs to the model.
    # Since we have (input_ids,), we can yield {'input_ids': ...}
    
    data_list = []
    for item in calibration_data_tuples:
        # item[0] is tensor of shape (1, seq_len)
        input_ids = item[0]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy().tolist()
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]
        data_list.append({'input_ids': input_ids})
        
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
    
    # Send wandb info online
    wandb.finish()
    
