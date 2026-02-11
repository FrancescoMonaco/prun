import argparse
import os
import torch
import pandas as pd
import logging
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import get_dataset
from similarity_check import prepare_calibration
from prune import get_tokenized_data

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

def get_hidden_states(model, input_ids, device="cuda"):
    """
    Runs the model and returns hidden states for all layers.
    """
    model.eval()
    with torch.no_grad():
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).to(device)
        else:
            input_ids = input_ids.to(device)
            
        outputs = model(input_ids, output_hidden_states=True)
        # hidden_states is a tuple of (layer_0, layer_1, ..., layer_N)
        # Each layer tensor: (batch_size, seq_len, hidden_dim)
        
        # Move to CPU to save GPU memory
        hidden_states = [h.cpu() for h in outputs.hidden_states]
        
    return hidden_states

def compute_layer_similarity(h1_list, h2_list):
    """
    Computes average cosine similarity between two lists of hidden states per layer.
    """
    similarities = []
    for h1, h2 in zip(h1_list, h2_list):
        # Flatten: (batch * seq, hidden)
        h1_flat = h1.view(-1, h1.shape[-1])
        h2_flat = h2.view(-1, h2.shape[-1])
        
        # Cosine similarity
        # We process in chunks to avoid OOM if very large, but here it's fine
        sim = torch.nn.functional.cosine_similarity(h1_flat, h2_flat, dim=1)
        similarities.append(sim.mean().item())
        
    return similarities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-7b")
    parser.add_argument("--dataset", type=str, default="c4", help="Dataset for calibration")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results/flow_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Tokenizer
    log.info(f"Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Prepare Data
    log.info("Preparing data...")
    raw_dataset = get_dataset(args.dataset)
    if isinstance(raw_dataset, dict):
         dataset = raw_dataset.get("train", list(raw_dataset.values())[0])
    else:
        dataset = raw_dataset
        
    tokenized = get_tokenized_data(dataset, tokenizer, args.dataset, return_tensors=True)
    calib_dataloader = [Dataset.from_list(tokenized)]
    
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Calibration samples
    log.info("Sampling Random Calibration...")
    rand_calib = prepare_calibration(
        model=st_model, dataloader=calib_dataloader, nsamples=args.nsamples, 
        type="random_sample", tokenizer=tokenizer, dataset_name=args.dataset, model_name=args.model.replace("/", "-")
    )
    
    log.info("Sampling Unique Tokens Calibration...")
    unique_calib = prepare_calibration(
        model=st_model, dataloader=calib_dataloader, nsamples=args.nsamples, 
        type="unique_tokens", tokenizer=tokenizer, dataset_name=args.dataset, model_name=args.model.replace("/", "-")
    )

    # Probe Data (Use a small separate random set to test flow)
    # We take slightly larger nsamples just to have variety, like 16
    log.info("Sampling Probe Data...")
    probe_samples = prepare_calibration(
        model=st_model, dataloader=calib_dataloader, nsamples=16, 
        type="random_sample", tokenizer=tokenizer, dataset_name=args.dataset, model_name=args.model.replace("/", "-")
    )
    
    # Batch the probe samples
    probe_input_ids = torch.cat([torch.tensor(x['input_ids']).unsqueeze(0) for x in probe_samples], dim=0)

    # 2. Baseline Flow
    log.info("Loading ORIGINAL model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map=args.device, trust_remote_code=True)
    
    log.info("Computing Original Flow...")
    h_orig = get_hidden_states(model, probe_input_ids, device)
    
    # Save original state dict to disk or CPU to reload later? 
    # Actually, reloading from disk is safer/cleaner than deepcopying huge models if VRAM is tight.
    # But let's try CPU copy.
    log.info("Backing up model state...")
    orig_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    
    # 3. Random Pruning Flow
    log.info("Applying Random Pruning...")
    # Convert calibration to Dataset object for oneshot
    rand_ds = Dataset.from_list(rand_calib)
    
    recipe = WandaPruningModifier(sparsity=args.sparsity, mask_structure="0:0", targets="__ALL__")
    oneshot(model=model, dataset=rand_ds, recipe=recipe)
    
    log.info("Computing Random Pruned Flow...")
    # Ensure model is on the correct device after pruning
    model.to(device)
    h_rand = get_hidden_states(model, probe_input_ids, device)
    
    # Compute similarity immediately to save memory?
    sim_rand = compute_layer_similarity(h_orig, h_rand)
    
    # Cleanup to save memory
    del h_rand
    
    # 4. Unique Pruning Flow
    log.info("Reloading model form backup...")
    model.load_state_dict(orig_state_dict)
    model.to(device)
    
    log.info("Applying Unique Tokens Pruning...")
    unique_ds = Dataset.from_list(unique_calib)
    
    recipe = WandaPruningModifier(sparsity=args.sparsity, mask_structure="0:0", targets="__ALL__")
    oneshot(model=model, dataset=unique_ds, recipe=recipe)
    
    log.info("Computing Unique Pruned Flow...")
    model.to(device)
    h_unique = get_hidden_states(model, probe_input_ids, device)
    
    sim_unique = compute_layer_similarity(h_orig, h_unique)
    del h_unique
    
    # 5. Save Results
    results = []
    for i, (sr, su) in enumerate(zip(sim_rand, sim_unique)):
        results.append({
            "model": args.model,
            "dataset": args.dataset,
            "layer_idx": i,
            "cosine_sim_random": sr,
            "cosine_sim_unique": su
        })
        
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"flow_{args.model.replace('/', '-')}_{args.dataset}.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Results saved to {csv_path}")
    
    # 6. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['layer_idx'], df['cosine_sim_random'], label='Random Calibration', marker='o')
    plt.plot(df['layer_idx'], df['cosine_sim_unique'], label='Unique Tokens Calibration', marker='s')
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity to Original Model')
    plt.title(f'Information Flow Preservation (Probe: {args.dataset})\nModel: {args.model.split("/")[-1]}')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(args.output_dir, f"flow_plot_{args.model.replace('/', '-')}_{args.dataset}.png")
    plt.savefig(plot_path)
    log.info(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
