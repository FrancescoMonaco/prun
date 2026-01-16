import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import Dataset

from sentence_transformers import SentenceTransformer

from data import get_dataset
from similarity_check import prepare_calibration
from wanda_analysis import WandaAnalysis
from prune import get_tokenized_data

# Configure logging
FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Wanda Sampling Difference Analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name or path")
    parser.add_argument("--datasets", nargs="+", default=["winogrande"], help="Datasets for calibration")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of samples")
    parser.add_argument("--techniques", nargs="+", default=["random", "most_similar", "most_dissimilar", "zipf"], help="Sampling techniques to compare")
    parser.add_argument("--reference", type=str, default="random", help="Reference technique")
    parser.add_argument("--max_layers", type=int, default=5, help="Max layers to plot")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentence_trsf = SentenceTransformer("all-MiniLM-L12-v2", device=device)

    # Load and tokenize datasets (following prune.py pattern)
    log.info(f"Loading datasets {args.datasets}...")
    all_tokenized_data = []
    for d_name in args.datasets:
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            continue
            
        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            if "train" in raw_dataset.keys():
                dataset = raw_dataset["train"]
            elif "test" in raw_dataset.keys():
                dataset = raw_dataset["test"]
            else:
                dataset = raw_dataset[list(raw_dataset.keys())[0]]
        else:
            dataset = raw_dataset

        tokenized_data = get_tokenized_data(dataset, tokenizer, d_name, return_tensors=True)
        all_tokenized_data.extend(tokenized_data)

    analyzer = WandaAnalysis(model, pruning_type="wanda", device=device)

    # Ensure all techniques including reference are processed
    all_techs = set(args.techniques) | {args.reference}
    
    for tech in all_techs:
        log.info(f"==> Preparing calibration for technique: {tech}")
        
        # Map to internal calibration type
        calibration_type = "random_sample"
        if tech == "random": 
            calibration_type = "random_sample"
        elif tech == "most_similar": 
            calibration_type = "prototype"
        elif tech == "most_dissimilar": 
            calibration_type = "most_different"
        elif tech == "decoupled": 
            calibration_type = "decoupled"
        elif tech == "least_perplexity": 
            calibration_type = "least_perplexity"
        elif tech == "zipf": 
            calibration_type = "zipf"
        elif tech == "unique_tokens": 
            calibration_type = "unique_tokens"
        else: 
            calibration_type = tech # Fallback

        calibration_data_dicts = prepare_calibration(
            model=model if calibration_type == "least_perplexity" else sentence_trsf,
            dataloader=[all_tokenized_data],
            nsamples=args.nsamples,
            type=calibration_type,
            tokenizer=tokenizer,
            dataset_name="-".join(args.datasets),
            model_name=args.model.replace("/", "-")
        )
        
        # Process calibration data into a Dataset
        data_list = []
        for item in calibration_data_dicts:
            input_ids = item["input_ids"]
            attention_mask = item.get("attention_mask")
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() == 2 and input_ids.shape[0] == 1:
                    input_ids = input_ids.squeeze(0)
                input_ids = input_ids.cpu().numpy().tolist()
            if isinstance(attention_mask, torch.Tensor):
                if attention_mask.dim() == 2 and attention_mask.shape[0] == 1:
                    attention_mask = attention_mask.squeeze(0)
                attention_mask = attention_mask.cpu().numpy().tolist()
            
            d = {"input_ids": input_ids}
            if attention_mask is not None:
                d["attention_mask"] = attention_mask
            data_list.append(d)

        calibration_dataset = Dataset.from_list(data_list)
        calib_loader = DataLoader(
            calibration_dataset, 
            batch_size=1, # Reduced batch size for 8B model memory safety
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, max_length=128)
        )
        
        log.info(f"==> Collecting stats for {tech}...")
        analyzer.collect(calib_loader)
        analyzer.compute_scores()
        analyzer.compute_activations_stats()
        analyzer.store_results(tech)
        log.info(f"==> Handled {tech}")

        # Cleanup to prevent OOM
        del data_list
        del calibration_dataset
        del calib_loader
        torch.cuda.empty_cache()

    # Plot summary heatmaps
    summary_dir = os.path.join("results", "analysis", args.model.replace("/", "-"), "-".join(args.datasets))
    summary_path = os.path.join(summary_dir, f"wanda_summary_vs_{args.reference}.pdf")
    log.info(f"==> Plotting summary heatmaps to {summary_path}")
    analyzer.plot_summary_heatmaps(reference=args.reference, save_path=summary_path)

    analyzer.remove_hooks()
    log.info("Analysis complete.")

if __name__ == "__main__":
    main()
