import os
from datasets import load_dataset, Dataset

# Create directories if they don't exist
os.makedirs("datasets/c4", exist_ok=True)
os.makedirs("datasets/pile", exist_ok=True)

def download_small_subset(dataset_id, config_name, split, num_samples, output_path, skip_samples=0):
    print(f"Streaming {num_samples} examples from {dataset_id} (split: '{split}', skipping: {skip_samples})...")
    
    # 1. Load in streaming mode
    ds_stream = load_dataset(dataset_id, config_name, split=split, streaming=True)
    
    # 2. Skip rows if we need to artificially separate train/test from a single split
    if skip_samples > 0:
        ds_stream = ds_stream.skip(skip_samples)
        
    # 3. Take only the requested number of samples
    small_ds_iterable = ds_stream.take(num_samples)
    
    # 4. Convert the iterable to a standard Dataset in memory
    small_ds = Dataset.from_generator(lambda: (yield from small_ds_iterable))
    
    # 5. Save to disk
    small_ds.to_parquet(output_path)
    print(f"Successfully saved to {output_path}!\n")

if __name__ == "__main__":
    # Define how many examples you want
    TRAIN_SIZE = 50000
    TEST_SIZE = 5000

    # --- C4 ---
    # C4 officially has 'train' and 'validation' splits
    download_small_subset("allenai/c4", "en", "train", TRAIN_SIZE, "datasets/c4/c4_train_subset.parquet")
    download_small_subset("allenai/c4", "en", "validation", TEST_SIZE, "datasets/c4/c4_validation_subset.parquet")

    # --- The Pile ---
    # Pile-uncopyrighted ONLY has a 'train' split.
    # We pull the first TRAIN_SIZE for training, and skip those to get the TEST_SIZE.
    download_small_subset("monology/pile-uncopyrighted", None, "train", TRAIN_SIZE, "datasets/pile/pile_train_subset.parquet")
    download_small_subset("monology/pile-uncopyrighted", None, "train", TEST_SIZE, "datasets/pile/pile_test_subset.parquet", skip_samples=TRAIN_SIZE)