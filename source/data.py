"""
Datasets
"""

import os
from pathlib import Path
from datasets import load_dataset


# Datasets to be used in the project
general_datasets = ["c4", "oscar", "redpajama", "pile"]
aritm_reasoning = ["gsm8k", "svamp", "mawps"]
nlu_inference = ["anli_r1", "esnli", "rte"]
commonsense_qa = ["boolq", "commonsense_qa", "race", "winogrande"]
translation = ["wmt14", "iwslt"]
coding = ["opc", "ds1000", "mbpp"]
mixed = ["c4", "gsm8k", "anli_r1", "boolq", "wmt14"]
all_datasets = (
    general_datasets
    + aritm_reasoning
    + nlu_inference
    + commonsense_qa
    + translation
    + coding
    + mixed
)

# Datasets should be in the datasets/ folder
DATASETS_PATH = Path(os.path.dirname(__file__)).parent / "datasets"
os.makedirs(DATASETS_PATH, exist_ok=True)

# Mapping from local name to HuggingFace dataset ID (and optional config)
DATASET_MAPPING = {
    "c4": ("allenai/c4", "en"),
    "oscar": ("oscar-corpus/OSCAR-2301", "en"),
    "redpajama": ("togethercomputer/RedPajama-Data-1T", None),
    "pile": ("monology/pile-uncopyrighted", None),
    "gsm8k": ("gsm8k", "main"),
    "svamp": ("ChilleD/SVAMP", None),
    "mawps": ("MU-NLPC/Calc-mawps", None),
    "anli_r1": ("facebook/anli", None),
    "esnli": ("esnli", None),
    "rte": ("super_glue", "rte"),
    "boolq": ("google/boolq", None),
    "commonsense_qa": ("commonsense_qa", None),
    "race": ("race", "all"),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "wmt14": ("wmt14", "de-en"),
    "iwslt": ("iwslt2017", "iwslt2017-en-de"),
    "opc": ("openai_humaneval", None),
    "ds1000": ("xlang-ai/DS-1000", None),
    "mbpp": ("mbpp", None),
}


def get_dataset(name, split=None):
    """
    Load a dataset by name. Downloads it to DATASETS_PATH if not present.

    Args:
        name (str): Name of the dataset (e.g., 'c4', 'gsm8k').
        split (str, optional): Split to load (e.g., 'train', 'validation').

    Returns:
        Dataset or DatasetDict: The loaded dataset.
    """
    if name not in DATASET_MAPPING:
        # Try loading directly if not in mapping
        hf_name = name
        config = None
    else:
        hf_name, config = DATASET_MAPPING[name]

    print(f"Loading dataset: {name} (HF: {hf_name}, Config: {config})")

    try:
        if config:
            dataset = load_dataset(
                hf_name, config, cache_dir=str(DATASETS_PATH), split=split
            )
        else:
            dataset = load_dataset(hf_name, cache_dir=str(DATASETS_PATH), split=split)

        return dataset
    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    for ds_name in ["gsm8k", "winogrande"]:
        dataset = get_dataset(ds_name, split="test")
        print(f"Loaded {ds_name} with {len(dataset)} examples.")
