#!/bin/bash
#SBATCH --job-name=prune_array
#SBATCH --output=logs/prune_%A_%a.out
#SBATCH --error=logs/prune_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --array=0-199  # 4 models * 10 datasets * 5 types = 200 tasks

# Define arrays
DATASETS=("winogrande" "gsm8k" "svamp" "boolq" "commonsense_qa" "ds1000" "race" "mawps" "wmt" "rte")
MODELS=("microsoft/Phi-2" "google/gemma-2b" "Qwen/Qwen1.5-1.8B" "meta-llama/Llama-3.2-1B")
PRUNING_TYPES=("most_similar" "random" "most_dissimilar" "most_similar_decoupled" "least_perplexity")

# Calculate indices based on SLURM_ARRAY_TASK_ID
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / (10 * 5)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (10 * 5)))
DATASET_IDX=$((REMAINDER / 5))
PRUNE_IDX=$((REMAINDER % 5))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
PRUNING_TYPE=${PRUNING_TYPES[$PRUNE_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: Model: $MODEL, Dataset: $DATASET, Type: $PRUNING_TYPE"

# Run the task
python source/prune.py --datasets $DATASET --model $MODEL --pruning_type $PRUNING_TYPE
