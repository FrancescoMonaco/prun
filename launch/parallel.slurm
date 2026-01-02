#!/bin/bash
#SBATCH --job-name=prune_array
#SBATCH --output=logs/prune_%A_%a.out
#SBATCH --error=logs/prune_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=medium
#SBATCH --array=0-89  # 3 models * 10 datasets * 1 types * 3 samples = 90 tasks

# Define arrays
DATASETS=("winogrande" "gsm8k" "svamp" "boolq" "commonsense_qa" "ds1000" "race" "mawps" "wmt" "rte")
MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B")
PRUNING_TYPES=( "distribution_matching_no_outliers" ) #"herding" "distribution_matching" "most_similar" "random" "most_dissimilar" "decoupled" "least_perplexity")
NUM_SAMPLES_PREFIX="--nsamples"
NUM_SAMPLES=(128 512 1024)

# Calculate indices based on SLURM_ARRAY_TASK_ID
TOTAL_DATASETS=${#DATASETS[@]}
TOTAL_MODELS=${#MODELS[@]}
TOTAL_PRUNING_TYPES=${#PRUNING_TYPES[@]}
TOTAL_SAMPLES=${#NUM_SAMPLES[@]}
TASK_ID=$SLURM_ARRAY_TASK_ID
SAMPLES_IDX=$(( TASK_ID % TOTAL_SAMPLES ))
PRUNE_IDX=$(( (TASK_ID / TOTAL_SAMPLES) % TOTAL_PRUNING_TYPES ))
DATASET_IDX=$(( (TASK_ID / (TOTAL_SAMPLES * TOTAL_PRUNING_TYPES)) % TOTAL_DATASETS ))
MODEL_IDX=$(( TASK_ID / (TOTAL_SAMPLES * TOTAL_PRUNING_TYPES * TOTAL_DATASETS) ))   

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
PRUNING_TYPE=${PRUNING_TYPES[$PRUNE_IDX]}
NSAMPLES=${NUM_SAMPLES[$SAMPLES_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: Model: $MODEL, Dataset: $DATASET, Type: $PRUNING_TYPE, Samples: $NSAMPLES"

# Run the task
python source/prune.py --datasets $DATASET --model $MODEL --pruning_type $PRUNING_TYPE --nsamples $NSAMPLES