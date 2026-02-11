#!/bin/bash
#SBATCH --job-name=flow_analysis
#SBATCH --output=logs/flow_%j.out
#SBATCH --error=logs/flow_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

# Run as an array samples for different models
#SBATCH --array=0-2

MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B")
MODEL_INDEX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$MODEL_INDEX]}

# Datasets to analyze
DATASETS=("boolq" "winogrande" "arc_challenge")

echo "Running Flow Analysis for Model: $MODEL"

for DATASET in "${DATASETS[@]}"; do
    echo "Processing Dataset: $DATASET"
    srun python source/flow_analysis.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --nsamples 128 \
        --sparsity 0.5 \
        --device cuda
done
