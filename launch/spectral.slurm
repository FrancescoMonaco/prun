#!/bin/bash
#SBATCH --job-name=spectral
#SBATCH --output=logs/spectral_%j_%a.out
#SBATCH --error=logs/spectral_%j_%a.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

# Run as an array for different models
#SBATCH --array=0-2

MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B")
MODEL_INDEX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$MODEL_INDEX]}

echo "Running Spectral Analysis for Model: $MODEL"

# Key optimisations already built into the script:
#  - GPU-accelerated SVD (10-100x vs CPU)
#  - Only key projection layers (not ALL linears)
#  - Single model load + CPU state-dict backup (no disk reloads)
# Use --all_layers to analyse every Linear layer (much slower).

srun python source/spectral_analysis.py \
    --model "$MODEL" \
    --datasets boolq winogrande hellaswag \
    --nsamples 128 \
    --sparsity 0.5 \
    --top_k 10 \
    --calibration_types random unique_tokens \
    --output_dir results/spectral_analysis
