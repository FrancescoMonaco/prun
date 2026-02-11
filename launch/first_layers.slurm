#!/bin/bash
#SBATCH --job-name=first_layers
#SBATCH --output=logs/first_layers_%j.out
#SBATCH --error=logs/first_layers_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

# Run as an array: one array task per model (0..2)
#SBATCH --array=0-2

MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B")
MODEL_INDEX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$MODEL_INDEX]}

# Settings for the analysis
# Comparing random samples vs unique tokens (random words) specifically on the first layers
NSAMPLES=128
SPARSITY=0.5
N_FIRST_LAYERS=5
DATASETS="c4 boolq winogrande arc_challenge"

echo "Running first layers analysis for model: $MODEL"
echo "Datasets: $DATASETS"
echo "First $N_FIRST_LAYERS layers"

srun python source/first_layers_analysis.py \
    --model "$MODEL" \
    --datasets $DATASETS \
    --nsamples $NSAMPLES \
    --sparsity $SPARSITY \
    --n_first_layers $N_FIRST_LAYERS \
    --output_csv "results/first_layers_overlap.csv" \
    --device cuda
