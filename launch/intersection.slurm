#!/bin/bash
#SBATCH --job-name=R50_v6_?
#SBATCH --output=logs/intersect_%j.out
#SBATCH --error=logs/intersect_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

# Run as an array: one array task per model (0..2)
# Outputs include the array task id for easier debugging
#SBATCH --array=0-2

MODEL_PREF="--model"
MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B") #"microsoft/Phi-2"

MODEL_INDEX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$MODEL_INDEX]}

srun python source/intersection_test.py $MODEL_PREF $MODEL
