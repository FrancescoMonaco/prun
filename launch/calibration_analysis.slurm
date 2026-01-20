#!/bin/bash
#SBATCH --job-name=cal_analysis
#SBATCH --output=logs/cal_%j_%a.out
#SBATCH --error=logs/cal_%j_%a.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=medium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it
#SBATCH --array=0-2

MODELS=("Qwen/Qwen3-8B" "google/gemma-7b" "meta-llama/Llama-3.2-3B")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

python source/calibr_analysis.py --model $MODEL
