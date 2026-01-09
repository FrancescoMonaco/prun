#!/bin/bash
#SBATCH --job-name=plot_distribution
#SBATCH --output=logs/plot_distribution_%j.out
#SBATCH --error=logs/plot_distribution_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it
#SBATCH --array=0-8  # 1 models * 9 datasets = 9 jobs

DATASET_PREF="--dataset"
DATASETS=("winogrande" "gsm8k" "boolq" "commonsense_qa"  "ds1000" "race" "mawps" "wmt" "rte")
MODEL_PREF="--model"
MODELS=("google/gemma-7b") # "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}]}

srun python3 plot/token_distribution.py \
    $DATASET_PREF $DATASET \
    $MODEL_PREF $MODEL \
    --output_dir plots/token_distributions

