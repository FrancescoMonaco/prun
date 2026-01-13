#!/bin/bash
#SBATCH --job-name=prune_plot
#SBATCH --output=logs/prune_%j.out
#SBATCH --error=logs/prune_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --partition=short
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

#conda activate prun

DATASET_PREF="--dataset"
DATASETS=("winogrande" "gsm8k" "svamp" "boolq" "commonsense_qa" "ds1000" "race" "mawps" "wmt" "rte")
MODEL_PREF="--model"
MODELS=(  "meta-llama/Llama-3.2-1B" "Qwen/Qwen3-1.7B" "microsoft/Phi-2" "google/gemma-2b")
PRUNING_PREF="--pruning_type"
PRUNING_TYPES=("most_similar" "random" "most_dissimilar" "most_similar_decoupled" "least_perplexity" "zipf")



echo "Starting plotting"
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "Plotting results for model: $MODEL on dataset: $DATASET"
        srun python plot/variance_plots.py $DATASET_PREF $DATASET $MODEL_PREF $MODEL
    done
done 
