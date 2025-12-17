#!/bin/bash
#SBATCH --job-name=prune_sequential
#SBATCH --output=logs/prune_%j.out
#SBATCH --error=logs/prune_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

#conda activate prun

DATASET_PREF="--datasets"
DATASETS=("winogrande" "gsm8k" "svamp" "boolq" "commonsense_qa" "ds1000" "race" "mawps" "wmt" "rte")
MODEL_PREF="--model"
MODELS=("microsoft/Phi-2" "google/gemma-2b" "Qwen/Qwen3-1.7B" "meta-llama/Llama-3.2-1B")
PRUNING_PREF="--pruning_type"
PRUNING_TYPES=("most_similar" "random" "most_dissimilar" "most_similar_decoupled" "least_perplexity")


for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for PRUNING_TYPE in "${PRUNING_TYPES[@]}"; do
            echo "Running pruning for model: $MODEL on dataset: $DATASET with pruning type: $PRUNING_TYPE"
            srun python source/prune.py $DATASET_PREF $DATASET $MODEL_PREF $MODEL $PRUNING_PREF $PRUNING_TYPE
        done
    done
done


echo "Starting plotting"
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "Plotting results for model: $MODEL on dataset: $DATASET"
        srun python plot/variance_plots.py --dataset $DATASET $MODEL_PREF $MODEL
    done
done 
