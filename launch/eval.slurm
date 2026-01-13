#!/bin/bash
#SBATCH --job-name=eval_prune
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100.80:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

# Run as an array: one array task per model (0..2)
# Outputs include the array task id for easier debugging
#SBATCH --array=0-2

DATASET_PREF="--datasets"
DATASETS=("winogrande gsm8k boolq" "commonsense_qa gsm8k boolq" "winogrande gsm8k boolq ds1000 race mawps wmt rte")
MODEL_PREF="--model"
MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B") #"microsoft/Phi-2"
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128 512 1024)
COMPRESSION_PREFIX="--compression_type"
COMPRESSION_TYPE="pruning"

# Select model based on the SLURM array task id
MODEL_INDEX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$MODEL_INDEX]}

for DATASET in "${DATASETS[@]}"; do
    for NSAMPLES in "${NUM_SAMPLES[@]}"; do
        echo "Running pruning for model: $MODEL (task $MODEL_INDEX) on dataset: $DATASET and samples: $NSAMPLES"
        srun python source/run_experiment.py $DATASET_PREF $DATASET $MODEL_PREF $MODEL $NUM_SAMPLES_PREFIX $NSAMPLES $SPARSITY_PREFIX $SPARSITY $COMPRESSION_PREFIX $COMPRESSION_TYPE
    done
done