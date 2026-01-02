#!/bin/bash
#SBATCH --job-name=eval_cola
#SBATCH --output=logs/cola_%j.out
#SBATCH --error=logs/cola_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it
#SBATCH --array=0-26  # 3 models * 3 datasets * 3 samples = 27 tasks

#conda activate prun

DATASET_PREF="--datasets"
DATASETS=("winogrande gsm8k boolq" "commonsense_qa gsm8k boolq" "winogrande gsm8k boolq ds1000 race mawps wmt rte")
MODEL_PREF="--model"
MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B") #"microsoft/Phi-2"
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128 512 1024)

# Calculate indices based on SLURM_ARRAY_TASK_ID
TOTAL_DATASETS=${#DATASETS[@]}
TOTAL_MODELS=${#MODELS[@]}
TOTAL_SAMPLES=${#NUM_SAMPLES[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID
SAMPLES_IDX=$(( TASK_ID % TOTAL_SAMPLES ))
DATASET_IDX=$(( (TASK_ID / TOTAL_SAMPLES) % TOTAL_DATASETS ))
MODEL_IDX=$(( TASK_ID / (TOTAL_SAMPLES * TOTAL_DATASETS) ))
MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
NSAMPLES=${NUM_SAMPLES[$SAMPLES_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES"
# Run the task
python source/eval_cola.py $DATASET_PREF $DATASET $MODEL_PREF $MODEL $NUM_SAMPLES_PREFIX $NSAMPLES $SPARSITY_PREFIX $SPARSITY