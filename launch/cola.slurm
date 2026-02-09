#!/bin/bash
#SBATCH --job-name=R50_v6_C
#SBATCH --output=logs/cola_%j.out
#SBATCH --error=logs/cola_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it
#SBATCH --array=0-107%4  # 3 models * 6 datasets * 3 samples * 2 compression types = 108 tasks

#conda activate prun

DATASET_PREF="--datasets"
DATASETS=("winogrande" "arc_challenge" "boolq" "hellaswag" "openbookqa" "rte")
MODEL_PREF="--model"
MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B")
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.5"
NUM_SAMPLES=(128 512 1024)
COMPRESSION_PREF="--compression_type"
COMPRESSION_TYPES=("pruning" "quantization")
OUTPUT_CSV_PREF="--output_csv"
OUTPUT_CSV="results/experiment_results_new.csv"

# Calculate indices based on SLURM_ARRAY_TASK_ID
TOTAL_DATASETS=${#DATASETS[@]}
TOTAL_MODELS=${#MODELS[@]}
TOTAL_SAMPLES=${#NUM_SAMPLES[@]}
TOTAL_COMPRESSION=${#COMPRESSION_TYPES[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

# Hierarchy: Model -> Dataset -> Sample -> Compression
COMPRESSION_IDX=$(( TASK_ID % TOTAL_COMPRESSION ))
REM_1=$(( TASK_ID / TOTAL_COMPRESSION ))
SAMPLES_IDX=$(( REM_1 % TOTAL_SAMPLES ))
REM_2=$(( REM_1 / TOTAL_SAMPLES ))
DATASET_IDX=$(( REM_2 % TOTAL_DATASETS ))
MODEL_IDX=$(( REM_2 / TOTAL_DATASETS ))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
NSAMPLES=${NUM_SAMPLES[$SAMPLES_IDX]}
COMPRESSION=${COMPRESSION_TYPES[$COMPRESSION_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID: Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES, Compression: $COMPRESSION"
# Run the task
python source/eval_cola.py $DATASET_PREF $DATASET $MODEL_PREF $MODEL $NUM_SAMPLES_PREFIX $NSAMPLES $SPARSITY_PREFIX $SPARSITY $COMPRESSION_PREF $COMPRESSION $OUTPUT_CSV_PREF $OUTPUT_CSV
