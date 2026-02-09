#!/bin/bash
#SBATCH --job-name=R50_v6_?
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

# Run as an array: one array task per (model, dataset)
#SBATCH --array=0-17

DATASET_PREF="--datasets"
DATASETS=("winogrande" "arc_challenge" "boolq" "hellaswag" "openbookqa" "rte") 
MODEL_PREF="--model"
MODELS=("google/gemma-7b" "Qwen/Qwen3-8B" "meta-llama/Llama-3.2-3B") #"microsoft/Phi-2"
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.5"
NUM_SAMPLES=(128 512 1024)
COMPRESSION_PREFIX="--compression_type"
COMPRESSION_TYPE=("pruning" "quantization")
PRUNING_PREFIX="--pruning_types"
PRUNING_TYPES=("random" "unique_tokens" "most_similar" "least_perplexity" "distribution_matching")

# Select model and dataset based on the SLURM array task id
# total tasks = len(MODELS) * len(DATASETS) = 3 * 6 = 18
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_INDEX=$((TASK_ID / 6))
DATASET_INDEX=$((TASK_ID % 6))

MODEL=${MODELS[$MODEL_INDEX]}
DATASET=${DATASETS[$DATASET_INDEX]}

SUBTASK_ID=0
for COMP in "${COMPRESSION_TYPE[@]}"; do
    for NSAMPLES in "${NUM_SAMPLES[@]}"; do
        for P_TYPE in "${PRUNING_TYPES[@]}"; do
            SUBTASK_ID=$((SUBTASK_ID + 1))
            echo "================================================================"
            echo "SUBTASK $SUBTASK_ID START: TaskID=$SLURM_ARRAY_TASK_ID, Model=$MODEL, Dataset=$DATASET"
            echo "Config: Comp=$COMP, NSamples=$NSAMPLES, Pruning=$P_TYPE"
            echo "================================================================"
            
            srun python source/run_experiment.py \
                $DATASET_PREF "$DATASET" \
                $MODEL_PREF "$MODEL" \
                $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                $SPARSITY_PREFIX "$SPARSITY" \
                $COMPRESSION_PREFIX "$COMP" \
                $PRUNING_PREFIX "$P_TYPE"
            
            echo "SUBTASK $SUBTASK_ID FINISHED: $P_TYPE"
            echo "----------------------------------------------------------------"
        done
    done
done