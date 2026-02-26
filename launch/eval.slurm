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
#SBATCH --array=0-41%7

DATASET_PREF="--datasets"
DATASETS=("winogrande" "arc_challenge" "boolq" "hellaswag" "openbookqa" "rte" "winogrande arc_challenge boolq hellaswag openbookqa rte") 
MODEL_PREF="--model"
MODELS=("google/gemma-2-9b-it") #"microsoft/Phi-2" "Qwen/Qwen3-8B" "google/gemma-7b" "meta-llama/Llama-3.1-8B-Instruct"
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128)
COMPRESSION_PREFIX="--compression_type"
COMPRESSION_TYPE=("pruning" "quantization")
PRUNING_PREFIX="--pruning_types"
PRUNING_TYPES=("unique_tokens" "random" "most_similar" "least_perplexity" "distribution_matching" "words_dataset")

# Select model, dataset and pruning type based on the SLURM array task id
# total tasks = len(MODELS) * len(DATASETS) * len(PRUNING_TYPES) = 1 * 7 * 6 = 42
NUM_DATASETS=${#DATASETS[@]}
NUM_PRUNING=${#PRUNING_TYPES[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL_INDEX=$((TASK_ID / (NUM_DATASETS * NUM_PRUNING)))
REMAINING=$((TASK_ID % (NUM_DATASETS * NUM_PRUNING)))
DATASET_INDEX=$((REMAINING / NUM_PRUNING))
PRUNING_INDEX=$((REMAINING % NUM_PRUNING))

MODEL=${MODELS[$MODEL_INDEX]}
DATASET=${DATASETS[$DATASET_INDEX]}
P_TYPE=${PRUNING_TYPES[$PRUNING_INDEX]}

SUBTASK_ID=0
for COMP in "${COMPRESSION_TYPE[@]}"; do
    for NSAMPLES in "${NUM_SAMPLES[@]}"; do
        SUBTASK_ID=$((SUBTASK_ID + 1))
        echo "================================================================"
        echo "SUBTASK $SUBTASK_ID START: TaskID=$SLURM_ARRAY_TASK_ID, Model=$MODEL, Dataset=$DATASET"
        echo "Config: Comp=$COMP, NSamples=$NSAMPLES, Pruning=$P_TYPE"
        echo "================================================================"
        
        srun python source/run_experiment.py \
            $DATASET_PREF $DATASET \
            $MODEL_PREF "$MODEL" \
            $NUM_SAMPLES_PREFIX "$NSAMPLES" \
            $SPARSITY_PREFIX "$SPARSITY" \
            $COMPRESSION_PREFIX "$COMP" \
            $PRUNING_PREFIX "$P_TYPE"
        
        echo "SUBTASK $SUBTASK_ID FINISHED: $P_TYPE"
        echo "----------------------------------------------------------------"
    done
done