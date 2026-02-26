#!/bin/bash
set -e

MAX_JOBS=4          # number of parallel jobs (= number of GPUs)
declare -a PIDS=()  # track background PIDs
declare -a GPU_BUSY=() # track which GPU slot is in use

# Wait until a GPU slot is free, return the free GPU index
acquire_gpu() {
    while true; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                # slot i is free
                unset PIDS[$i]
                echo "$i"
                return
            fi
        done
        # All slots busy – wait a moment and retry
        sleep 5
    done
}

# Launch a job on a specific GPU slot, background it
launch_job() {
    local GPU_ID=$1; shift
    CUDA_VISIBLE_DEVICES=$GPU_ID python source/run_experiment.py "$@" &
    PIDS[$GPU_ID]=$!
}

DATASET_PREF="--datasets"
DATASETS=("winogrande" "arc_challenge" "boolq" "hellaswag" "openbookqa" "rte" "winogrande arc_challenge boolq hellaswag openbookqa rte")
MODEL_PREF="--model"
MODELS=("google/gemma-2-9b-it")
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128 512 1024)
COMPRESSION_PREFIX="--compression_type"
COMPRESSION_TYPE=("pruning" "quantization")
PRUNING_PREFIX="--pruning_types"
PRUNING_TYPES=("unique_tokens" "random" "most_similar" "least_perplexity" "distribution_matching" "words_dataset")

TASK_ID=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for P_TYPE in "${PRUNING_TYPES[@]}"; do
            SUBTASK_ID=0
            for COMP in "${COMPRESSION_TYPE[@]}"; do
                for NSAMPLES in "${NUM_SAMPLES[@]}"; do
                    SUBTASK_ID=$((SUBTASK_ID + 1))

                    # Wait for a free GPU slot (blocks if all 4 are busy)
                    if [[ ${#PIDS[@]} -lt $MAX_JOBS ]]; then
                        # Find the first unused GPU index
                        for ((g=0; g<MAX_JOBS; g++)); do
                            if [[ -z "${PIDS[$g]}" ]] || ! kill -0 "${PIDS[$g]}" 2>/dev/null; then
                                GPU_SLOT=$g
                                break
                            fi
                        done
                    else
                        GPU_SLOT=$(acquire_gpu)
                    fi

                    echo "================================================================"
                    echo "TASK $TASK_ID | SUBTASK $SUBTASK_ID -> GPU $GPU_SLOT"
                    echo "Model=$MODEL, Dataset=$DATASET, Comp=$COMP, NSamples=$NSAMPLES, Pruning=$P_TYPE"
                    echo "================================================================"

                    LOG="logs/eval_t${TASK_ID}_s${SUBTASK_ID}_gpu${GPU_SLOT}.log"
                    launch_job "$GPU_SLOT" \
                        $DATASET_PREF $DATASET \
                        $MODEL_PREF "$MODEL" \
                        $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                        $SPARSITY_PREFIX "$SPARSITY" \
                        $COMPRESSION_PREFIX "$COMP" \
                        $PRUNING_PREFIX "$P_TYPE" \
                        > "$LOG" 2>&1

                done
            done
            TASK_ID=$((TASK_ID + 1))
        done
    done
done

# Wait for all remaining background jobs to finish
echo "Waiting for all jobs to finish..."
wait
echo "All done."
