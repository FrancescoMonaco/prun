#!/bin/bash
set -e

MAX_JOBS=2          # Number of parallel jobs (GPU slots)
GPU_OFFSET=2        # Starting GPU index
declare -a PIDS=()  # Track background PIDs

# Wait until a GPU slot is free and return the free index
acquire_gpu() {
    while true; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset PIDS[$i]
                echo "$i"
                return
            fi
        done
        sleep 5
    done
}

# Launch a job on a specific GPU slot in background
launch_job() {
    local GPU_ID=$1; shift
    local REAL_GPU=$((GPU_ID + GPU_OFFSET))
    CUDA_VISIBLE_DEVICES=$REAL_GPU python "$@" &
    PIDS[$GPU_ID]=$!
}

# ---- Configuration ----
DATASET_PREF="--datasets"
DATASETS=(
    "winogrande"
    "arc_challenge"
    "boolq"
    "hellaswag"
    "openbookqa"
    "rte"
    "mmlu"
    "wmt14"
    "anli_r1"
    "svamp"
    "gsm8k"
    "pile"
    "wikitext"
    "c4"
    "winogrande arc_challenge boolq hellaswag openbookqa rte"
)
MODEL_PREF="--model"
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it")
NUM_SAMPLES_PREFIX="--nsamples"
NUM_SAMPLES=(128)
SPARSITY_PREFIX="--sparsity"
SPARSITIES=("0.25")
COMPRESSION_PREF="--compression_type"
COMPRESSION="2ssp"
PRUNING_PREFIX="--pruning_types"
# Calibration data selection strategies to test with 2SSP
PRUNING_TYPES=("unique_tokens" "words_dataset")
OUTPUT_CSV_PREF="--output_csv"
OUTPUT_CSV="results/2ssp_experiment_results.csv"

mkdir -p logs

TASK_ID=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for NSAMPLES in "${NUM_SAMPLES[@]}"; do
            for SPARSITY in "${SPARSITIES[@]}"; do
                for P_TYPE in "${PRUNING_TYPES[@]}"; do

                    # Acquire a free GPU slot (blocks if all busy)
                    if [[ ${#PIDS[@]} -lt $MAX_JOBS ]]; then
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
                    echo "TASK $TASK_ID -> GPU $GPU_SLOT"
                    echo "Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES, Sparsity: $SPARSITY, Calibration: $P_TYPE"
                    echo "================================================================"

                    LOG="logs/2ssp_task${TASK_ID}_gpu${GPU_SLOT}.log"

                    launch_job "$GPU_SLOT" \
                        source/run_experiment.py \
                        $DATASET_PREF $DATASET \
                        $MODEL_PREF "$MODEL" \
                        $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                        $SPARSITY_PREFIX "$SPARSITY" \
                        $COMPRESSION_PREF "$COMPRESSION" \
                        $PRUNING_PREFIX "$P_TYPE" \
                        $OUTPUT_CSV_PREF "$OUTPUT_CSV" \
                        > "$LOG" 2>&1

                    TASK_ID=$((TASK_ID + 1))
                done
            done
        done
    done
done

# =============================================
# Part 2: COLA calibration curation + 2SSP
# =============================================
COLA_OUTPUT_CSV="results/2ssp_cola_experiment_results.csv"

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for NSAMPLES in "${NUM_SAMPLES[@]}"; do
            for SPARSITY in "${SPARSITIES[@]}"; do

                # Acquire a free GPU slot (blocks if all busy)
                if [[ ${#PIDS[@]} -lt $MAX_JOBS ]]; then
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
                echo "TASK $TASK_ID -> GPU $GPU_SLOT  [COLA + 2SSP]"
                echo "Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES, Sparsity: $SPARSITY"
                echo "================================================================"

                LOG="logs/2ssp_cola_task${TASK_ID}_gpu${GPU_SLOT}.log"

                launch_job "$GPU_SLOT" \
                    source/eval_cola.py \
                    $DATASET_PREF $DATASET \
                    $MODEL_PREF "$MODEL" \
                    $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                    $SPARSITY_PREFIX "$SPARSITY" \
                    $COMPRESSION_PREF "$COMPRESSION" \
                    --pruning_types cola \
                    $OUTPUT_CSV_PREF "$COLA_OUTPUT_CSV" \
                    > "$LOG" 2>&1

                TASK_ID=$((TASK_ID + 1))
            done
        done
    done
done

# Wait for all remaining background jobs
echo "Waiting for all jobs to finish..."
wait
echo "All done."
nvidia-smi
