#!/bin/bash
set -e

MAX_JOBS=2          # Numero di job paralleli (GPU 2-3 per cola.sh)
GPU_OFFSET=2        # Le GPU 0-1 sono usate da eval.sh
declare -a PIDS=()  # Tracciamento dei PID in background

# Attende finché uno slot GPU non è libero e restituisce l'indice della GPU libera
acquire_gpu() {
    while true; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                # Slot i è libero
                unset PIDS[$i]
                echo "$i"
                return
            fi
        done
        # Tutti gli slot sono occupati - attendi un momento e riprova
        sleep 5
    done
}

# Lancia un job su uno specifico slot GPU in background
launch_job() {
    local GPU_ID=$1; shift
    local REAL_GPU=$((GPU_ID + GPU_OFFSET))
    CUDA_VISIBLE_DEVICES=$REAL_GPU python source/eval_cola.py "$@" &
    PIDS[$GPU_ID]=$!
}

DATASET_PREF="--datasets"
DATASETS=("winogrande" "arc_challenge" "boolq" "hellaswag" "openbookqa" "rte" "mmlu" "wmt14" "anli_r1" "svamp" "gsm8k" "pile" "wikitext" "c4" "winogrande arc_challenge boolq hellaswag openbookqa rte")
MODEL_PREF="--model"
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it")
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128)
COMPRESSION_PREF="--compression_type"
COMPRESSION_TYPES=("pruning" "quantization" "awq")
OUTPUT_CSV_PREF="--output_csv"
OUTPUT_CSV="results/cola_experiments.csv"

mkdir -p logs

TASK_ID=0
# Gerarchia: Model -> Dataset -> Sample -> Compression
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for NSAMPLES in "${NUM_SAMPLES[@]}"; do
            for COMPRESSION in "${COMPRESSION_TYPES[@]}"; do
                
                # Attendi uno slot GPU libero (si blocca se tutti i MAX_JOBS sono occupati)
                if [[ ${#PIDS[@]} -lt $MAX_JOBS ]]; then
                    # Trova il primo indice di GPU non utilizzato
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
                echo "Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES, Compression: $COMPRESSION"
                echo "================================================================"

                LOG="logs/cola_task${TASK_ID}_gpu${GPU_SLOT}.log"
                
                launch_job "$GPU_SLOT" \
                    $DATASET_PREF "$DATASET" \
                    $MODEL_PREF "$MODEL" \
                    $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                    $SPARSITY_PREFIX "$SPARSITY" \
                    $COMPRESSION_PREF "$COMPRESSION" \
                    $OUTPUT_CSV_PREF "$OUTPUT_CSV" \
                    > "$LOG" 2>&1
                
                TASK_ID=$((TASK_ID + 1))
            done
        done
    done
done

# Attendi il completamento di tutti i job in background rimanenti
echo "Waiting for all jobs to finish..."
wait
echo "All done."
nvidia-smi
