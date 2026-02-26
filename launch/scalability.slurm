#!/bin/bash
#SBATCH --job-name=R50_v6_?
#SBATCH --output=logs/scalability_%j.out
#SBATCH --error=logs/scalability_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

DATASET_PREFIX="--datasets"
DATASETS=("winogrande" "hellaswag" "arc_challenge")

for dataset in "${DATASETS[@]}"; do
    DATASET_PREFIX+=" $dataset"
done
srun python source/scalability_test.py $DATASET_PREFIX