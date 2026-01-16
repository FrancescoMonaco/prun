#!/bin/bash
#SBATCH --job-name=activations_diff
#SBATCH --output=logs/act_%j.out
#SBATCH --error=logs/act_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=medium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

python source/wanda_sampling_diff.py