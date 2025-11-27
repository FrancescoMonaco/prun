#!/bin/bash
#SBATCH --job-name=similarity_check
#SBATCH --output=logs/similarity_check_%j.out
#SBATCH --error=logs/similarity_check_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=

srun conda activate prun && python source/similarity_check.py