#!/bin/bash
#SBATCH --job-name=cal_analysis
#SBATCH --output=logs/cal_%j.out
#SBATCH --error=logs/cal_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --partition=medium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francescopio.monaco@unitn.it

python source/calibr_analysis.py