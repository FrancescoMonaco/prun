#!/bin/bash
#SBATCH --job-name=prune_sequential
#SBATCH --output=logs/prune_%j.out
#SBATCH --error=logs/prune_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=medium
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=francescopio.monaco@unitn.it

#conda activate prun


echo "Starting job 1: most_similar"
srun python source/prune.py --pruning_type=most_similar

echo "Starting job 2: random"
srun python source/prune.py --pruning_type=random

echo "Starting job 3: most_dissimilar"
srun python source/prune.py --pruning_type=most_dissimilar

echo "Starting job 4: decoupled"
srun python source/prune.py --pruning_type=most_similar_decoupled

echo "Starting job 5: least_perplexity"
srun python source/prune.py --pruning_type=least_perplexity