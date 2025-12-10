#!/bin/bash
#SBATCH --job-name=prune_sequential
#SBATCH --output=logs/prune_%j.out
#SBATCH --error=logs/prune_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=francescopio.monaco@unitn.it

#conda activate prun


echo "Starting job 1: most_similar"
srun python source/prune.py --pruning_type most_similar --datasets gsm8k svamp commonsense_qa ds1000 --model meta-llama/Llama-3.2-1B

echo "Starting job 2: random"
srun python source/prune.py --pruning_type random --datasets gsm8k svamp commonsense_qa ds1000 --model meta-llama/Llama-3.2-1B

echo "Starting job 3: most_dissimilar"
srun python source/prune.py --pruning_type most_dissimilar --datasets gsm8k svamp commonsense_qa ds1000 --model meta-llama/Llama-3.2-1B

echo "Starting job 4: decoupled"
srun python source/prune.py --pruning_type most_similar_decoupled --datasets gsm8k svamp commonsense_qa ds1000 --model meta-llama/Llama-3.2-1B

echo "Starting job 5: least_perplexity"
srun python source/prune.py --pruning_type least_perplexity --datasets gsm8k svamp commonsense_qa ds1000 --model meta-llama/Llama-3.2-1B