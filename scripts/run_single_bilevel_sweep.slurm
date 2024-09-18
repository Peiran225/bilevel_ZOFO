#!/bin/bash
#SBATCH --job-name=bilevel_sweeps
#SBATCH --output=logs/out-%x-%A_%a.log  # A for Job ID, a for Array Task ID
#SBATCH --error=logs/err-%x-%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-sxm:1
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --mem=64G

set -x -e

source ~/.bashrc
conda activate zollm

cd /ihchomes/rezashkv/research/projects/bilevel_ZOFO/ || exit
wandb agent "reza_shkv/zo_bench/d1hu9sst"
