#!/bin/bash
#SBATCH --job-name=bilevel  # Specify a name for your job
#SBATCH --output=outputs/bilevel.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/bilevel.log # AUC_p_tuning_main_errors.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --mem=128G                  # Memory per node (32GB in this example)
#SBATCH --qos medium
# SBATCH --nodelist=cbcb28              # Specify the node to use

#SBATCH --qos huge-long
#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng


cd /fs/nexus-scratch/peiran/FO_Prompt_tuning_ZO_Fine_tuning/ZO-LLM/zo-bench


python3 run_bilevel.py