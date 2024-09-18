#!/bin/bash
#SBATCH --job-name=bilevel_sweeps
#SBATCH --output=logs/out-%x-%A_%a.log  # A for Job ID, a for Array Task ID
#SBATCH --error=logs/err-%x-%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --mem=64G

set -x -e

source ~/.bashrc
conda activate zollm

cd /ihchomes/rezashkv/research/projects/bilevel_ZOFO/ || exit
# Gather all config files into an array
config_dir="./sweep/"
config_files=()

for task_dir in $config_dir/*; do
  # if SST2, WIC, WinoGrande, and WSC are not in the task_dir, skip
  if [[ $task_dir != *"SST2"* ]] && [[ $task_dir != *"WIC"* ]] && [[ $task_dir != *"WinoGrande"* ]] && [[ $task_dir != *"WSC"* ]]; then
    echo "Skipping $task_dir"
    continue
  fi
  for trainer_dir in $task_dir/*; do
    for config_file in $trainer_dir/*; do
      config_files+=($config_file)
    done
  done
done

# Calculate the number of config files
num_configs=${#config_files[@]}

# If SLURM_ARRAY_TASK_ID is not set, submit as an array job with the correct range
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  # Resubmit the script as an array job
  sbatch --array=0-$((num_configs - 1)) "$0"
  exit 0
fi

array_job_id=$((SLURM_JOB_ID - SLURM_ARRAY_TASK_ID - 1))

# Ensure the task ID is within range
if [ "$SLURM_ARRAY_TASK_ID" -ge "$num_configs" ]; then
  echo "SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of range ($num_configs)."
  exit 1
fi

# Use SLURM_ARRAY_TASK_ID to select the corresponding config
config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

# Run the sweep for the selected config
srun wandb sweep $config_file

# Extract wandb ID and run the agent
source /ihchomes/rezashkv/research/projects/bilevel_ZOFO/scripts/wandbid_from_slurm_id.sh
wandbid=$(extract_wandbid_id "$array_job_id")
wandb_id="reza_shkv/zo_bench/$wandbid"

srun wandb agent $wandb_id
