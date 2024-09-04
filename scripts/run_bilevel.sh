#!/bin/bash
#SBATCH --job-name=true_bilevel  # Specify a name for your job
#SBATCH --output=outputs/true_bilevel.log       # AUC_converge.log  Specify the output log file
#SBATCH --error=errors/true_bilevel.log        # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1       # Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --time=48:00:00           # Maximum execution time (HH:MM:SS)
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --mem=128G                  # Memory per node (32GB in this example)
#SBATCH --qos medium
# SBATCH --nodelist=cbcb28              # Specify the node to use

#SBATCH --qos huge-long
#SBATCH --account cbcb-heng
#SBATCH --partition cbcb-heng


cd /fs/nexus-scratch/peiran/bilevel_ZOFO
# --max_steps=20000
for LEARNING_RATE in 1e-3
do 
python3 run_bilevel.py --prompt_tuning=True --num_virtual_tokens=10 --prompt_init_by_real_tokens=True \
                       --model_name="facebook/opt-1.3b" --task_name="SST2" \
                       --learning_rate=$LEARNING_RATE --weight_decay=0 \
                       --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end=True \
                       --evaluation_strategy="steps" --save_strategy="steps" --save_total_limit=1 \
                       --eval_steps=1 --max_steps=100 --logging_steps=1 --num_eval=1000 \
                       --num_train=1000 --num_dev=500 \
                       --perturbation_mode="two_side" --trainer="bilevel_minimax2" \
                       --optimizer="adam" --train_set_seed=0 --lr_scheduler_type="constant" \
                       --eval_steps=1 --save_steps=500 --Lambda=1000 \
                       --lower_level_num_train_steps=1 --output_dir="ouput/opt_bilevel" \
                       --upper_learning_rate=1e-4 --upper_optimizer="adam"
done
                       