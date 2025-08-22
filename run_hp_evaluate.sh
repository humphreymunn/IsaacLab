#!/usr/bin/env bash
#SBATCH --job-name=hp_sweep
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --account=OD-235390
#SBATCH --array=0-11

# Run the experiment for each array task
./isaaclab.sh -p batch_eval.py --task_index "${SLURM_ARRAY_TASK_ID}"