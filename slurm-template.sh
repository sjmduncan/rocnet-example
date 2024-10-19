#!/bin/bash

# use sinfo to list the available partitions on the cluster

#SBATCH --account={SLURM_USERNAME}
#SBATCH --partition={TGT_SLURM_PARTITION}
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@2700
# 2700sec = 45mins (adjust to match your max epoch time)

source {abs-path-to-rocnet-example/.venv}
cd {abs-path-to-rocnet-example}
echo "SLURM-START $(date)"
exec python train.py {abs-path-to-training-folder}
echo "SLURM-DONE  $(date)"