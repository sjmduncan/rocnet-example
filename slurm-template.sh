#!/bin/bash

# use sinfo to list the available partitions on the cluster

#SBATCH --account={{SLURM_USERNAME}}
#SBATCH --partition={{SLURM_TGT_PARTITION}}
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@300
# TERM signal sent 300sec = 5 mins before KILL
# Note that this might not always work as expected, for lots of boring SLURM/POSIX reasons

source {{abs-path-to-rocnet-example/.venv}}
cd {{abs-path-to-rocnet-example}}
echo "SLURM-START $(date)"
exec python train.py {{abs-path-to-training-folder}}
echo "SLURM-DONE  $(date)"