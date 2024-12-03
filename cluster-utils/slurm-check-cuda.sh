#!/bin/bash

# use sinfo to list the available partitions 

#SBATCH --account={{SLURM_USERNAME}}
#SBATCH --partition={{SLURM_TGT_PARTITION}}
#SBATCH --gpus-per-node=1
#SBATCH --mem=4GB
#SBATCH --time=0:00:30

echo "SLURM-START $(date)"
nvidia-smi
echo "SLURM-DONE  $(date)"