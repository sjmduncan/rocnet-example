#!/bin/bash

# use sinfo to list the available partitions on the cluster

#SBATCH --account={{SLURM_USERNAME}}
#SBATCH --partition={{SLURM_TGT_PARTITION}}
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@300
# TERM signal sent 300sec = 5 mins before KILL
# Note that this might not always work as expected, for lots of boring SLURM/POSIX reasons

workdir="$(pwd)"
repodir = "{{ABS_PATH_TO_ROCNET_EXAMPLE_REPO_DIR}}"

source "$repodir/.venv/bin/activate"
cd "$repodir"
echo "SLURM start $(date)"
echo "      workdir $workdir"
exec python train.py "$workdir" --resume-from "$workdir"
echo "SLURM done  $(date)"