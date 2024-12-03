#!/bin/bash

# use sinfo to list the available partitions on the cluster
# use the `slurm-check-installed-cuda.sh` to see what's installed on the partition/node you're using

#SBATCH --account={{SLURM_USERNAME}}
#SBATCH --partition={{SLURM_TGT_PARTITION}}
#SBATCH --gpus-per-node=1
#SBATCH --mem=4GB
#SBATCH --time=1:00:30

repodir = "{{ABS_PATH_TO_ROCNET_EXAMPLE_REPO_DIR}}"

source "$repodir/.venv/bin/activate"
cd "$repodir"
echo "SLURM-START $(date)"
which python
python --version
exec python cluster/slurm-check-pytorch.py
exec python train.py {{abs-path-to-training-folder}}
echo "SLURM-DONE  $(date)"
