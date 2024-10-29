# RocNet Example

An example of how to prepare a RocNet dataset, train a model, and evaluate the accuracy of the model

![An example of RocNet compression](./media/rocnet-compression.png "asdf")

## Setup

Some prerequisites need specific versions, limited by Open3D and by the python versions supported by the cluster.:

- **Python 3.11**:  
- `open3d==0.18.0`: Was the stable version during RocNet development
- `numpy==1.26.4`: Required for compatibility with Open3d 0.18 (you'll get some fun silent crashes if you use a newer version with Open3D 0.18)
- **Cuda 11.8**: Only required if you want to use the 80GB A100 nodes on the Otago Uni cluster (otherwise use whatever CUDA you want)


### Instructions

1. Install python 3.11 ([here](https://www.python.org/downloads/))
2. Install CUDA 11.8 ([here](https://developer.nvidia.com/cuda-toolkit-archive))
3. Run `setup.bat` (windows/cmd) *or* `setup.sh` (linux or windows/git-bash)
4. Download  and extract [example-data.zip](https://share.sjmd.dev/rocnet/example-data.zip), this contains two subfolders:
   1. `dataset` - a dataset with tiles and original `.laz` files for training and testing
   2. `training` - config files for training and testing, and an example of a training run

To use the example scripts make sure that the virutal environment from step 3 above is active, and then invoke the train/test/examine scripts like this (for more info on usage, invoke the scripts with the `--help` argument):

```bash
# Plot loss curve of training run, print some info about the resulting model
python examine_training_run.py ./data/weights/

# Load some tiles from the test dataset, encode/decode them, print compression ratio
# and some meterics of lossiness, and visualise the original and recovered
python test_tile.py ./data/weights

# Load some tiles from the test dataset, encode/decode them, print compression ratio
# and some meterics of lossiness, and visualise the original and recovered
python test_file.py ./data/weights --visualise

# Start a new training run with the configuration in ./data/weights/train.toml
python train.py ./data/weights
```

## Cluster Training

1. Follow the above setup instructions, but do it on the cluster
2. Modify `slurm-template.sh`:
   1. `SLURM_USERNAME` should be your login username on the cluster
   2. `SLURM_TGT_PARTITION` is the target partition (run `sinfo` on the cluster to get a list of partitions)
   3. `{abs-path-*}` should be absolute paths to the code and to the folder of training runs
3. Modify `train.toml` in the training run folder so that the `dataset` value is an absolute path
4. Run `sbatch slurm-template.sh`, run `squeue --me` and you should see the job listed. If it's running there will be a `R` in the `ST` column, and you should get:
   1. A new training run in the training run folder (a subfolder with the name `train_{TIMESTAMP}`)
   2. A log file in the working directory with the name `slurm-{JOBID}.out`