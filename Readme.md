# RocNet Example

This is a working example for training and evaluating a RocNet model from a LiDAR dataset.

![An example of RocNet compression](./media/rocnet-compression.png "asdf")

## Dependencies

Some prerequisites need specific versions, limited by Open3D and by the python versions supported by the cluster.:

- **Python 3.11**: Required by Open3D 0.18  
- CUDA (this example, and also `setup.[bat,sh]` assumes CUDA 11.8)
- `open3d==0.18.0`: Was the stable version during RocNet development
- `numpy==1.26.4`: Required for compatibility with Open3d 0.18 (you'll get some fun silent crashes if you use a newer version with Open3D 0.18)


## Quickstart

1. Install python 3.11 ([here](https://www.python.org/downloads/))
2. Install CUDA 11.8 ([here](https://developer.nvidia.com/cuda-toolkit-archive))
3. Acquire this repository `git clone --depth 1 https://altitude.otago.ac.nz/rocnet/rocnet-example.git`
4. Run `setup.bat` (windows/cmd) *or* `setup.sh` (linux or windows/git-bash)
5. Download [example-data.zip](https://share.sjmd.dev/rocnet/example-data.zip) (approx 1.3GB), this is a set of source `.laz` files, a dataset which supports voxel grid resolutions up to 128, and a training run (including model weights) for a model that uses 64-grid inputs. It contains three subfolders:
   1. `laz` - a colletion of `.laz` files
   2. `dataset` - a dataset of tiles created from the `.laz` files
   3. `weights` - a training config file, and an example training run with a set of model weights and training progress snapshots
6. Copy the `data` folder inside `example-data.zip` to `rocnet/example/data`

To use the example scripts make sure that the virutal environment from step 3 above is active, and then invoke the train/test/examine scripts like this:

```bash
# Plot loss curve of training run, print some info about the resulting model
python examine_training_run.py ./data/weights/

# Load some tiles from the test dataset, encode/decode them, print compression ratio
# and some meterics of lossiness, and visualise the original and recovered
# Use the N and B keys to cycle through the example tiles
python test_tile.py ./data/weights

# Load some tiles from the test dataset, encode/decode them, print compression ratio
# and some meterics of lossiness, and visualise the original and recovered
python test_file.py ./data/weights --visualise

# Start a new training run with the configuration in ./data/weights/train.toml
python train.py ./data/weights
```

## General Usage

Usage pattern for any script goes like this (using `train.py` as an example):

1. Activate the python environment
2. run `python train.py $TGT_WORK_DIR`.
3. If `$TGT_WORK_DIR/train.toml` exists then training run and model config is loaded from that file and a training run is started.
4. If `$TGT_WORK_DIR/train.toml` does not exist it will be created and populated with default values, the script will then exist and prompt you to modify the defaul values and re-run step 1.

### Config File Names and Defaults

The above training example is the same for the other scripts, however they load different config files depending on what they do

- `examine_dataset.py` expects `train.toml`
- `tile.py` expects `tile.toml` 
- `test_file.py`, `test_tile.py` and `examine_training_run.py` expect `test.toml`

Default values for config are described in the rocnet package, except for `tile.toml` which is defined in `tile.py`.

### Usage Documentation

Class, function, and module documentation is in the docstrings.

All scripts accept a `--help` argument which will provide brief invocation instructions.

### Starting from Scratch

These instructions will get you something like the example data, but perhaps with a larger input dataset for training and tiling.

1. Acquire LIDAR data (e.g. from [opentopography](https://opentopography.org/))
2. Run `python tile.py $TGT_OUT_DIR` to create a dataset in `$TGT_OUT_DIR`, the script will exit, and you should edit `$TGT_OUT_DIR/tile.toml`:
   1. `input_dir` should point to the folder containing the laz files acquired in step 1 (e.g. `./data/laz/`)
   2. `grid_dim` and `vox_size` should be chosen so that most of the scan fits within the height of `grid_dim` and `vox_size` should be chosen so that continuous surfaces produce continuous 'shells' of occupied voxels. `grid_dim` must be a power of two.
   3. Ensure that the relevant transforms are added (especially for smaller `.laz` scans)
   4. Set `clean: true` if you need to clean outliers and noisy points from the input data (e.g. if you didn't use the 'exclude noise' option when)
   See the note about `transforms` in `tile.py` an decide if you need to add any
3. Re-run `python tile.py $TGT_OUT_DIR` to create the tiled dataset (this will probably take some time.)
4. Run `python train.py $TGT_WORK_DIR`, edit `train.toml` so that:
   1. `dataset_path` to point to the dataset folder 
   2. `grid_dim` should be a power of two, and less than or equal to the dataset `grid_dim`
   3. You may need to change `max_samples`, and/or `batch_size` depending on your hardware, datset size, `grid_dim`, and model config (in the `[model]` section of the config file)
5. Re-run `python train.py $TGT_WORK_DIR` to start a training run, it'll create a `train_<TIMESTAMP>` folder which will contain a bunch of stuff, including the log file and snapshots of the model weights and model loss values during training.
   1. If you hit out-of-memory errors check the log file to see how much is being used, and modify `batch_size`, `max_samples`, or some of the model parameters to reduce memory consumption.
6. After training is finished, you can use `python examine_training_run.py $TGT_WORK_DIR` (repeating the usage pattern of editing `test.toml`) to see the loss graph, or
   - `python test_tile.py $TGT_WORK_DIR` to visualise individual original and encoded/recovered tiles
   - `python test_file.py $TGT_WORK_DIR --visualise` to compute some lossiness and compression ratio metrics, and also to visualise the original file and the encoded/recovered file (with the file(s) specified in `test.toml`)


## Train with HPC/SLURM


Set up the necessary software, pyenv, and get the dataset:

1. Get this repository
2. Check which versions of python and CUDA are installed, check that the pytorch `index_url` in `setup.sh` is correct for your version of CUDA
3. Run setup.sh to get all the prerequisites
4. Transfer a dataset to the cluster (`$DATASET_DIR`)

To create a rocnet `$TGT_WORK_DIR`, follow the General Usage instructions and edit the resulting `train.toml`.
Copy `slurm-template.sh` there, rename it to `slurm.sh` and fill in all placeholders denoted by {{double-curly-braces}}.
To start a training run `cd $TGT_WORK_DIR` and then `sbatch slurm.sh` which will produce a slurm log file at `slurm-{job-num}.out` in `$TGT_WORK_DIR`. 
Use `squeue --me` to see the status of the job, how long it's been running, etc.
