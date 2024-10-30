# RocNet Example

An example of how to prepare a RocNet dataset, train a model, and evaluate the accuracy of the model

![An example of RocNet compression](./media/rocnet-compression.png "asdf")

## Setup

Some prerequisites need specific versions, limited by Open3D and by the python versions supported by the cluster.:

- **Python 3.11**: Required by Open3D 0.18  
- **Cuda 11.8**: Only required if you want to use the 80GB A100 nodes on the Otago Uni cluster (otherwise use whatever CUDA you want)
- `open3d==0.18.0`: Was the stable version during RocNet development
- `numpy==1.26.4`: Required for compatibility with Open3d 0.18 (you'll get some fun silent crashes if you use a newer version with Open3D 0.18)


### Quickstart

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

Usage pattern for any script goes like this:

1. run `python SCRIPT.py PATH`. This will create a `.toml` config file in `PATH`, prompt you to edit the file (e.g. to set paths and variables), and then exit.
2. edit `PATH/*.toml` to set the approprate paths and values
3. re-run `python SCRIPT.py PATH` to actually run the script

The main .toml file names are `test.toml` and `train.toml`, but you can have others (e.g. `tiler.toml` for creating tiled datasets).

General usage instructions can be used by invoking a script with `-h` or `--help`, and the process of training a model goes a bit like this:

1. Acquire LIDAR data (e.g. from [opentopography](https://opentopography.org/))
   1. `data/laz` from the Quickstart is an example
2. Run `python tile.py PATH_OUT` where `PATH_OUT` is a directory where the tile dataset is created. If `PATH_OUT/tiler.toml` does not exist, the script will create it and prompt you to edit it.
   1. `input_dir` should point to the folder containing the laz files acquired in step 1 (e.g. `./data/laz/`)
   2. `grid_dim` and `vox_size` should be chosen so that most of the scan fits within the height of `grid_dim` and `vox_size` should be chosen so that continuous surfaces produce continuous 'shells of occupied voxels
   3. Ensure that the relevant transforms are added (especially for smaller `.laz` scans)
   4. `clean` to ensure that the pointclouds are cleaned before tiling
3. Create a dataset of 'tiles' which can be efficiently loaded and used to train a RocNet model
4. Run `python train.py PATH`, which will create `PATH` and `PATH/train.toml` with default values, the script will then exit and prompt you to edit the newly created train.toml, which at a minimum needs
   1. `dataset_path` to point to the dataset folder 
   2. `grid_dim` should be a power of two to 
5. Re-run `python train.py PATH` to start a training run
6. After it's finished, use the `test_*` and `examine_*` scripts to evaluate the result.