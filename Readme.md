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
5. Download  and extract [example-data.zip](https://share.sjmd.dev/rocnet/example-data.zip) (approx 1.3GB), this is a set of source `.laz` files, a dataset which supports voxel grid resolutions up to 128, and a training run (including model weights) for a model that uses 64-grid inputs. It contains three subfolders:
   1. `laz` - a colletion of `.laz` files
   2. `dataset` - a dataset of tiles created from the `.laz` files
   3. `training` - a training config file, and an example training run with a set of model weights and training progress snapshots

To use the example scripts make sure that the virutal environment from step 3 above is active, and then invoke the train/test/examine scripts like this:

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