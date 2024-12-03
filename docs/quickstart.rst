Quickstart
==========

These instructions are known compatible versions of python/torch/CUDA and open3d. To use other versions you need to look inside ``setup.sh`` or ``setup.bat``.

1. Install python 3.11 from here: https://www.python.org/downloads/
2. Install CUDA 11.8 from here: https://developer.nvidia.com/cuda-toolkit-archive
3. Get code ``git clone --depth 1 https://altitude.otago.ac.nz/rocnet/rocnet-example.git``
4. Install dependencies:  ``setup.bat`` (windows/cmd) or ``setup.sh`` (linux or windows/git-bash)
5. Download data from here: https://share.sjmd.dev/rocnet/example-data.zip (approx 1.3GB). It contains:

   1. ``laz`` - a colletion of ``.laz`` files
   2. ``dataset`` - a dataset of tiles created from the ``.laz`` files
   3. ``weights`` - a training config file, and an example training run with a set of model weights and training progress snapshots

6. Copy the ``data`` folder inside ``example-data.zip`` to ``rocnet-example/data``

To use the example scripts make sure that the virutal environment from step 3 above is active, and then invoke the train/test/examine scripts like this:

.. code-block:: bash

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
