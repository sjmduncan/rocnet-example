import argparse
import glob
import os.path as pth

import numpy as np
import rocnet.tiler as tiler
from rocnet.utils import write_file

import utils

DEFAULT_CONFIG = {
    "input_dir": "../rocnet.data/raw/test",
    "grid_dim": 256,
    "vox_size": 0.25,
    "train_fraction": 0.85,
    "clean": True,
    "save_intermediate": True,
    # Transforms to apply to the .laz files before tiling, of the form [tx, ty, tz, rz] which is a 3D translation and a rotation (radians) about the vertical axis
    # Note that the vertical axis is the one with the smallest extent
    # Choose translation values values that are
    # 1. not integer multiples of the voxel size
    # 2. not integer multiples of the grid_dim
    # 3. not integer multiples the leaf_dim you expect to use in the RocNet model (this is probably 32)
    "transforms": [[0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.0], [0.6, 0.4, 0.7, 0.0]],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test_tile.py", description="Evaluate and visualise the per-tile performance and lossiness of one or more models")
    parser.add_argument("folder", help="Folder where the tiled dataset wll be created")
    args = parser.parse_args()
    run = utils.Run(args.folder, "tiler", "tiler", False, DEFAULT_CONFIG)

    files = glob.glob(pth.join(run.cfg.input_dir, "*la[sz]"))
    n_train = int(run.cfg.train_fraction * len(files))
    n_test = len(files) - n_train

    train_mask = np.zeros(len(files))
    train_mask[np.random.choice(range(len(train_mask)), n_train, replace=False)] = 1

    files_types = ["train" if t else "test" for t in train_mask]

    [tiler.tile_file(f, run.dir, run.cfg.grid_dim, run.cfg.vox_size, run.cfg.transforms, t, 0, run.cfg.clean, run.cfg.save_intermediate) for f, t in list(zip(files, files_types))]

    meta = tiler.DEFAULT_METADATA | run.cfg
    meta["files_in"] = list(zip(files_types, [pth.split(f)[1] for f in files]))
    write_file(pth.join(run.dir, "meta.toml"), meta, overwrite_ok=True)
