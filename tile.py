import argparse
import glob
import os.path as pth

import numpy as np

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


def _get_args():
    parser = argparse.ArgumentParser(prog="tile.py", description="Convert one or more .laz files to a set of .npy files for efficient loading during training")
    parser.add_argument("folder", help="Folder where the .laz files are")
    parser.add_argument("--output-folder", help="Where to put the output ( train & test .npy files, meta.toml, and tile.toml)")
    parser.add_argument("--grid-dim", help="Grid dimension of the voxel grid contained in one octree", type=int, default=None)
    parser.add_argument("--leaf-dim", help="Size of the octree leaf block", type=int, default=None)
    parser.add_argument("--vox-size", help="Voxel size to use when quantising the intput point cloud", type=int, default=None)
    parser.add_argument("--train-fraction", help="Percentage of samples to use for training", type=float, default=None)
    parser.add_argument("--visualise-only", help="Render the tiling of each file, but don't create any output in ", action="store_true", default=True)
    return parser


if __name__ == "__main__":
    parser = _get_args()
    args = parser.parse_args()
    run = utils.Run(args.folder, "tiler", "tiler", False, DEFAULT_CONFIG)

    files = glob.glob(pth.join(run.cfg.input_dir, "*la[sz]"))
    n_train = int(run.cfg.train_fraction * len(files))
    n_test = len(files) - n_train

    train_mask = np.zeros(len(files))
    train_mask[np.random.choice(range(len(train_mask)), n_train, replace=False)] = 1

    files_types = ["train" if t else "test" for t in train_mask]

    # [tiler.tile_file(f, run.dir, run.cfg.grid_dim, run.cfg.vox_size, run.cfg.transforms, t, 0, run.cfg.clean, run.cfg.save_intermediate) for f, t in list(zip(files, files_types))]

    # meta = tiler.DEFAULT_METADATA | run.cfg
    # meta["files_in"] = list(zip(files_types, [pth.split(f)[1] for f in files]))
    # write_file(pth.join(run.dir, "meta.toml"), meta, overwrite_ok=True)
