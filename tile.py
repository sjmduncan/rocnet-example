"""Process a set of point cloud files to produce a set of relatively compact .npy files which can be loaded quickly for training and testing"""

import argparse
import copy
import glob
import logging
import os
import os.path as pth
import sys

import laspy as lp
import numpy as np
import open3d as o3d
import toml
from rocnet import utils as rutil
from rocnet.dataset import DEFAULT_METADATA

import utils

logger = logging.getLogger(__name__)
log_handler_stdout = logging.StreamHandler(sys.stdout)
logger.addHandler(log_handler_stdout)
logger.setLevel(level=logging.INFO)

DEFAULT_CONFIG = {
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
    # 3. not integer multiples the leaf_dim you expect to use in the RocNet model
    "transforms": [[0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.0], [0.6, 0.4, 0.7, 0.0]],
}


def tile_filename(bottom_left: np._typing.ArrayLike, suffix: str = "", ext: str = ".npy") -> str:
    """Create a filename of the form x_y_z_{suffix}{ext} where x,y,z are coords of bottom_left"""
    return f"{bottom_left[0]}_{bottom_left[1]}_{bottom_left[2]}_{suffix}{ext}"


def parse_tile_filename(file_path: str) -> np.array:
    """parse the file name of the form x_y_z_* to retrieve the coord. Raises ValueError if filename can't be parsed"""
    try:
        return np.float64(pth.splitext(pth.basename(file_path))[0].split("_")[:3])
    except ValueError:
        raise ValueError(f"file_path could not be parsed to 3D vector: {file_path}")


def laz_to_points(path: str, clean: bool, save_intermediate: bool) -> o3d.geometry.PointCloud:
    """Load a .laz file, optionally clean outliers and if cleaning optionally save the cleaned file for future use

    Load a .laz file as a point cloud, optionally perform statistical outlier removal.
    If clean=true and save_intermediate=true then the cleaned point cloud is saved with the
    file name '{path}.clean.ply' to avoid having to re-clean the pointcloud in the future.

    path: .laz file to load
    clean: enable outlier cleaning
    save_intermediate:
    """

    cleaned_pts_path = f"{path}.clean.ply"
    if clean and pth.exists(cleaned_pts_path):
        return o3d.io.read_point_cloud(cleaned_pts_path)
    else:
        laz = lp.read(path)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(laz.xyz))
        if clean:
            _, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            pc = pc.select_by_index(ind)
            if save_intermediate:
                o3d.io.write_point_cloud(cleaned_pts_path, pc)

        return pc


def pc_to_tiles(pts: np.array, vox_size: float, tile_grid_dim: int):
    """Quantise and tile a pointcloud to uint8 tile occpancy grids

    Quantise and deduplicate a pointcloud to vox_size, divide the result into cube-shaped tiles
    of size tile_grid_dim in voxels.

    pts: array of points
    vox_size: voxel size, vox_index = pts[idx] // vox_size
    tile_grid_dim: size of each tile in voxels, must be 64, 128, or 256
    shift: shift: 3D shift vector to apply to the voxel grid before tiling, array of int
    returns: list(corners, tiles)
              corners - grid indices of the bottom-left corners of the tiles (array, dtype=int)
              tiles - indices of occupied voxels for this tile (array dtype=uint8)
    """
    assert tile_grid_dim in [64, 128, 256]
    grid_pts = np.unique((pts // vox_size).astype(int), axis=0)
    tile_stack_idx = (grid_pts[:, :2] // tile_grid_dim).astype(int)
    tile_stack_corners = np.unique(tile_stack_idx, axis=0)
    tiles = [grid_pts[np.all(tile_stack_idx == corner, axis=1)] for corner in tile_stack_corners]

    tile_bottoms = [np.min(t[:, 2]) for t in tiles]
    corners_grid = [np.concatenate([tile_grid_dim * z[0], [z[1]]]) for z in zip(tile_stack_corners, tile_bottoms)]
    tiles = [z[0] - z[1] for z in zip(tiles, corners_grid)]
    corners_world = [vox_size * c for c in corners_grid]
    ## FIXME: deal with tall tiles properly.
    tiles = [t[t[:, 2] < tile_grid_dim] for t in tiles]

    pmax = np.max([np.max(t) for t in tiles])
    pmin = np.min([np.min(t) for t in tiles])
    assert pmax < tile_grid_dim and pmin >= 0

    return list(zip(corners_world, [t.astype("uint8") for t in tiles]))


def tiles_folder_name(path, vox_sz, grid_dim, shift=[0, 0, 0, 0]):
    return f"{path}_{vox_sz}v_{grid_dim}t_{shift[0]}_{shift[1]}_{shift[2]}_{shift[3]}_tiles"


def load_tile_as_pt_array(file_path, grid_dim, scale: int = None):
    """Load a pre-quantised tile created by tiler.ipynb, check that it fits within grid_dim. Returns the bottom-left corner (meters), and the grid indices of the occupied voxels in the tile"""
    if not pth.exists(file_path):
        raise FileNotFoundError(f"File doesn't exists: {file_path}")
    pts = np.load(file_path, allow_pickle=True)
    if scale is not None:
        pts = np.unique((pts * scale) // 1, axis=0).astype("int")
    return pts[pts[:, 2] < grid_dim]


def points_to_tiles(pts: np.array, tiled_pts_path: str, save_intermediate: bool, vox_sz: float, grid_dim: int) -> tuple[list, list]:
    """Load/clean a .laz file, return a list of 'tiles' and their bottom-left corners

    Invokes laz_to_points with path and clean, scales the resulting cloud by 1/vox_sz
    and divides the scaled cloud into cube-shaped 'tiles' with edge lengths of grid_dim.

    To avoid re-computing tiles (e.g. to re-run the tile filter, or to res-sort train/test subsets)
    the the un-sorted tiles can be saved in '{path}_{vox_sz}v_{grid_dim}t_tiles/' with file names
    computed by tile_filename.

    path: the .laz file
    clean: clean flag to pass to laz_to_points
    vox_sz: the voxel size, which should match the point spacing of the .laz file
    grid_dim: the edge length of a cube-shaped tile
    shift: 3D shift vector to apply to the voxel grid before tiling, array of int
    """

    if pth.exists(tiled_pts_path):
        files = glob.glob(pth.join(tiled_pts_path, "*.npy"))
        tile_pts = [load_tile_as_pt_array(f, grid_dim) for f in files]
        tile_bl = [parse_tile_filename(f) for f in files]
        tileset = list(zip(tile_bl, tile_pts))
    else:
        tileset = pc_to_tiles(pts, vox_sz, grid_dim)
        if save_intermediate:
            os.makedirs(tiled_pts_path, exist_ok=True)
            [save_voxel_tile(pth.join(tiled_pts_path, tile_filename(t[0], ext=".npy")), t[1]) for t in tileset]

    return tileset


def sort_tiles(tileset, test_fraction: float, smallest: int):
    """Sort tileset into training, testing/validation, and small-tile subsets

    First small tiles (those with only very few points) are excluded from the list
    The remaining good tiles are randomly split into training and testing, 'test_fraction'
    fraction of the tiles selected for testing

    tileset: list of (corner, tile) tuples
    test_ratio: fraction of non-small tiles to use for testing. Probably 0.15 or 0.2
    smallest: min number of points for non-small tile
    returns: train, test, small (are lists of (corner, tile))"""

    counted = list(zip([np.prod(t[1].shape) for t in tileset], tileset))

    small = [t[1] for t in counted if t[0] <= smallest]
    notsmall = [t[1] for t in counted if t[0] > smallest]
    n_test = int(test_fraction * len(notsmall))

    test_sel = np.zeros(len(notsmall), dtype=np.uint8)
    test_sel[np.random.choice(range(len(notsmall)), size=n_test, replace=False)] = 1

    train_tiles = [t for idx, t in enumerate(notsmall) if test_sel[idx] == 0]
    test_tiles = [t for idx, t in enumerate(notsmall) if test_sel[idx] == 1]

    return train_tiles, test_tiles, small


def save_voxel_tile(path, pts):
    pts_u8 = np.array(pts).astype("uint8")
    with open(path, "wb") as tile_file:
        np.save(tile_file, pts_u8)


def save_tiles(out_dir: str, file_t: str, suffix: str, tileset: list[np.array]):
    """Save the files as .ply files, with one tile per file. This is lossless."""

    def save_set(dirname, tiles):
        if len(tiles) == 0:
            return
        set_dir = pth.join(out_dir, dirname)
        os.makedirs(set_dir, exist_ok=True)
        [save_voxel_tile(pth.join(set_dir, tile_filename(t[0], suffix)), t[1]) for t in tiles]

    save_set(file_t, tileset)


def tile_file(path_in: str, file_t: str, out_dir: str, tile_sz: int, vox_sz: float, transforms: list[np.array], clean: bool, save_intermediate: bool):
    """Process a single .laz/.las/.ply/.pcd fil"""
    pc = laz_to_points(path_in, clean, save_intermediate)
    rot_axis = np.argmin(pc.get_max_bound() - pc.get_min_bound())
    for tx in transforms:
        tiled_pts_path = tiles_folder_name(path_in, vox_sz, tile_sz, tx)

        pc_tx = copy.deepcopy(pc)

        rot_xyz = [0, 0, 0]
        rot_xyz[rot_axis] = tx[3]
        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(rot_xyz)

        pc_tx.rotate(rot_mat, pc_tx.get_center())
        pc_tx.translate(tx[:3])

        tileset = points_to_tiles(np.asarray(pc_tx.points), tiled_pts_path, save_intermediate, vox_sz, tile_sz)
        suffix = f"{tx[0]}_{tx[1]}_{tx[2]}_{tx[3]}_{pth.basename(path_in)}"
        save_tiles(out_dir, file_t, suffix, tileset)


def produce_dataset(files: list, cfg: dict, out_dir):
    logger.info(f"Producing dataset :{out_dir}")
    logger.info(f"Tiler config:\n{toml.dumps(cfg)}")
    n_train = int(run.cfg.train_fraction * len(files))
    train_mask = np.zeros(len(files))
    train_mask[np.random.choice(range(len(train_mask)), n_train, replace=False)] = 1
    files_types = ["train" if t else "test" for t in train_mask]

    fn = 0
    for f, ft in zip(files, files_types):
        fn += 1
        logger.info(f"{fn:>3}/{len(files):<3} {ft:5} {f}")
        tile_file(f, ft, out_dir, cfg.grid_dim, cfg.vox_size, cfg.transforms, cfg.clean, cfg.save_intermediate)
    meta = copy.deepcopy(DEFAULT_METADATA)
    meta["grid_dim"] = cfg.grid_dim
    meta["vox_size"] = cfg.vox_size
    rutil.write_file(pth.join(out_dir, "meta.toml"), meta, overwrite_ok=True)


def _get_args():
    parser = argparse.ArgumentParser(prog="tile.py", description="Convert one or more .laz files to a set of .npy files for efficient loading during training")
    parser.add_argument("folder", help="Dataset output folder. Will contain tile.toml (config for tiler), meta.toml (dataset metadata), test and train subfolders, and a log file")
    parser.add_argument("--input-folder", help="Folder containing one or more .las or .laz files", required=True)
    return parser


if __name__ == "__main__":
    parser = _get_args()
    args = parser.parse_args()
    run = utils.Run(args.folder, "tiler", "tiler", False, DEFAULT_CONFIG)

    files = glob.glob(pth.join(args.input_folder, "*la[sz]"))

    produce_dataset(files, run.cfg, args.folder)
