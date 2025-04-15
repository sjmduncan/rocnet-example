"""A set of utilities for working with instances of RocNet, datasets, and pointclouds. This is both a script, and a module that can be imported."""

import argparse
import glob
import json
import logging
import os.path as pth
import subprocess
import sys
from datetime import datetime
from os import makedirs, remove, rmdir
from shutil import copyfile

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict as ed
from rocnet.utils import ensure_file, load_file

logger = logging.getLogger(__name__)
log_handler_stdout = logging.StreamHandler(sys.stdout)
logger.addHandler(log_handler_stdout)
logger.setLevel(level=logging.INFO)


def hausdorff(p1, p2, two_sided=True):
    """Compute the Hausdorff distance between p1 and p2 (or just from p1 to p2 if two_sided=False)

    :param p1: first point cloud
    :type p1: open3d.geometry.PointCloud
    :param p2: second point cloud
    :type p2: open3d.geometry.PointCloud
    :param two_sided: set to false to compute the distance from p1 to p2 instead of in both directions (default=True)
    :type two_sided: bool
    :returns: hausdorff distance (float)
    """
    try:
        nn_dist1 = np.max(np.asarray(p1.compute_point_cloud_distance(p2)), axis=0)
        if two_sided:
            nn_dist2 = np.max(np.asarray(p2.compute_point_cloud_distance(p1)), axis=0)
            return (nn_dist1 * nn_dist1) / 2 + (nn_dist2 * nn_dist2) / 2
        return nn_dist1 * nn_dist1
    except ValueError:
        return -1


def chamfer(p1, p2):
    """Compute the chamfer distance between p1 and p2

    :param p1: first point cloud
    :type p1: open3d.geometry.PointCloud
    :param p2: second point cloud
    :type p2: open3d.geometry.PointCloud
    :returns: chamfer distance (float)
    """
    try:
        nn_dist1 = np.asarray(p1.compute_point_cloud_distance(p2))
        nn_dist2 = np.asarray(p2.compute_point_cloud_distance(p1))
        return sum(nn_dist1) / nn_dist1.shape[0] + sum(nn_dist2) / nn_dist2.shape[0]
    except ValueError:
        return -1
    except ZeroDivisionError:
        return -1


def hamming(v1, v2, two_sided=True):
    """Compute the hamming distance between v1 and v2 (or just from v1 to v2 if two_sided=False)

    :param p1: first point cloud
    :type p1: open3d.geometry.PointCloud
    :param p2: second point cloud
    :type p2: open3d.geometry.PointCloud
    :param two_sided: set to false to compute the distance from p1 to p2 instead of in both directions (default=True)
    :type two_sided: bool
    :returns: hamming distance (float)
    """
    vi2 = o3d.utility.Vector3dVector([v2.get_voxel_center_coordinate(v.grid_index) for v in v2.get_voxels()])
    v2_in_v1 = np.array(v1.check_if_included(vi2))
    result = np.sum(v2_in_v1 == 0)
    if two_sided:
        vi1 = o3d.utility.Vector3dVector([v1.get_voxel_center_coordinate(v.grid_index) for v in v1.get_voxels()])
        v1_in_v2 = np.array(v2.check_if_included(vi1))
        result += np.sum(v1_in_v2 == 0)

    return result


def vox_points(pts: o3d.geometry.PointCloud, vox_size: float):
    """Quantise a point cloud to the provided voxel size, return a point cloud of the centers of the occupied voxels

    :param pts: Pointcloud to voxelise
    :type pts: open3d.geometry.PointCloud
    :param vox_size: Voxel size to
    :type pts: float
    :returns: Pointcloud where the points are the centers of the occupied voxels in voxel-grid space
    """
    vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pts, voxel_size=vox_size)
    vox_pts = np.array([v.grid_index for v in vox.get_voxels()]) + vox.origin.astype(int)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vox_pts))


def compare_pts(ref: np.array, cmp: np.array, vox_size: float) -> dict:
    """Computes the hausdorff (in both directions), and chamfer distances between the input cloud, also the hausdorff and hamming distances with both ref and cmp quantised to vox_size

    :param ref: array of raw points, probably straight from a LIDAR or other point cloud source
    :type ref: numpy.array
    :param cmp: array of points retrieved from a RocNet code
    :type cmp: numpy.array
    :param vox_size: size of the voxels used to quantise the point cloud for RocNet
    :type vox_size: float

    :returns: a dict of all the metrics, where quantised metrics (i.e. between voxel grids) are prefixed by ``q_``
    """

    logger.info("compare_pts: Create point clouds")
    cmp_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cmp))
    ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref))
    logger.info("compare_pts: Create voxel grids")
    cmp_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(cmp_pc, vox_size)
    ref_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(ref_pc, vox_size)
    ref_pc_ds = vox_points(ref_pc, vox_size)

    logger.info("compare_pts: hausdorff")
    hausdorff_plus = hausdorff(cmp_pc, ref_pc, False)
    hausdorff_minus = hausdorff(ref_pc, cmp_pc, False)
    logger.info("compare_pts: chamfer")
    chamfer_dist = chamfer(ref_pc, cmp_pc)
    n_pts_in = len(ref)
    n_pts_out = len(cmp)
    logger.info("compare_pts: quantised hausdorff")
    q_hausdorff_plus = hausdorff(cmp_pc, ref_pc_ds, False)
    q_hausdorff_minus = hausdorff(ref_pc_ds, cmp_pc, False)
    logger.info("compare_pts: quantised chamfer")
    q_chamfer = chamfer(ref_pc_ds, cmp_pc)
    q_n_pts = len(ref_pc_ds.points)
    logger.info("compare_pts: hamming")
    hamming_minus = hamming(cmp_vox, ref_vox, False)
    hamming_plus = hamming(ref_vox, cmp_vox, False)

    return ed(
        {
            "hausdorff_plus": hausdorff_plus,
            "hausdorff_minus": hausdorff_minus,
            "chamfer": chamfer_dist,
            "n_pts_in": n_pts_in,
            "n_pts_out": n_pts_out,
            "q_hausdorff_plus": q_hausdorff_plus,
            "q_hausdorff_minus": q_hausdorff_minus,
            "q_chamfer": q_chamfer,
            "q_n_pts": q_n_pts,
            "hamming_minus": hamming_minus,
            "hamming_plus": hamming_plus,
        }
    )


def dir_type(dir):
    """Checks type of directory based on its contents:
        - 'training-run' contains train.log and no subdirectories
        - 'run-collection' contains train.toml but not train.log, may or may not contain subdirectories
        - 'run-collection-collection' contains at least one subdirectory which is a run collection

    :param dir: folder to check
    :type dir: str
    :returns: one of ``'training-run'``, ``'run-collection'``, ``'run-collection-collection'`` or None
    """
    dir_contents = glob.glob(pth.join(dir, "*"))
    dir_dirs = [d for d in dir_contents if pth.isdir(d)]
    dir_files = [f for f in dir_contents if pth.isfile(f)]

    dir_dirnames = [pth.split(d)[1] for d in dir_dirs]
    dir_filenames = [pth.split(d)[1] for d in dir_files]

    if "train.log" in dir_filenames and len(dir_dirnames) == 0:
        return "training-run"

    if "train.toml" in dir_filenames and "train.log" not in dir_filenames:
        return "run-collection"

    if any([dir_type(d) == "run-collection" for d in dir_dirs]):
        return "run-collection-collection"

    return None


def search_runs(dir: str, run_type="notempty"):
    """Search for all training runs in the provided directory

    :param dir: Folder which you expect to be a training run, a run collection, or a run collection collection
    :type dir: str
    :param run_type: Type of run to find, one of ``'empty'``, ``'notempty'``, or ``'all'``
    :type run_type: str

    :returns: a flat list of all of the training runs in the folder
    """

    if run_type not in ["empty", "notempty", "all"]:
        raise ValueError(f"run_type should be one of ['empty', 'notempty', 'all']. Current value={run_type}")

    if isinstance(dir, list):
        return [r for rr in [search_runs(p, run_type) for p in dir] for r in rr]

    assert pth.exists(dir), f"Path does not exist: {dir}"

    dt = dir_type(dir)
    if dt == "training-run":
        if run_type == "all":
            return [dir]

        snapshots = glob.glob(pth.join(dir, "*.pth"))
        n_models = len(snapshots)
        if n_models > 0 and "model.pth" not in [pth.basename(s) for s in snapshots]:
            logger.warning(f"Has snapshots but no final model.pth: {dir}")
            return [dir]
        elif n_models == 0 and run_type == "empty":
            return [dir]
        elif n_models > 0 and run_type == "notempty":
            return [dir]

        return None

    elif dt == "run-collection":
        dirs = glob.glob(pth.join(dir, "train_*"))
        dirs = [d for d in dirs if pth.isdir(d)]
        runs = [r for r in [search_runs(d, run_type) for d in dirs]]
        runs = [r for r in runs if r is not None]
        return [r for rr in runs for r in rr]
    elif dt == "run-collection-collection":
        runs_tmp = [search_runs(p, run_type) for p in glob.glob(pth.join(dir, "*")) if pth.isdir(p)]
        runs_tmp = [r for r in runs_tmp if r is not None]
        return [rc for rcc in runs_tmp for rc in rcc]


def parse_training_run(run_dir):
    """List all info about a training run, including the loss, which snapshot has optimal validation loss, and whether the final model file exists

    :param run_dir: path to the training run folder
    :type run_dir: str

    :returns: Dictionary containing model information
    """

    has_final_model = pth.exists(pth.join(run_dir, "model.pth"))
    snapshots = glob.glob(pth.join(run_dir, "model_*_training.pth"))
    snapshots.sort()
    snapshot_epochs = [int(pth.split(p)[1].split("_")[1]) for p in snapshots]
    loss = np.loadtxt(snapshots[-1][:-12] + "loss.csv")
    snapshot_dicts = [torch.load(f[:-13] + ".pth", map_location="cpu") for f in snapshots]
    snapshot_meta = [d["metadata"] for d in snapshot_dicts]
    snapshot_losses = [d["loss"][-1] for d in snapshot_meta]

    return {
        "has_final_model": has_final_model,
        "final_model": pth.join(run_dir, "model.pth"),
        "optimal_snapshot_idx": np.argmin(snapshot_losses),
        "snapshots": list(zip(snapshots, snapshot_epochs, snapshot_losses)),
        "loss": loss,
    }


def run_epochs(run_dir):
    """Find all snapshot epochs for model_*_training.pth files, return list in ascending order"""
    epochs = [int(pth.split(p)[1].split("_")[1]) for p in glob.glob(pth.join(run_dir, "model_*_training.pth"))]
    epochs.sort()
    return epochs


def model_id(model_config: dict):
    """Return the ID of the model derived from model_config"""
    s_or_f = "f" if model_config.has_root_encoder else "s"
    return f"{model_config.grid_dim}-{s_or_f}{model_config.feature_code_size}"


def describe_run(run):
    """Returns a dict with various metadata about the training run, including run collection, train start time, epoch, and model UID (result of model_id plus the collection name and the start time)"""
    collection = pth.basename(pth.split(run)[0])
    time = pth.split(run)[1].replace("train_", "")
    cfg = load_file(pth.join(run, "train.toml"), quiet=True)
    epoch = run_epochs(run)
    max_epoch = max(epoch) if len(epoch) > 0 else 0
    return ed({"collection": collection, "time": time, "epochs": max_epoch, "uid": f"{collection}-{time}-{model_id(cfg['model'])}"})


def compact_view(geometries, bbox_size=None):
    """Transform a list of geometries (in-place) so that they're stacked end-to-end along the X-axis for better comparison

    :param geometries: list of Open3D geometries to transform
    :type run_dir: list
    :param bbox_size: cubic bounding box of each object, if not specified then the bounding box is computed to match the bounds of each geometry element
    :type bbox_size: float

    :returns: list of bounding boxes of the models
    """
    [g.translate(-g.get_min_bound()) for g in geometries]
    bl = [g.get_min_bound() for g in geometries]
    if bbox_size is None:
        tr = [g.get_max_bound() for g in geometries]
    else:
        tr = [b + np.array([bbox_size, bbox_size, bbox_size]) for b in bl]
    sizes = [r - b for b, r in zip(bl, tr)]
    x_offsets = [s[0] * 1.1 for s in sizes]
    x_offsets = np.cumsum(x_offsets)
    np.insert(x_offsets, 0, 0)
    corners = [[bl[0][0] + o, bl[0][1], bl[idx][2]] for idx, o in enumerate(x_offsets)]
    txs = [c - b for c, b in zip(corners, bl)]
    [g.translate(t) for g, t in zip(geometries, txs)]
    boxes = [o3d.geometry.AxisAlignedBoundingBox(b, t) for b, t in zip(bl, tr)]
    for i, b in enumerate(boxes):
        b.color = [0, 0, 0] if i == 0 else [0, 1, 0]
    [g.translate(t) for g, t in zip(boxes, txs)]
    return boxes


def visualise_interactive(data, metrics, bbox_size, model_meta):
    """Visualise a list of geometries with a matching list of metrics and model metadata, with keyboard controls for cycling through models and samples

    :param data: list of lists of geometries to render
    :type data: list
    :param metrics: list of lists of metrics to print for each geometry sample
    :type metrics: list
    :param bbox_size: bbox_size to pass to compact_geometries
    :type bbox_size: float
    :param model_meta: list of metadata for each model
    :type model_meta: list

    """
    dataset_idx = 0
    sample_idx = 0
    look_sample_idx = 0

    bounding_boxes = [[compact_view(s, bbox_size) for s in d] for d in data]

    def print_metric():
        print()

        for metric, meta in zip(np.array(metrics[dataset_idx][sample_idx]).T, model_meta):
            print(f"{meta.collection:>24} {meta.time} {metric[0]:6.1f} {metric[1]:6.1f} {metric[2]:5.3f}")

    def update(vis):
        view = json.loads(vis.get_view_status())["trajectory"][0]
        vis.clear_geometries()
        [vis.add_geometry(g, True) for g in data[dataset_idx][sample_idx]]
        [vis.add_geometry(g, True) for g in bounding_boxes[dataset_idx][sample_idx]]
        vis.get_view_control().set_front(view["front"])
        vis.get_view_control().set_lookat(view["lookat"])
        vis.get_view_control().set_up(view["up"])
        vis.get_view_control().set_zoom(view["zoom"])
        print_metric()

    def next_dataset(vis):
        nonlocal dataset_idx
        dataset_idx = (dataset_idx + 1) % len(data)
        update(vis)

    def prev_dataset(vis):
        nonlocal dataset_idx
        dataset_idx = (dataset_idx - 1 + len(data)) % len(data)
        update(vis)

    def next_sample(vis):
        nonlocal sample_idx
        sample_idx = (sample_idx + 1) % len(data[dataset_idx])
        update(vis)

    def lookat(vis):
        ctr = vis.get_view_control()
        ctr.set_lookat(bounding_boxes[dataset_idx][0][look_sample_idx].get_center())

    def lookprev(vis):
        nonlocal look_sample_idx
        n = len(data[dataset_idx][0])
        look_sample_idx = (look_sample_idx - 1 + n) % n
        lookat(vis)

    def looknext(vis):
        nonlocal look_sample_idx
        n = len(data[dataset_idx][0])
        look_sample_idx = (look_sample_idx + 1) % n
        lookat(vis)

    def prev_sample(vis):
        nonlocal sample_idx
        sample_idx = (sample_idx - 1 + len(data[dataset_idx])) % len(data[dataset_idx])
        update(vis)

    key_to_callback = {}
    key_to_callback[ord("N")] = next_sample
    key_to_callback[ord("B")] = prev_sample
    key_to_callback[ord("F")] = next_dataset
    key_to_callback[ord("R")] = prev_dataset
    key_to_callback[ord("S")] = lookprev
    key_to_callback[ord("W")] = looknext
    print("\n\nKeyboard Controls:")
    print("N - Next tile")
    print("B - Previous tile")
    print("F - Next dataset")
    print("R - Previous dataset")
    print("W - LookAt next sample for current tile")
    print("S - LookAt prev sample for current tile")
    print_metric()
    geometries = [g for g in data[dataset_idx][sample_idx]] + [g for g in bounding_boxes[dataset_idx][sample_idx]]
    o3d.visualization.draw_geometries_with_key_callbacks(geometries, key_to_callback)


class Run:
    """Start a new run with {out_dir} as the working dir, {out_dir}/{cfg_type}.toml as the config, and {out_dir}/{run_type}_ as the log and/or run_dir prefix

    :param out_dir: directory which contains the specified .toml config file, and where the output will go
    :type out_dir: str
    :param cfg_type: type of config to load, which is also the basename of the config file
    :type cfg_type: str
    :param run_type: prefix of basename of log file and if is_collection==True also self.run_dir
    :type run_type: stry
    :param is_collection: self.run_dir is {out_dir}/{run_type}_{START_TIME} (use this for training, benchmarking, or things which produce an annoying number of output files)
    :type is_collection: bool
    :param default_config: dict with default config values, should contain the whole structure of the config file
    :type default_config: dict
    :param seed: random seed to use for torch.seed NOTE: this should also use torch.use_deterministic_algorithms(True) but it's not that simple with CUDA
    :type seed: int

    Useful Attributes:
    """

    def __init__(self, out_dir: str, cfg_type: str, run_type: str, is_collection: bool = False, default_config: dict = None, seed: int = None):
        self.TIME_FMT = "%Y-%m-%d_%H.%M.%S"
        "Filename-firendly timestamp format string: '%Y-%m-%d_%H.%M.%S'"
        self.START_TIME = datetime.now().strftime(self.TIME_FMT)
        "Time when this run was started"

        self.logger = logger
        "Logger instance for convenience"

        self.run_type = run_type
        self.cfg_path = pth.join(out_dir, f"{cfg_type}.toml")
        if pth.exists(self.cfg_path):
            self.cfg = load_file(self.cfg_path, default_config)
            "Config dict for whatever you're running, probably loaded from a .toml file"
        elif default_config is not None:
            ensure_file(self.cfg_path, default_config)
            self.logger.warning(f"New default config file created: {self.cfg_path}")
            self.logger.warning("Please check its contents and re-run the script")
            exit(0)
        else:
            raise ValueError(f"No default_config provided, but config file does not exist: {self.cfg_path}")

        self._run_prefix = pth.join(out_dir, run_type)

        self.dir = out_dir
        self.run_dir = f"{self._run_prefix}_{self.START_TIME}" if is_collection else out_dir
        "Directory where all the output goes"
        makedirs(self.run_dir, exist_ok=True)
        if is_collection:
            logging.basicConfig(filename=pth.join(self.run_dir, f"{self.run_type}.log"), level=logging.INFO)
        else:
            logging.basicConfig(filename=pth.join(self.run_dir, f"{self.run_type}_{self.START_TIME}.log"), level=logging.INFO)
        if seed is not None:
            self.logger.info(f"Training with seed: {seed}")
            torch.manual_seed(seed)
            # FIXME can't have this with CUDA torch.use_deterministic_algorithms(True)

    def _git_snapshot(self):
        # If git exists and this is being run in a repository, then a file with the name
        # <timestamp>-<git-hash>.diff will be created with the patch of the current repository changes
        # to be able to record exactly which version of the software was used for a training run
        GIT_HASH = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if GIT_HASH.stderr == "":
            git_diff_name = f"{GIT_HASH.stdout[:-1]}.diff"
            with open(pth.join(self.run_dir, git_diff_name), "w") as f:
                subprocess.run(["git", "diff"], stdout=f)
        else:
            raise RuntimeError(f"Error running git: {GIT_HASH.stderr}")


def clean_empty(folder: str, noop: bool):
    """Remove all empty training runs (i.e. those without any .pth files).

    :param folder: training run, run collection, or collection of run collections to clean
    :type folder: str
    :param noop: set to true to print the operations without actually doing them
    :type noop: bool
    """
    runs = search_runs(folder, "empty")
    [[logger.info(f"Deleting {f}") for f in glob.glob(pth.join(d, "*"))] for d in runs]
    [logger.info(f"Deleting {f}") for f in runs]
    if noop:
        logger.info("Invoked with '--noop'; No files actually deleted")
    else:
        [[remove(f) for f in glob.glob(pth.join(d, "*"))] for d in runs]
        [rmdir(f) for f in runs]
    logger.info("done")


def clean_intermediate(folder: str, noop: bool):
    """Delete all training run snapshots except for those with the optimla validation loss, and the latest snapshots (which might not be the same)

    :param folder: training run, run collection, or collection of run collections to clean
    :type folder: str
    :param noop: set to true to print the operations without actually doing them
    :type noop: bool
    """
    logger.info(f"Removing intermediate snapshots from {folder}")
    runs = search_runs(folder, "notempty")
    runs = search_runs(folder, "notempty")
    runs_info = [parse_training_run(r) for r in runs]
    for r in runs_info:
        [logger.info(f"Keeping  {fn}") for fn in glob.glob(r["snapshots"][r["optimal_snapshot_idx"]][0][:-13] + "*")]
        if len(r["snapshots"]) != (r["optimal_snapshot_idx"] + 1):
            [logger.info(f"Keeping  {fn}") for fn in glob.glob(r["snapshots"][-1][0][:-13] + "*")]
        [[logger.info(f"Deleting {fn}") for fn in glob.glob(s[0][:-13] + "*")] for idx, s in enumerate(r["snapshots"][:-1]) if idx != r["optimal_snapshot_idx"]]
        if not noop:
            [[remove(fn) for fn in glob.glob(s[0][:-13] + "*")] for idx, s in enumerate(r["snapshots"]) if idx != r["optimal_snapshot_idx"]]

    if noop:
        logger.info("Invoked with '--noop'; No files actually deleted")
    logger.info("done")


def tidy(folder: str, noop: bool):
    """Ensure that a training run contains a model.pth file which corresponds to the model weights which produced the optimal validation loss

    :param folder: training run, run collection, or collection of run collections to tidy
    :type folder: str
    :param noop: set to true to print the operations without actually doing them
    :type noop: bool
    """
    logger.info(f"Tidying training runs in {folder}")
    runs = search_runs(folder, "notempty")
    runs_info = [parse_training_run(r) for r in runs]
    for r in runs_info:
        if r["has_final_model"]:
            logger.info(f"Final model already exist: {r['final_model']}")
        else:
            logger.info(f"Using as final model: {r['snapshots'][r['optimal_snapshot_idx']][0][:-13] + '.pth'}")
            if not noop:
                copyfile(r["snapshots"][r["optimal_snapshot_idx"]][0][:-13] + ".pth", r["final_model"])
    if noop:
        logger.info("Invoked with '--noop'; No files actually copied")
    logger.info("done")


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Training run or collection of training runs to operate on.")
    parser.add_argument("--clean-empty", help="Delete training run folders which did not produce *.pth snapshots. Use with caution, might delete stuff while training is in progress.", action="store_true")
    parser.add_argument("--clean-intermediate", help="Delete all sub-optimal snapshots in a training run, save only the best (optimal validation loss score) and latest (which might not be the same).", action="store_true")
    parser.add_argument("--tidy", help="Ensure that there's a 'model.pth' file which matches the snapshot with the best validation score.", action="store_true")
    parser.add_argument("--noop", help="List file deletion/modification without actually doing them.", action="store_true")
    return parser


if __name__ == "__main__":
    parser = _get_args()

    args = parser.parse_args()

    if args.tidy:
        tidy(args.folder, args.noop)
    if args.clean_empty:
        clean_empty(args.folder, args.noop)
    if args.clean_intermediate:
        clean_intermediate(args.folder, args.noop)
