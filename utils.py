"""A set of utilities for working with instances of RocNet, datasets, and pointclouds"""

import argparse
import glob
import logging
import os.path as pth
import subprocess
import sys
from datetime import datetime
from os import makedirs

from easydict import EasyDict as ed

import numpy as np
import open3d as o3d
from rocnet.utils import ensure_file, load_file
import json

logger = logging.getLogger(__name__)


def hausdorff(p1, p2, two_sided=True):
    """Compute the Hausdorff distance between p1 and p2 (or from p1 to p2 if two_sided=False)"""
    try:
        nn_dist1 = np.max(np.asarray(p1.compute_point_cloud_distance(p2)), axis=0)
        if two_sided:
            nn_dist2 = np.max(np.asarray(p2.compute_point_cloud_distance(p1)), axis=0)
            return (nn_dist1 * nn_dist1) / 2 + (nn_dist2 * nn_dist2) / 2
        return nn_dist1 * nn_dist1
    except ValueError:
        return -1


def chamfer(p1, p2):
    "Compute the chamfer distance between p1 and p2"
    try:
        nn_dist1 = np.asarray(p1.compute_point_cloud_distance(p2))
        nn_dist2 = np.asarray(p2.compute_point_cloud_distance(p1))
        return sum(nn_dist1) / nn_dist1.shape[0] + sum(nn_dist2) / nn_dist2.shape[0]
    except ValueError:
        return -1
    except ZeroDivisionError:
        return -1


def hamming(v1, v2, two_sided=True):
    vi2 = o3d.utility.Vector3dVector([v2.get_voxel_center_coordinate(v.grid_index) for v in v2.get_voxels()])
    v2_in_v1 = np.array(v1.check_if_included(vi2))
    result = np.sum(v2_in_v1 == False)
    if two_sided:
        vi1 = o3d.utility.Vector3dVector([v1.get_voxel_center_coordinate(v.grid_index) for v in v1.get_voxels()])
        v1_in_v2 = np.array(v2.check_if_included(vi1))
        result += np.sum(v1_in_v2 == False)

    return result


def vox_points(pts: o3d.geometry.PointCloud, vox_size: float):
    """Voxelize pointcloud, retrieve voxel centers, and return a pointcloud of those"""
    vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pts, voxel_size=vox_size)
    vox_pts = np.array([v.grid_index for v in vox.get_voxels()]) + vox.origin.astype(int)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vox_pts))


def compare_pts(ref: np.array, cmp: np.array, vox_size: float) -> dict:
    """Computes the hausdorff (in either direction), and chamfer distances between the input cloud, also the hausdorff and hamming distances (both in either direction) after quantising with both to vox_size

    ref: array of raw points, probably straight from a LIDAR or other point cloud source
    cmp: array of points retrieved from a RocNet code
    vox_size: size of the voxels used to quantise the point cloud for RocNet

    returns: dict of the metrics, with keys for quantised versions prefixed with q_
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
    - 'training-run' contains train.toml and train.log and no subdirectories
    - 'run-collection' contains train.toml but not train.log, no subdirectory check because it might not have been run yet
    - 'run-collection-collection' contains subdirectories and at least one of them is a run-cllection

    dir: dir to check

    """
    dir_contents = glob.glob(pth.join(dir, "*"))
    dir_dirs = [d for d in dir_contents if pth.isdir(d)]
    dir_files = [f for f in dir_contents if pth.isfile(f)]

    dir_dirnames = [pth.split(d)[1] for d in dir_dirs]
    dir_filenames = [pth.split(d)[1] for d in dir_files]

    if "train.toml" in dir_filenames and "train.log" in dir_filenames and len(dir_dirnames) == 0:
        return "training-run"

    if "train.toml" in dir_filenames and "train.log" not in dir_filenames:
        return "run-collection"

    if any([dir_type(d) == "run-collection" for d in dir_dirs]):
        return "run-collection-collection"

    return "unknown"


def search_runs(parent, run_type="notempty"):
    """Recursive search for all trainig runs contained in parent. Returns a flat list of the full paths to all training runs"""
    if run_type not in ["empty", "notempty", "all"]:
        raise ValueError(f"run_type should be one of ['empty', 'notempty', 'all']. Current value={run_type}")

    if isinstance(parent, list):
        return [r for rr in [search_runs(p, run_type) for p in parent] for r in rr]

    dt = dir_type(parent)
    if dt == "training-run":
        if run_type == "all":
            return [parent]

        snapshots = glob.glob(pth.join(parent, "*.pth"))
        n_models = len(snapshots)
        if n_models > 0 and "model.pth" not in [pth.basename(s) for s in snapshots]:
            print(f"Has snapshots but no final model.pth: {parent}")
        elif n_models == 0 and run_type == "empty":
            return [parent]
        elif n_models > 0 and run_type == "notempty":
            return [parent]

        return None

    elif dt == "run-collection":
        dirs = glob.glob(pth.join(parent, "train_*"))
        dirs = [d for d in dirs if pth.isdir(d)]
        runs = [r for r in [search_runs(d, run_type) for d in dirs]]
        runs = [r for r in runs if r is not None]
        return [r for rr in runs for r in rr]
    elif dt == "run-collection-collection":
        runs_tmp = [search_runs(p, run_type) for p in glob.glob(pth.join(parent, "*")) if pth.isdir(p)]
        runs_tmp = [r for r in runs_tmp if r is not None]
        return [rc for rcc in runs_tmp for rc in rcc]


def run_epochs(run_dir):
    """find all snapshot epochs for model_*_training.pth files, return list in ascending order"""
    epochs = [int(pth.split(p)[1].split("_")[1]) for p in glob.glob(pth.join(run_dir, "model_*_training.pth"))]
    epochs.sort()
    return epochs


def model_id(mc: dict):
    """Return"""
    s_or_f = "f" if mc.has_root_encoder else "s"
    return f"{mc.grid_dim}-{s_or_f}{mc.feature_code_size}"


def run_id(rc: dict):
    """Get the id of a training run"""
    return f"{model_id(rc.model)}"


def describe_run(run):
    collection = pth.basename(pth.split(run)[0])
    time = pth.split(run)[1].replace("train_", "")
    cfg = load_file(pth.join(run, "train.toml"), quiet=True)
    note = cfg.note
    epoch = run_epochs(run)
    max_epoch = max(epoch) if len(epoch) > 0 else 0
    return ed({"collection": collection, "time": time, "epochs": max_epoch, "note": note, "uid": f"{collection}-{time}-{run_id(cfg)}"})


def compact_view(geometries, gdim=None):
    [g.translate(-g.get_min_bound()) for g in geometries]
    bl = [g.get_min_bound() for g in geometries]
    if gdim is None:
        tr = [g.get_max_bound() for g in geometries]
    else:
        tr = [b + np.array([gdim, gdim, gdim]) for b in bl]
    sizes = [r - l for l, r in zip(bl, tr)]
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


def vis_simple(data, paths, nonblocking=False):
    sample_idx = 0
    view = None

    def save_rec_vew(vis):
        jfile = pth.join(pth.split(paths[sample_idx])[0], "view.json")
        jstr = vis.get_view_status()
        with open(jfile, "w") as jfile_out:
            jfile_out.write(jstr)

    def load_rec_vew(vis):
        nonlocal view
        jfile = pth.join(pth.split(paths[sample_idx])[0], "view.json")
        if not pth.exists(jfile):
            return
        with open(jfile, "r") as jfile_in:
            jstr = jfile_in.read()
        view = json.loads(jstr)["trajectory"][0]
        vis.set_view_status(jstr)
        vis.poll_events()
        vis.update_renderer()

    def snapshot(vis):
        vis.capture_screen_image(paths[sample_idx], do_render=False)

    def update(vis):
        vis.clear_geometries()
        vis.add_geometry(data[sample_idx], True)
        vis.get_view_control().set_front(view["front"])
        vis.get_view_control().set_lookat(view["lookat"])
        vis.get_view_control().set_up(view["up"])
        vis.get_view_control().set_zoom(view["zoom"])
        vis.poll_events()
        vis.update_renderer()
        vis.reset_view_point(True)
        print(paths[sample_idx])
        snapshot(vis)

    def save_all(vis):
        nonlocal sample_idx
        for s in range(len(data)):
            sample_idx = s
            load_rec_vew(vis)
            update(vis)

    def next_sample(vis):
        nonlocal sample_idx
        sample_idx = (sample_idx + 1) % len(data)
        update(vis)

    def prev_sample(vis):
        nonlocal sample_idx
        sample_idx = (sample_idx - 1 + len(data)) % len(data)
        update(vis)

    key_to_callback = {}
    key_to_callback[ord("N")] = next_sample
    key_to_callback[ord("B")] = prev_sample
    key_to_callback[ord("S")] = snapshot
    key_to_callback[ord("T")] = save_rec_vew
    key_to_callback[ord("Y")] = load_rec_vew
    key_to_callback[ord("U")] = save_all
    tx_vec = -data[-1].get_center()
    [d.translate(tx_vec) for d in data]
    if nonblocking:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        save_all(vis)
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries_with_key_callbacks([data[sample_idx]], key_to_callback)


def visualise_interactive(data, metrics, gdim, model_meta):
    dataset_idx = 0
    sample_idx = 0
    look_sample_idx = 0

    bounding_boxes = [[compact_view(s, gdim) for s in d] for d in data]

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
    """Handle some runtime paths and config files

    Useful members:
      START_TIME: filename-friendly timestamp of when this object was initialised
      logger: logger instance for top-level runtime modules to use
      cfg_path: path to the config file
      cfg: easydict containing the loaded config file
      run_dir: where all the output goes (is the same as dir if created with is_collection==False)
      dir: directory containing the config file, and the output
    """

    def __init__(self, out_dir: str, cfg_type: str, run_type: str, is_collection: bool = False, default_config: dict = None):
        """Start a new run with {out_dir} as the working dir, {out_dir}/{cfg_type}.toml as the config, and {out_dir}/{run_type}_ as the log and/or run_dir prefix

        out_dir: directory which contains the config file and will contain the output
        cfg_type: basename of the config file
        run_type: prefix of basename of log file and if is_collection==True also self.run_dir
        is_collection: self.run_dir is {out_dir}/{run_type}_{START_TIME} (use this for training, benchmarking, or things which produce an annoying number of output files)
        """

        self.TIME_FMT = "%Y-%m-%d_%H.%M.%S"
        self.START_TIME = datetime.now().strftime(self.TIME_FMT)

        self.logger = logging.getLogger(__name__)
        self.log_handler_stdout = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(self.log_handler_stdout)

        self.run_type = run_type
        self.cfg_path = pth.join(out_dir, f"{cfg_type}.toml")
        if pth.exists(self.cfg_path):
            self.cfg = load_file(self.cfg_path, default_config)
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
        makedirs(self.run_dir, exist_ok=True)
        if is_collection:
            logging.basicConfig(filename=pth.join(self.run_dir, f"{self.run_type}.log"), level=logging.INFO)
        else:
            logging.basicConfig(filename=pth.join(self.run_dir, f"{self.run_type}_{self.START_TIME}.log"), level=logging.INFO)

    def git_snapshot(self):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="utils.py", description="Utils for working with training runs, models, and datasets")
    parser.add_argument("folder", help="Folder to operate on. Can be a training run or a dataset")
    parser.add_argument("--clean-empty", help="Delete training run folders which did not produce *.pth snapshots (e.g. crashes, hangs, training too slow)", action="store_true")

    args = parser.parse_args()
