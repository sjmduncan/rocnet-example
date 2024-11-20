"""Evaluate the compression ratio and various accuracy metrics over (some of) the test tiles in the dataset"""

import argparse
import os.path as pth

import open3d as o3d
import torch
from rocnet.dataset import Dataset, load_points
from rocnet.rocnet import RocNet

import utils

DEFAULT_CONFIG = {
    "n_samples": 20,
    "use_cuda": True,
    "datasets": ["../rocnet.data/default"],
    "models": ["../rocnet.weights/default"],
    "files": ["../rocnet.data/raw/default/default.laz"],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test_tile.py", description="Evaluate and visualise the per-tile performance and lossiness of one or more models")
    parser.add_argument("folder", help="Folder containing either train.toml or a test.toml file which points to multiple training runs to process")
    args = parser.parse_args()

    run = utils.Run(args.folder, "test", "test", False, DEFAULT_CONFIG)

    model_paths = [pth.join(m, "model.pth") for m in utils.search_runs(run.cfg.models)]
    models = [RocNet(c) for c in model_paths]
    if not all([m.cfg.grid_dim == models[0].cfg.grid_dim for m in models]):
        run.logger.error("Models must all have the same grid dim")
        [run.logger.error(cp, m.cfg.grid_dim) for (cp, m) in zip(run.cfg.models, models)]
        raise ValueError("Model grid_dims must all be the same")
    model_ids = [utils.describe_run(pth.split(r)[0]) for r in model_paths]
    datasets = [Dataset(d, models[0].cfg.grid_dim, train=False, max_train_samples=run.cfg.n_samples) for d in run.cfg.datasets]

    original_pointsets = [[load_points(d, dset.metadata.grid_dim, 1.0 / dset.grid_div, dset.metadata.vox_size) for d in dset.files] for dset in datasets]
    original_pcsets = [[o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points[:, :3])) for points in gridset] for gridset in original_pointsets]
    if models[0].cfg.voxel_channels == 3:
        for pcsets, ptsets in zip(original_pcsets, original_pointsets):
            for pc, ps in zip(pcsets, ptsets):
                pc.colors = o3d.utility.Vector3dVector(ps[:, 3:].astype("float") / 255.0)

    with torch.no_grad():
        encoded_codesets = [[[m.compress_points(sample) for m in models] for sample in originals] for originals in original_pointsets]
        recovered_ptsets = [[[m.uncompress_points(c) for m, c in zip(models, codes)] for codes in codeset] for codeset in encoded_codesets]

    recovered_pcsets = [[[o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(p[:, :3].astype("float"))) for p in pts] for pts in ptsets] for ptsets in recovered_ptsets]
    if models[0].cfg.voxel_channels == 3:
        for pcsets, ptsets in zip(recovered_pcsets, recovered_ptsets):
            for pcs, pts in zip(pcsets, ptsets):
                for pc, ps in zip(pcs, pts):
                    pc.colors = o3d.utility.Vector3dVector(ps[:, 3:].astype("float"))

    dataset = [[[o] + r for o, r in zip(oset, rset)] for oset, rset in zip(original_pcsets, recovered_pcsets)]
    hdm = [[[utils.hausdorff(dset[0], d, False) for d in dset[1:]] for dset in dsets] for dsets in dataset]
    hdp = [[[utils.hausdorff(d, dset[0], False) for d in dset[1:]] for dset in dsets] for dsets in dataset]
    cham = [[[utils.chamfer(d, dset[0]) for d in dset[1:]] for dset in dsets] for dsets in dataset]
    metrics = [[[p, m, c] for p, m, c in zip(hp, hm, chm)] for hp, hm, chm in zip(hdp, hdm, cham)]

    utils.visualise_interactive(dataset, metrics, models[0].cfg.grid_dim, model_ids)
