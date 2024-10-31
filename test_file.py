"""Evaluate the compression ratio and various accuracy metrics over whole .laz files instead of just tiles"""

import argparse
import os.path as pth

import laspy as lp
from rocnet.file import RocNetFile
from rocnet.utils import load_file, save_file
from rocnet.tiler import laz_to_points
from os import makedirs
import utils
from test_tile import DEFAULT_CONFIG
import open3d as o3d
import numpy as np


vox_sz = 1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test_file.py", description="Evaluate and visualise the per-file performance and lossiness of one or more models")
    parser.add_argument("folder", help="Folder containing either train.toml or a test.toml file which points to multiple training runs to process")
    parser.add_argument("--visualise", help="Render the original and recovered point clouds side by side", action="store_true", default=False)
    args = parser.parse_args()

    run = utils.Run(args.folder, "test", "test-file", False, DEFAULT_CONFIG)

    files_dir = pth.join(run.run_dir, "test-file")
    makedirs(files_dir, exist_ok=True)
    model_paths = [pth.join(m, "model.pth") for m in utils.search_runs(run.cfg.models)]
    codecs = [RocNetFile(c) for c in model_paths]
    if not all([m._model.cfg.grid_dim == codecs[0]._model.cfg.grid_dim for m in codecs]):
        run.logger.error("Models must all have the same grid dim")
        [run.logger.error(cp, m._model.cfg.grid_dim) for (cp, m) in zip(run.cfg.models, codecs)]
        raise ValueError("Model grid_dims must all be the same")
    models_files = [item for inner in [[[m, f] for m in codecs] for f in run.cfg.files] for item in inner]
    files_out = [pth.join(files_dir, f"{pth.basename(n)}_{utils.describe_run(pth.split(m._model.model_path)[0])['uid']}.rnt") for m, n in models_files]

    files_all = [[m[0], m[1], fo] for m, fo in zip(models_files, files_out)]

    for codec, file_in, file_out in files_all:
        results_file = f"{file_out}.toml"
        if pth.exists(results_file):
            results = load_file(results_file, quiet=True)
            pts_in = np.asarray(laz_to_points(file_in, True, False).points)
            pts_out = codec.decode(file_out)
        else:
            pts_in = np.asarray(laz_to_points(file_in, True, False).points)
            try:
                if not pth.exists(file_out):
                    codec.encode(file_out, pts_in, vox_sz, bundle_decoder=False)
                pts_out = codec.decode(file_out)
            except Exception as e:
                run.logger.error(f"Codec failed on file: {file_in}")
                run.logger.error(e)
                continue
            results = utils.compare_pts(pts_in, pts_out, vox_sz)
            results.file_header = codec.examine(file_out)
            results.size_in = pth.getsize(file_in)
            results.size_out = pth.getsize(file_out)
            results.compression_ratio = results.size_in / results.size_out
            save_file(results_file, results, overwrite=True)
        print("\n", flush=True)
        run.logger.info(f"{utils.describe_run(pth.split(codec._model.model_path)[0])['uid']:48} {results.compression_ratio:5.1f} {results.hamming_plus:10}+ {results.hamming_minus:10}- {results.chamfer:5.3f}")
        run.logger.info(f"{pth.basename(file_in):48} {'':5} {100*results.hamming_plus/results.n_pts_in:9.1f}%+ {100*results.hamming_minus/results.n_pts_in:9.3}%-")
        if args.visualise:
            pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_in))
            pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_out))
            utils.compact_view([pc1, pc2])
            o3d.visualization.draw_geometries([pc1, pc2])

    print("done")
