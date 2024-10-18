"""Evaluate the compression ratio and various accuracy metrics over whole .laz files instead of just tiles"""

import argparse
import os.path as pth

import laspy as lp
import toml
from rocnet.file import RocNetFile
from rocnet.utils import load_file, save_file
from os import makedirs
import utils
from test_tile import DEFAULT_CONFIG
import open3d as o3d


def model_id(m):
    s_or_f = "f" if m.cfg.has_root_encoder else "s"
    size = m.cfg.feature_code_size
    return f"{s_or_f}{size:04}"


vox_sz = 1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py", description="Evaluate the performance of one or more RocNet models")
    parser.add_argument("folder", help="folder containing test.toml configuration for where to load the models from")
    parser.add_argument("--visualise", help="Render the original and recovered point clouds side by side", action="store_true")
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
    models_files = [item for inner in [[[m, f] for f in run.cfg.files] for m in codecs] for item in inner]
    files_out = [pth.join(files_dir, f"{pth.basename(n)}_{model_id(m._model)}.rnt") for m, n in models_files]

    files_all = [[m[0], m[1], fo] for m, fo in zip(models_files, files_out)]
    for codec, file_in, file_out in files_all:
        results_file = f"{file_out}.toml"
        if pth.exists(results_file):
            results = load_file(results_file, quiet=True)
        else:
            pts_in = lp.read(file_in)
            try:
                if not pth.exists(file_out):
                    codec.encode(file_out, pts_in.xyz, vox_sz, bundle_decoder=False)
                pts_out = codec.decode(file_out)
            except Exception as e:
                run.logger.error(f"Codec failed on file: {file_in}")
                run.logger.error(e)
                continue
            results = utils.compare_pts(pts_in.xyz, pts_out, vox_sz)
            results.file_header = codec.examine(file_out)
            results.size_in = pth.getsize(file_in)
            results.size_out = pth.getsize(file_out)
            results.compression_ratio = results.size_in / results.size_out
            results.model = model_id(codec._model)
            save_file(results_file, results, overwrite=False)
        print(toml.dumps(results))
        if args.visualise:
            pts_in = lp.read(file_in)
            pts_out = codec.decode(file_out)
            pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_in.xyz))
            pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_out))
            utils.compact_view([pc1, pc2])
            o3d.visualization.draw_geometries([pc1, pc2])

    print("done")
