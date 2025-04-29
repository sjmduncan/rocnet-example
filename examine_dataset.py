import argparse

import matplotlib.pyplot as plt
import numpy as np
from rocnet.dataset import Dataset, load_points

import utils

import open3d as o3d


def _get_args():
    parser = argparse.ArgumentParser(prog="examine_dataset.py", description="Examine the composition of a dataset and/or visualise the point  clouds")
    parser.add_argument("folder", help="Folder containing meta.toml and the train and test folders containing the tiles")
    parser.add_argument("--grid-dim", help="Grid dimension of the voxel grid contained in one octree", default=128)
    parser.add_argument("--leaf-dim", help="Size of the octree leaf block", default=32)
    parser.add_argument("--max-samples", help="Maximum number of samples to load from the dataset", default=20)
    parser.add_argument("--train", help="Load the training dataset", default=True)
    parser.add_argument("--visualise", help="Render the point clouds", action="store_true")
    return parser


if __name__ == "__main__":
    parser = _get_args()
    args = parser.parse_args()
    utils.logger.info(f"Examining dataset: {args.folder} ({'train' if args.train else 'test'})")
    utils.logger.info(f"grid_dim/leaf_dim = {args.grid_dim}/{args.leaf_dim}")
    dataset = Dataset(args.folder, args.grid_dim, args.train, max_samples=args.max_samples)
    utils.logger.info(f"Loading {len(dataset.files)} samples")
    if args.visualise and len(dataset.files) > 20:
        utils.logger.warning(f"Loading more than 20 files (n={len(dataset.files)}) with --visualise=True; This will be slow and use a lot of memory.")

    dataset.read_files(args.grid_dim, args.leaf_dim)

    ltc = [t.leaf_type_count() for t in dataset]
    ltc_total = np.sum(ltc, axis=0)
    ltc_total[2] = ltc_total[2] / len(ltc)
    l_total = int(sum(ltc_total[:2]))
    utils.logger.info(f"samples {len(dataset.files)}")
    utils.logger.info(f"leaves {l_total}")
    utils.logger.info(f"mixed {100 * ltc_total[1] / l_total:4.1f}% ({int(ltc_total[1])})")
    utils.logger.info(f"empty {100 * ltc_total[0] / l_total:4.1f}% ({int(ltc_total[0])})")

    plt.figure()
    plt.hist(np.array(ltc)[:, 0], range=(0, 8), bins=8, alpha=0.8, label="Empty")
    plt.hist(np.array(ltc)[:, 1], range=(0, 8), bins=8, alpha=0.8, label="Mixed")
    plt.title("Distribution of leaf type count per octree")
    plt.legend()

    plt.figure()
    expected_occupancy = 0.33
    plt.hist(np.array(ltc)[:, 2], range=(0, expected_occupancy), bins=30)
    plt.title("Distribution of average mixed leaf occupancy per octree")

    if args.visualise:
        utils.logger.info("Loading files for visualization")
        pts = [load_points(f, 1.0 / dataset.grid_div) for f in dataset.files]
        ptclouds = np.array([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[:, :3])) for p in pts])

        for p, g in zip(pts, ptclouds):
            if p.shape[1] == 6:
                g.colors = o3d.utility.Vector3dVector(p[:, 3:])
        n = 5
        ptclouds = ptclouds.reshape((1, n, int(len(ptclouds) / n)))
        utils.visualise_interactive(ptclouds, args.grid_dim)

    plt.show()
