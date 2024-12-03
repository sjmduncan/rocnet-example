import argparse

import matplotlib.pyplot as plt
import numpy as np
from rocnet.dataset import Dataset

import utils


def _get_args():
    parser = argparse.ArgumentParser(prog="test.py", description="Examine the composition of a dataset and/or visualise the point  clouds")
    parser.add_argument("folder", help="folder containing test.toml configuration for where to load the models from")
    parser.add_argument("--grid-dim", help="Grid dimension of the voxel grid contained in one octree", default=128)
    parser.add_argument("--leaf-dim", help="Size of the octree leaf block", default=32)
    parser.add_argument("--max-samples", help="Maximum number of samples to load from the dataset", default=None)
    parser.add_argument("--visualise", help="Render the point clouds", action="store_true")
    return parser


if __name__ == "__main__":
    parser = _get_args
    args = parser.parse_args()

    run = utils.Run(args.folder, "train", "examine-dataset", False)

    dataset = Dataset(run.cfg.dataset_path, args.grid_dim, args.train, max_samples=args.max_samples)
    dataset.read_files(args.grid_dim, args.leaf_dim)

    ltc = [t.leaf_type_count() for t in dataset]
    ltc_total = np.sum(ltc, axis=0)
    ltc_total[2] = ltc_total[2] / len(ltc)
    l_total = int(sum(ltc_total[:2]))
    print(f"files {len(dataset.files)}\nleaves {l_total}\n  mixed {100 * ltc_total[1]/l_total:4.1f}% ({int(ltc_total[1])}) \n  empty {100 * ltc_total[0]/l_total:4.1f}% ({int(ltc_total[0])})")

    plt.figure()
    plt.hist(np.array(ltc)[:, 0], range=(0, 8), bins=8, alpha=0.8, label="Empty")
    plt.hist(np.array(ltc)[:, 1], range=(0, 8), bins=8, alpha=0.8, label="Mixed")
    plt.title("Distribution of leaf type count per octree")
    plt.legend()

    plt.figure()
    expected_occupancy = 0.33
    plt.hist(np.array(ltc)[:, 2], range=(0, expected_occupancy), bins=30)
    plt.title("Distribution of average mixed leaf occupancy per octree")
    plt.show()
