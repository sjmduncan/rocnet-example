"""Examine the loss decay curve of one or more training runs"""

import argparse
import os.path as pth

import matplotlib.pyplot as plt
import numpy as np
from rocnet.rocnet import RocNet

import utils
from test_tile import DEFAULT_CONFIG


def plot_loss(loss, title, run):
    _, ax1 = plt.subplots(figsize=(10, 7))
    sel = [2, 3, 4, 5, 6, 7]
    ax1.plot(loss[:, sel])

    labels = ["Recon", "Label", "Total"]
    min_loss = np.min(loss[:, sel], axis=0)
    epoch_min_loss = np.argmin(loss[:, sel], axis=0)
    print(f"plotting {title} {pth.split(run)[1][6:]} min_loss={min_loss[-1]:<6.1f} at epoch {epoch_min_loss[-1]:3}")
    labels = labels + [l + "_val" for l in labels]
    labels = labels + [f"min {l:4}" + f"({x:5.1f}," + f"{y:5.1f})" for l, x, y in zip(labels, epoch_min_loss, min_loss)]

    [ax1.plot(x, y, "o") for x, y in zip(epoch_min_loss, min_loss)]
    ax1.legend(labels, ncols=4)
    ax1.set_title(title)
    ax1.set_xlim(0, loss[:, 7].shape[0])

    ax2 = ax1.twinx()
    ax2.plot(loss[:, 1], alpha=0.5)
    ax2.legend(["LR"], loc="upper left")
    ax2.set_ylim(0, 1.1 * np.max(loss[:, 1]))
    return ax1, ax2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py", description="Evaluate the performance of one or more RocNet models")
    parser.add_argument("folder", help="folder containing test.toml configuration for where to load the models from")
    args = parser.parse_args()
    run = utils.Run(args.folder, "test", "examine", False, DEFAULT_CONFIG)
    runs = utils.search_runs(run.cfg.models)
    epoc = [utils.run_epochs(r) for r in runs]
    loss = [np.loadtxt(pth.join(r, f"model_{max(e)}_loss.csv")) for r, e in zip(runs, epoc)]
    models = [RocNet(pth.join(r, "model.pth")) for r in runs]
    run_descriptions = [utils.describe_run(r) for r in runs]
    title = [f"{d['collection']}" for d in run_descriptions]
    plts = [plot_loss(l, t, r) for l, t, r in zip(loss, title, runs)]

    plt.show()
