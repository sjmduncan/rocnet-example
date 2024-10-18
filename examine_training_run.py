"""Examine the loss decay curve of one or more training runs"""

import os.path as pth

import matplotlib.pyplot as plt
import numpy as np

import utils

from test_tile import DEFAULT_CONFIG


def plot_loss(loss, title, run):
    _, ax1 = plt.subplots(figsize=(10, 7))
    sel = [2, 3, 4, 5, 6, 7]
    ax1.plot(loss[:, sel])

    labels = ["R", "L", "T"]
    min_loss = np.min(loss[:, sel], axis=0)
    epoch_min_loss = np.argmin(loss[:, sel], axis=0)
    print("plotting ", title, run, epoch_min_loss[-1], min_loss[-1])
    labels = labels + [l + "_V" for l in labels]
    labels = labels + [f"{l:4}" + f"({x:5.1f}," + f"{y:5.1f})" for l, x, y in zip(labels, epoch_min_loss, min_loss)]

    [ax1.plot(x, y, "x") for x, y in zip(epoch_min_loss, min_loss)]
    ax1.legend(labels, ncols=4)
    ax1.set_title(title)
    ax1.set_xlim(0, loss[:, 7].shape[0])

    ax2 = ax1.twinx()
    ax2.plot(loss[:, 1], alpha=0.5)
    ax2.legend(["LR"], loc="upper left")
    ax2.set_ylim(0, 1.1 * np.max(loss[:, 1]))
    return ax1, ax2


run = utils.Run("../rocnet.test/dunedin128", "test", "examine", False, DEFAULT_CONFIG)
runs = utils.search_runs(run.cfg.models)
epoc = [utils.run_epochs(r) for r in runs]
loss = [np.loadtxt(pth.join(r, f"model_{max(e)}_loss.csv")) for r, e in zip(runs, epoc)]
run_descriptions = [utils.describe_run(r) for r in runs]

title = [f"{d['collection']}: {d['note']}" for d in run_descriptions]
plts = [plot_loss(l, t, r) for l, t, r in zip(loss, title, runs)]

plt.show()
