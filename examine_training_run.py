import argparse
import os.path as pth

import matplotlib.pyplot as plt
import numpy as np
from rocnet.rocnet import RocNet

import utils
from test_tile import DEFAULT_CONFIG
from torchinfo import summary
from rocnet.utils import sizeof_fmt as sf


def _plot_loss(loss, title, run, n_params, logger):
    _, ax1 = plt.subplots(figsize=(10, 7))
    sel = [2, 3, 4, 5, 6, 7]
    ax1.plot(loss[:, sel])

    labels = ["rec", "lab", "tot"]
    min_loss = np.min(loss[:, sel], axis=0)
    epoch_min_loss = np.argmin(loss[:, sel], axis=0)
    logger.info(f"plotting {title} {pth.split(run)[1][6:]}")
    logger.info(f"         min_loss {min_loss[-1]:.1f} at epoch {epoch_min_loss[-1] + 1}")
    logger.info(f"        n_weights {sf(n_params[0], '', True)} {sf(n_params[1], '', True)} {sf(n_params[2], '', True)} (encoder node_classifier decoder)")

    labels = labels + [label + "_vld" for label in labels]
    labels = labels + [f"min {label:4}" + f"({x:5.1f}," + f"{y:5.1f})" for label, x, y in zip(labels, epoch_min_loss, min_loss)]

    [ax1.plot(x, y, "o") for x, y in zip(epoch_min_loss, min_loss)]
    ax1.legend(labels, ncols=4)
    ax1.set_title(title)
    ax1.set_xlim(0, loss[:, 7].shape[0])

    ax2 = ax1.twinx()
    ax2.plot(loss[:, 1], alpha=0.5)
    ax2.legend(["LR"], loc="upper left")
    ax2.set_ylim(0, 1.1 * np.max(loss[:, 1]))
    return ax1, ax2


def _get_args():
    parser = argparse.ArgumentParser(prog="test.py", description="Evaluate the performance of one or more RocNet models")
    parser.add_argument("folder", help="folder containing test.toml configuration for where to load the models from")
    parser.add_argument("--visualise", help="Plot the loss and learning rate curves", action="store_true")
    return parser


if __name__ == "__main__":
    parser = _get_args()
    args = parser.parse_args()
    run = utils.Run(args.folder, "test", "examine", False, DEFAULT_CONFIG)
    runs = utils.search_runs(run.cfg.models)
    epoc = [utils.run_epochs(r) for r in runs]
    loss = [np.loadtxt(pth.join(r, f"model_{max(e)}_loss.csv")) for r, e in zip(runs, epoc)]
    models = [RocNet(pth.join(r, "model.pth")) for r in runs]
    model_summaries = [[summary(m.encoder, verbose=0), summary(m.decoder.node_classifier, verbose=0), summary(m.decoder, verbose=0)] for m in models]
    model_n_params = [[n[0].total_params, n[1].total_params, n[2].total_params - n[1].total_params] for n in model_summaries]
    run_descriptions = [utils.describe_run(r) for r in runs]
    title = [f"{d['collection']}" for d in run_descriptions]
    plts = [_plot_loss(label, t, r, p, run.logger) for label, t, r, p in zip(loss, title, runs, model_n_params)]
    if args.visualise:
        plt.show()
