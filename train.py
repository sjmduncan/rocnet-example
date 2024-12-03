"""Train a rocnet model"""

import argparse
import math
import os.path as pth
import signal

import rocnet.utils
from rocnet.dataset import Dataset
from rocnet.rocnet import DEFAULT_CONFIG as MODEL_DEFAULTS
from rocnet.trainer import DEFAULT_CONFIG as TRAIN_DEFAULTS
from rocnet.trainer import Trainer, check_training_cfg
import torch


import utils


def _get_args():
    parser = argparse.ArgumentParser(prog="train.py", description="Start rocnet training")
    parser.add_argument("folder", help="Output <folder> which will contain train.toml and one or more train_<TIMESTAMP>")
    parser.add_argument("--resume-from", help="Resume from the best (in terms of loss) snapshot of a previous training run, either a training run or a collection of runs in which case the most recent run in the collection is used", default=None)
    parser.add_argument("--seed", help="Integer for torch.manual_seed, will also invoke torch.use_deterministic_algorithms()", default=None)
    return parser


if __name__ == "__main__":
    parser = _get_args()

    args = parser.parse_args()

    TRAIN_DEFAULTS["model"] = MODEL_DEFAULTS
    run = utils.Run(args.folder, "train", "train", True, TRAIN_DEFAULTS, args.seed)
    check_training_cfg(run.cfg)

    if args.resume_from is not None:
        dt = utils.dir_type(args.resume_from)
        if dt == "training-run":
            resume_path = args.resume_from
        elif dt == "run-collection":
            runs = utils.search_runs(args.resume_from)
            runs.sort()
            resume_path = runs[-1]
        else:
            raise FileNotFoundError(f"Folder does not contain a non-empty training run: {args.resume_from}")

        run_meta = torch.load(pth.join(resume_path, "model.pth"))["metadata"]
        min_epoch = run_meta["epoch"] + 1
        run.logger.info(f"Resuming: {resume_path}")
        run.logger.info(f"        : epoch {min_epoch}")

    try:
        run._git_snapshot()
    except RuntimeError as e:
        run.logger.warning(f"Failed to save git snapshot: {e}")

    stopping = False

    def on_epoch(current_epoch, max_epochs, train_loss, valid_loss):
        return stopping

    def handle_signal(sig, frame):
        global stopping
        if sig == signal.SIGTERM or sig == signal.SIGINT:
            run.logger.info(f"Caught signal {sig}. Stoppping after this epoch")
            stopping = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    dataset = Dataset(run.cfg.dataset_path, run.cfg.model.grid_dim, train=True, max_samples=run.cfg.max_samples, file_list=pth.join(run.dir, "train_files.csv"))

    if run.cfg.max_samples is not None:
        max_test_samples = int(math.ceil(run.cfg.max_samples * (1 - dataset.metadata.train_fraction)))
    else:
        max_test_samples = None

    valid_dataset = Dataset(run.cfg.dataset_path, run.cfg.model.grid_dim, train=False, max_samples=max_test_samples, file_list=pth.join(run.dir, "valid_files.csv"))
    trainer = Trainer(run.run_dir, run.cfg, dataset, valid_dataset)
    if args.resume_from is not None:
        trainer.load_snapshot(pth.join(resume_path, f"model_{min_epoch}"))
        trainer.start_epoch = min_epoch
    rocnet.utils.save_file(pth.join(run.run_dir, "train.toml"), run.cfg, False)
    trainer.train(on_epoch)
