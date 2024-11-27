"""Train a rocnet model"""

import argparse
import os.path as pth
import signal

import rocnet.utils
from rocnet.dataset import Dataset
from rocnet.rocnet import DEFAULT_CONFIG as MODEL_DEFAULTS
from rocnet.trainer import DEFAULT_CONFIG as TRAIN_DEFAULTS
from rocnet.trainer import Trainer, check_training_cfg

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train.py", description="Start rocnet training")
    parser.add_argument("folder", help="Output <folder> which will contain train.toml and one or more train_<TIMESTAMP>")

    args = parser.parse_args()

    TRAIN_DEFAULTS["model"] = MODEL_DEFAULTS
    run = utils.Run(args.folder, "train", "train", True, TRAIN_DEFAULTS)
    check_training_cfg(run.cfg)

    try:
        run.git_snapshot()
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
        max_test_samples = run.cfg.max_samples * (1 - dataset.metadata.train_fraction)
    else:
        max_test_samples = None

    valid_dataset = Dataset(run.cfg.dataset_path, run.cfg.model.grid_dim, train=False, max_samples=max_test_samples, file_list=pth.join(run.dir, "valid_files.csv"))
    trainer = Trainer(run.run_dir, run.cfg, dataset, valid_dataset)
    rocnet.utils.save_file(pth.join(run.run_dir, "train.toml"), run.cfg, False)
    trainer.train(on_epoch)
