"""Train a rocnet model"""

import argparse
import os.path as pth
import signal

from rocnet.dataset import Dataset
from rocnet.rocnet import DEFAULT_CONFIG as MODEL_DEFAULTS
from rocnet.trainer import DEFAULT_CONFIG as TRAIN_DEFAULTS
from rocnet.trainer import Trainer
import rocnet.utils

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train.py", description="Start rocnet training")
    parser.add_argument("folder", help="output folder, containing config in train.toml and where training run output", default="../rocnet.weights/default")
    args = parser.parse_args()

    TRAIN_DEFAULTS["model"] = MODEL_DEFAULTS
    run = utils.Run(args.folder, "train", "train", True, TRAIN_DEFAULTS)

    try:
        run.git_snapshot()
    except RuntimeError as e:
        run.logger.warning(f"Failed to save git snapshot: {e}")

    stopping = False

    def on_epoch(current_epoch, max_epochs, train_loss, valid_loss, do_snapshot):
        return stopping

    def handle_signal(sig, frame):
        global stopping
        if sig == signal.SIGTERM or sig == signal.SIGINT:
            run.logger.info(f"Caught signal {sig}. Stoppping after this epoch")
            stopping = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    dataset = Dataset(run.cfg.dataset_path, run.cfg.model.grid_dim, train=True, max_samples=run.cfg.max_samples, file_list=pth.join(run.dir, "train_files.csv"))
    valid_dataset = Dataset(run.cfg.dataset_path, run.cfg.model.grid_dim, train=False, max_samples=int(run.cfg.max_samples * 0.2 / 0.8), file_list=pth.join(run.dir, "valid_files.csv"))
    trainer = Trainer(run.run_dir, run.cfg, dataset, valid_dataset)
    rocnet.utils.save_file(pth.join(run.run_dir, "train.toml"), run.cfg, False)
    trainer.train(on_epoch)
