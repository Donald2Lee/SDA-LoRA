import argparse
import os
from pathlib import Path

import numpy as np
import torch

from models.solvers import Solvers
from process_data import SEEDProcessor
from utils.logger import Logger
from utils.other_utils import load_yaml_config, merge_opts_to_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="SDA-LoRA training for SEED")
    parser.add_argument("--name", type=str, required=True, help="experiment name")
    parser.add_argument("--config_file", type=str, required=True, help="path of config file")
    parser.add_argument("--output", type=str, default="outputs", help="result directory")
    parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    parser.add_argument("--cudnn_deterministic", action="store_true", help="use deterministic cuDNN")
    parser.add_argument("--seed", type=int, default=616, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--adapt_src", action="store_true", help="adapt LaBraM to source subjects")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="override YAML options")
    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, args.name)
    return args


def validate_entry(args):
    config_path = Path(args.config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if args.gpu < 0:
        raise ValueError("gpu must be non-negative")
    if args.opts and len(args.opts) % 2:
        raise ValueError("Config overrides must be KEY VALUE pairs")


def log_summary(logger, config, all_metrics, adapt_src):
    logger.log_info("=" * 60)
    if adapt_src:
        logger.log_info("All target source-adaptation results")
        for target, metrics in all_metrics.items():
            logger.log_info(
                f"target={target} | Train Acc/F1: "
                f"{metrics['source_train_acc']:.4f}/{metrics['source_train_f1']:.4f}"
                f" | Val Acc/F1: {metrics['source_val_acc']:.4f}/{metrics['source_val_f1']:.4f}"
            )
        val_accs = [metrics["source_val_acc"] for metrics in all_metrics.values()]
        val_f1s = [metrics["source_val_f1"] for metrics in all_metrics.values()]
        logger.log_info(
            f"All targets summary | Val Acc: {np.mean(val_accs):.4f} ({np.std(val_accs):.4f})"
            f" | Val F1: {np.mean(val_f1s):.4f} ({np.std(val_f1s):.4f})"
        )
        return

    method = config["adapt"]["method"]
    logger.log_info(f"All target adaptation results | Method={method}")
    for shot in config["data"]["target_shot_per_class"]:
        accs = [metrics[method][shot]["target_query_acc"] for metrics in all_metrics.values()]
        f1s = [metrics[method][shot]["target_query_f1"] for metrics in all_metrics.values()]
        logger.log_info(
            f"Shot={shot} | Acc: {np.mean(accs):.4f} ({np.std(accs):.4f})"
            f" | F1: {np.mean(f1s):.4f} ({np.std(f1s):.4f})"
        )


def main():
    args = parse_args()
    validate_entry(args)
    seed_everything(args.seed, args.cudnn_deterministic)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"using GPU {args.gpu}")

    config = merge_opts_to_config(load_yaml_config(args.config_file), args.opts)
    logger = Logger(args)
    logger.save_config(config)

    processor = SEEDProcessor(
        raw_root=config["data"]["raw_root"],
        processed_root=config["data"]["processed_root"],
        window_sec=config["data"]["window_sec"],
        sample_rate=config["data"]["sample_rate"],
    )
    processor.save_to_pt()

    all_metrics = {}
    for target in config["data"]["target_subjects"]:
        datasets = processor.load_dataset(config, target=target)
        solver = Solvers(config, args, target, datasets, logger)
        all_metrics[target] = solver.adapt_src() if args.adapt_src else solver.adapt()

    log_summary(logger, config, all_metrics, args.adapt_src)


if __name__ == "__main__":
    main()
