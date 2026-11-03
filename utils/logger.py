import os
import sys
import time

import torch
import yaml


class Logger:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.save_dir, exist_ok=True)
        self.config_dir = os.path.join(args.save_dir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)
        write_args(args, os.path.join(self.config_dir, "args.txt"))
        log_dir = os.path.join(args.save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.text_writer = open(os.path.join(log_dir, "log.txt"), "a", encoding="utf-8")
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            self.log_info("using tensorboard")
        else:
            self.tb_writer = None

    def log_info(self, info):
        print(info)
        message = f"{time.strftime('%Y-%m-%d-%H-%M')}: {info}"
        self.text_writer.write(message + ("" if message.endswith("\n") else "\n"))
        self.text_writer.flush()

    def add_scalars(self, **kwargs):
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(**kwargs)

    def save_config(self, config):
        with open(os.path.join(self.config_dir, "config.yaml"), "w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)


def write_args(args, path):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(f"==> torch version: {torch.__version__}\n")
        stream.write(f"==> cudnn version: {torch.backends.cudnn.version()}\n")
        stream.write(f"==> Cmd:\n{sys.argv}\n==> args:\n")
        for key, value in sorted(vars(args).items()):
            stream.write(f"  {key}: {value}\n")
