import random
import warnings

import numpy as np
import torch
import yaml


def load_yaml_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def merge_opts_to_config(config, opts):
    if not opts:
        return config
    for index in range(0, len(opts), 2):
        keys = opts[index].split(".")
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = yaml.safe_load(opts[index + 1])
    return config


def seed_everything(seed, cudnn_deterministic=False):
    print(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = not cudnn_deterministic
    if cudnn_deterministic:
        warnings.warn("Deterministic cuDNN may reduce training speed.")
