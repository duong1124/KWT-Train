import torch
import yaml
import random
import os
import numpy as np
import torch.nn as nn

def get_config(config_file: str) -> dict:

    with open(config_file, "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    if base_config["exp"]["device"] == "auto":
        base_config["exp"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_config["hparams"]["device"] = base_config["exp"]["device"]

    return base_config

def seed_everything(seed: str) -> None:

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Set seed {seed}')


def count_params(model: nn.Module) -> int:
    return sum(map(lambda p: p.data.numel(), model.parameters()))


def calc_step(epoch: int, n_batches: int, batch_index: int) -> int:
    return (epoch - 1) * n_batches + (1 + batch_index)


def log(log_dict: dict, step: int, config: dict) -> None:
    log_message = f"Step: {step} | " + " | ".join([f"{k}: {v}" for k, v in log_dict.items()])

    # write logs to disk
    if config["exp"]["log_to_file"]:
        log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

        with open(log_file, "a+") as f:
            f.write(log_message + "\n")

    # show logs in stdout
    if config["exp"]["log_to_stdout"]:
        print(log_message)