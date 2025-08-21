from torch import optim, nn
from torch.optim import lr_scheduler

from kwt import kwt_from_name, KWT

def get_scheduler(optimizer: optim.Optimizer, scheduler_type: str, T_max: int) -> lr_scheduler._LRScheduler:

    if scheduler_type == "cosine_annealing":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-8)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


def get_optimizer(net: nn.Module, opt_config: dict) -> optim.Optimizer:

    if opt_config["opt_type"] == "adamw":
        optimizer = optim.AdamW(net.parameters(), **opt_config["opt_kwargs"])
    else:
        raise ValueError(f'Unsupported optimizer {opt_config["opt_type"]}')

    return optimizer


def get_model(model_config: dict) -> nn.Module:

    if model_config["name"] is not None:
        return kwt_from_name(model_config["name"])
    else:
        return KWT(**model_config)