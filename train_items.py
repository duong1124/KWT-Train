import torch

from torch import optim, nn
from torch.optim import lr_scheduler

class WarmUpLR(lr_scheduler._LRScheduler):

    def __init__(self, optimizer: optim.Optimizer, total_iters: int, last_epoch: int = -1):
        """Initializer for WarmUpLR"""

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Learning rate will be set to base_lr * last_epoch / total_iters."""

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class LabelSmoothingLoss(nn.Module):

    def __init__(self, num_classes: int, smoothing : float = 0.1, dim : int = -1):

        super().__init__()

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = num_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


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


def save_model(epoch: int, val_acc: float, save_path: str, net: nn.Module, optimizer : optim.Optimizer = None, log_file : str = None) -> None:

    ckpt_dict = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else optimizer
    }

    torch.save(ckpt_dict, save_path)

    log_message = f"Saved {save_path} with accuracy {val_acc}."
    print(log_message)

    if log_file is not None:
        with open(log_file, "a+") as f:
            f.write(log_message + "\n")
