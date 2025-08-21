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
