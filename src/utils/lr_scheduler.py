import math

from torch.optim.lr_scheduler import LambdaLR
from transformers import get_constant_schedule_with_warmup

class ConstantWithWarmup:
    def __init__(self, num_warmup_steps: int):
        self.num_warmup_steps = num_warmup_steps

    def __call__(self, optimizer):
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps
        )


class WarmupCosineDecay:
    """Linear warmup followed by cosine decay to a floor ratio."""

    def __init__(
        self,
        total_steps: int,
        num_warmup_steps: int = 0,
        min_lr_ratio: float = 0.1,
    ):
        self.total_steps = int(total_steps)
        self.num_warmup_steps = int(num_warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.num_warmup_steps < 0:
            raise ValueError("num_warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

    def __call__(self, optimizer):
        warmup = min(self.num_warmup_steps, self.total_steps)
        decay_steps = max(self.total_steps - warmup, 1)

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step) / float(max(1, warmup))

            progress = min(float(step - warmup) / float(decay_steps), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda)


class WarmupHoldCosineDecay:
    """Linear warmup, fixed LR hold, then cosine decay to a floor ratio."""

    def __init__(
        self,
        total_steps: int,
        num_warmup_steps: int = 0,
        hold_steps: int = 0,
        min_lr_ratio: float = 0.1,
    ):
        self.total_steps = int(total_steps)
        self.num_warmup_steps = int(num_warmup_steps)
        self.hold_steps = int(hold_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.num_warmup_steps < 0:
            raise ValueError("num_warmup_steps must be non-negative")
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")

    def __call__(self, optimizer):
        warmup = min(self.num_warmup_steps, self.total_steps)
        hold_end = min(max(self.hold_steps, warmup), self.total_steps)
        decay_steps = max(self.total_steps - hold_end, 1)

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step) / float(max(1, warmup))
            if step < hold_end:
                return 1.0

            progress = min(float(step - hold_end) / float(decay_steps), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda)
