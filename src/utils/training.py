"""Training utilities: flags, LR schedule, loss functions, logging.

Extracted from resources/CFM/train.py and resources/CFM/utils.py.
"""

import math

import numpy as np
import torch
from torch import Tensor
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def set_flags():
    """Set performance flags and seed for reproducibility."""
    torch.manual_seed(159753)
    np.random.seed(159753)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def get_lr(cfg: DictConfig, step: int) -> float:
    """Linear warm-up followed by linear decay back to min_lr until max_steps.

    Caps at min_lr after that.
    """
    min_lr, max_lr = cfg.trainer.min_lr, cfg.trainer.max_lr
    warmup, max_steps = cfg.trainer.warmup_steps, cfg.trainer.max_steps

    if step < warmup:
        lr = min_lr + (max_lr - min_lr) * step / warmup
    elif step <= max_steps:
        decay_ratio = (step - warmup) / (max_steps - warmup)
        lr = max_lr - (max_lr - min_lr) * decay_ratio
    else:
        lr = min_lr

    return max(min_lr, min(lr, max_lr))


def get_loss_fn(model, flow):
    """Create loss function for conditional flow matching.

    Args:
        model: Neural network that predicts velocity field.
        flow: Flow object with step() and target() methods.

    Returns:
        Callable that takes a batch tensor and returns the MSE loss.
    """
    mse = torch.nn.MSELoss()

    def loss_fn(batch: Tensor) -> Tensor:
        t = torch.rand(batch.shape[0], device=batch.device)
        x0 = torch.randn_like(batch)

        xt = flow.step(t, x0, batch)
        pred_vel = model(xt, t)
        true_vel = flow.target(t, x0, batch)

        loss = mse(pred_vel, true_vel)
        return loss

    return loss_fn


def print_steps_info(cfg: DictConfig, loader: DataLoader):
    """Print training step statistics."""
    batches_per_epoch = len(loader)
    effective_samples = batches_per_epoch * loader.batch_size
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / cfg.trainer.accumulation_steps
    )

    print(
        f"samples/epoch = {effective_samples:,}  |  "
        f"batches/epoch = {batches_per_epoch:,}  |  "
        f"optimizer-steps/epoch = {optimizer_steps_per_epoch:,}  "
        f"(accum_steps = {cfg.trainer.accumulation_steps})"
    )

    return effective_samples, batches_per_epoch, optimizer_steps_per_epoch
