"""Checkpoint save/load utilities.

Copied from resources/CFM/utils.py — checkpoint logic preserved as-is.
"""

import torch


def make_checkpoint(path, step, epoch, model, optim=None, scaler=None, ema_model=None):
    """Save training checkpoint to disk."""
    checkpoint = {
        "epoch": int(epoch),
        "step": int(step),
        "model_state_dict": model.state_dict(),
    }

    if optim is not None:
        checkpoint["optim_state_dict"] = optim.state_dict()

    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optim=None, scaler=None, ema_model=None):
    """Load training checkpoint from disk."""
    checkpoint = torch.load(path, weights_only=True)
    step = int(checkpoint["step"])
    epoch = int(checkpoint["epoch"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if optim is not None:
        optim.load_state_dict(checkpoint["optim_state_dict"])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        ema_model.eval()

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    model.eval()

    return step, epoch, model, optim, scaler, ema_model
