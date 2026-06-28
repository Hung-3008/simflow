"""Training entry point for flow matching models.

Refactored from resources/CFM/train.py — training loop logic preserved as-is.
Uses factory pattern for model, flow, and dataset creation.

Usage:
    python -m src.train
    python -m src.train data.dataset=stl10 data.img_size=64
"""

import os
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.model import create_model
from src.flows import create_flow
from src.datasets import create_dataloaders
from src.utils.checkpoint import make_checkpoint, load_checkpoint
from src.utils.visualization import make_im_grid
from src.utils.training import set_flags, get_lr, get_loss_fn, print_steps_info


def _resolve_data_root(cfg):
    """Resolve data root (and optional val_root) relative to original working directory."""
    root = Path(get_original_cwd()) / cfg.data.root
    root.mkdir(parents=True, exist_ok=True)
    download = bool(cfg.data.download) and not any(root.iterdir())
    cfg.data.root = str(root)
    cfg.data.download = download

    # Resolve val_root if specified
    if cfg.data.get("val_root") is not None:
        val_root = Path(get_original_cwd()) / cfg.data.val_root
        cfg.data.val_root = str(val_root)

    return cfg


@torch.no_grad()
def eval_sample(cfg: DictConfig, epoch: int, model, ema_model, flow, device: str = "cuda") -> None:
    """Generate and save sample images for evaluation."""
    model.eval()
    ema_model.eval()

    print(f"Generating samples at epoch {epoch}")
    shape = (64, 3, cfg.sample.size, cfg.sample.size)

    gen_x = flow.sample(model, shape, num_steps=2, device=device)
    gen_x_ema = flow.sample(ema_model, shape, num_steps=2, device=device)
    gen_x = gen_x[-1]
    gen_x_ema = gen_x_ema[-1]

    assert gen_x.shape == shape

    image = make_im_grid(gen_x, (8, 8))
    image.save(f"samples/{epoch}.png")
    image_ema = make_im_grid(gen_x_ema, (8, 8))
    image_ema.save(f"samples/ema_{epoch}.png")


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    os.makedirs("samples", exist_ok=True)

    set_flags()
    cfg = _resolve_data_root(cfg)
    device = "cuda"

    # --- Factory-based component creation ---
    model = create_model(cfg).to(device)
    # dynamic=True: compile once for any batch size (avoids recompilation)
    if cfg.trainer.get("compile", True):
        model = torch.compile(model, dynamic=True)

    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
    )

    flow = create_flow(cfg)

    # --- Training setup ---
    loss_fn = get_loss_fn(model, flow)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.trainer.min_lr)
    train_loader, _ = create_dataloaders(cfg)
    scaler = torch.amp.GradScaler()

    print_steps_info(cfg, train_loader)

    # --- Checkpoint resume ---
    ckpt_path = cfg.trainer.get("ckpt", None)
    if ckpt_path is not None:
        step, curr_epoch, model, optim, scaler, ema_model = load_checkpoint(
            ckpt_path, model, optim, scaler, ema_model
        )
        print(f"Loaded checkpoint [step {step} ({curr_epoch})]")
    else:
        step = 0
        curr_epoch = 0

    accumulation_steps = int(cfg.trainer.accumulation_steps)

    # --- Training loop (preserved from CFM) ---
    for epoch in range(curr_epoch, cfg.trainer.epochs + 1):
        model.train()
        ema_model.train()

        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)

            if i % accumulation_steps == 0:
                optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device):
                loss = loss_fn(x) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optim)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optim)
                scaler.update()

                ema_model.update_parameters(model)

                for g in optim.param_groups:
                    lr = get_lr(cfg, step)
                    g["lr"] = lr

                if (step + 1) % cfg.trainer.log_freq == 0:
                    true_loss = loss.item() * accumulation_steps
                    print(
                        f"Step: {step} ({epoch}) | Loss: {true_loss:.5f} | Grad: {grad.item():.5f} | Lr: {lr:.3e}"
                    )

                step += 1

        eval_sample(cfg, epoch, model, ema_model, flow, device=device)

        # Periodic checkpoint save
        save_freq = cfg.trainer.get("save_freq", 50)
        if (epoch + 1) % save_freq == 0:
            make_checkpoint(
                f"ckp_epoch{epoch+1}_step{step}.tar",
                step, epoch, model, optim, scaler, ema_model,
            )
            print(f"Checkpoint saved at epoch {epoch+1}, step {step}")

    make_checkpoint(f"ckp_final_step{step}.tar", step, epoch, model, optim, scaler, ema_model)


if __name__ == "__main__":
    main()
