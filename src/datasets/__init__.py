"""Dataset factory and registry.

Usage:
    from src.datasets import create_dataloaders
    train_loader, test_loader = create_dataloaders(cfg)
"""

from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.utils.registry import Registry

# Global dataset registry
DATASET_REGISTRY = Registry("dataset")


def create_dataloaders(cfg: DictConfig):
    """Create train and test data loaders from config.

    Config format:
        data:
          root: "data/"
          img_size: 32
          batch_size: 128
          num_workers: 16
          dataset: "cifar10"
          download: false
    """
    root = Path(cfg.data.root)
    root.mkdir(parents=True, exist_ok=True)
    download = bool(cfg.data.download) and not any(root.iterdir())

    name = cfg.data.dataset.lower()
    bs = cfg.data.batch_size
    nw = cfg.data.num_workers

    # Use registry to create dataset
    create_fn = DATASET_REGISTRY.get(name)
    train_set, test_set = create_fn(
        root=str(root),
        img_size=cfg.data.img_size,
        download=download,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4,
    )

    return train_loader, test_loader


# Import submodules to trigger registration
import src.datasets.image_datasets  # noqa: F401, E402
