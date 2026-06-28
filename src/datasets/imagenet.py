"""ImageNet-1K dataset for flow matching training.

Reads the flat-directory structure produced by prepare_imagenet.py:
    <root>/
        n01440764_10026.jpeg
        n01440764_10026.meta.json   # {"class": 0}
        n01440764_10027.jpeg
        ...

Each image is paired with a .meta.json sidecar containing the integer class id.
"""

import json
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from torch.utils.data import Dataset

from src.datasets import DATASET_REGISTRY


_IMG_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class FlatImageNetDataset(Dataset):
    """Dataset for AlphaFlow-style flat ImageNet directory.

    Args:
        root: Path to the flat directory (e.g. data/imagenet/train).
        transform: torchvision transform applied to each PIL image.
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform

        # Collect all image files
        self._samples = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in _IMG_EXTENSIONS
        )
        assert len(self._samples) > 0, (
            f"No images found in {self.root}. "
            "Make sure prepare_imagenet.py has been run."
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        img_path = self._samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load label from sidecar .meta.json
        meta_path = img_path.with_suffix("").with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                label = int(json.load(f)["class"])
        else:
            label = -1

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def _make_imagenet_transforms(img_size: int = 256):
    """Standard ImageNet train/val transforms.

    Train: RandomResizedCrop + RandomHorizontalFlip + Normalize
    Val:   Resize + CenterCrop + Normalize
    """
    mean = [0.5, 0.5, 0.5]
    std  = [0.5, 0.5, 0.5]

    train_tf = v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

    val_tf = v2.Compose([
        v2.Resize(img_size, antialias=True),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])

    return train_tf, val_tf


@DATASET_REGISTRY.register("imagenet")
def create_imagenet(
    root: str,
    img_size: int = 256,
    val_root: str | None = None,
    **kwargs,
):
    """Create ImageNet-1K train/val datasets.

    Args:
        root:     Path to flat train directory.
        img_size: Spatial resolution to crop/resize to.
        val_root: Path to flat val directory (defaults to root/../val).
    """
    if val_root is None:
        val_root = str(Path(root).parent / "val")

    train_tf, val_tf = _make_imagenet_transforms(img_size)
    train_set = FlatImageNetDataset(root, transform=train_tf)
    val_set   = FlatImageNetDataset(val_root, transform=val_tf)

    print(
        f"[imagenet] train={len(train_set):,} images | "
        f"val={len(val_set):,} images | "
        f"resolution={img_size}x{img_size}"
    )
    return train_set, val_set
