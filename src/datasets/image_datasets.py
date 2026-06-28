"""Image dataset wrappers for flow matching training.

Copied from resources/CFM/utils.py — dataset/transform logic preserved as-is.
Refactored into dataset registry pattern.
"""

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torchvision import datasets

from src.datasets import DATASET_REGISTRY


def _make_base_transforms(size=None):
    """Create base train/test transforms.

    Args:
        size: If provided, prepend Resize + CenterCrop.

    Returns:
        (train_transform, test_transform) tuple.
    """
    base_train_tf = [
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    base_test_tf = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]

    if size is not None:
        resize_tf = [v2.Resize(size, antialias=True), v2.CenterCrop(size)]
        train_tf = v2.Compose(resize_tf + base_train_tf)
        test_tf = v2.Compose(resize_tf + base_test_tf)
    else:
        train_tf = v2.Compose(base_train_tf)
        test_tf = v2.Compose(base_test_tf)

    return train_tf, test_tf


@DATASET_REGISTRY.register("cifar10")
def create_cifar10(root, img_size=32, download=True, **kwargs):
    """Create CIFAR-10 train/test datasets."""
    train_tf, test_tf = _make_base_transforms()
    train_set = datasets.CIFAR10(root, train=True, download=download, transform=train_tf)
    test_set = datasets.CIFAR10(root, train=False, download=download, transform=test_tf)
    return train_set, test_set


@DATASET_REGISTRY.register("stl10")
def create_stl10(root, img_size=64, download=True, **kwargs):
    """Create STL-10 train/test datasets."""
    train_tf, test_tf = _make_base_transforms(size=img_size)
    train_set = datasets.STL10(root, split="unlabeled", download=download, transform=train_tf)
    test_set = datasets.STL10(root, split="test", download=download, transform=test_tf)
    return train_set, test_set
