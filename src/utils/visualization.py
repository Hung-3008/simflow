"""Visualization utilities for generated images.

Copied from resources/CFM/utils.py — image grid and unloader preserved as-is.
"""

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from einops import rearrange


unloader = v2.Compose(
    [
        v2.Lambda(lambda t: (t + 1) * 0.5),
        v2.Lambda(lambda t: t.permute(0, 2, 3, 1)),
        v2.Lambda(lambda t: t * 255.0),
    ]
)


def make_im_grid(x0: torch.Tensor, xy: tuple = (1, 10)):
    """Create a PIL image grid from a batch of tensors.

    Args:
        x0: Image tensor of shape (B, C, H, W) in [-1, 1] range.
        xy: Tuple (rows, cols) for grid layout.

    Returns:
        PIL Image of the grid.
    """
    x, y = xy
    im = unloader(x0.cpu())
    B, C, H, W = x0.shape
    im = (
        rearrange(im, "(x y) h w c -> (x h) (y w) c", x=B // x, y=B // y)
        .numpy()
        .astype(np.uint8)
    )
    im = v2.ToPILImage()(im)
    return im
