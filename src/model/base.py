"""Abstract base class for generative models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Base class for all models in the flow matching framework.

    All models must implement the forward() method with at minimum
    (x, t) inputs. Models intended for SiMFlow should also accept
    an optional `r` parameter for dual-time conditioning.
    """

    @abstractmethod
    def forward(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).
            t: Time tensor of shape (B,).
            **kwargs: Additional conditioning (e.g., r for SiMFlow, labels).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        ...

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        num_params = self.get_num_params()
        return (
            f"{self.__class__.__name__}("
            f"params={num_params:,}"
            f")"
        )
