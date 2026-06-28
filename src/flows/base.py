"""Abstract base class for flow matching methods."""

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor


class BaseFlow(ABC):
    """Base class for flow matching probability paths.

    All flows must implement:
    - step(): interpolate between noise x0 and data x1 at time t
    - target(): compute the velocity target for training
    - sample(): generate samples using ODE integration
    """

    @abstractmethod
    def step(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Interpolate between noise and data at time t.

        Args:
            t: Time tensor of shape (B,).
            x0: Noise tensor of shape (B, C, H, W).
            x1: Data tensor of shape (B, C, H, W).

        Returns:
            Interpolated tensor x_t of shape (B, C, H, W).
        """
        ...

    @abstractmethod
    def target(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute the velocity target for flow matching loss.

        Args:
            t: Time tensor of shape (B,).
            x0: Noise tensor of shape (B, C, H, W).
            x1: Data tensor of shape (B, C, H, W).

        Returns:
            Target velocity tensor of shape (B, C, H, W).
        """
        ...

    @abstractmethod
    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        num_steps: int = 5,
        device: str = "cuda",
    ) -> Tensor:
        """Generate samples by integrating the learned velocity field.

        Args:
            model: Trained velocity prediction model.
            shape: Shape of samples to generate (B, C, H, W).
            num_steps: Number of ODE integration steps.
            device: Device to generate on.

        Returns:
            Generated samples tensor.
        """
        ...
