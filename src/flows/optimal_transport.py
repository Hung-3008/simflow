"""Optimal Transport conditional flow matching.

Copied from resources/CFM/flow.py — all math preserved as-is.
Refactored to use BaseFlow and flow registry.
"""

import torch
from torch import nn, Tensor
from torchdiffeq import odeint

from src.flows.base import BaseFlow
from src.flows import FLOW_REGISTRY


@FLOW_REGISTRY.register("optimal_transport")
class OptimalTransportFlow(BaseFlow):
    """Optimal Transport probability path for conditional flow matching.

    Implements the linear interpolation path:
        x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1

    with velocity target:
        v_t = x1 - (1 - sigma_min) * x0
    """

    def __init__(self, sigma_min: float = 1e-2):
        super().__init__()
        self.sigma_min = sigma_min

    @torch.compile
    def step(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        t = t[:, None, None, None]
        mu = t * x1
        sigma = 1 - (1 - self.sigma_min) * t
        return sigma * x0 + mu

    @torch.compile
    def target(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        return x1 - (1 - self.sigma_min) * x0

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape: tuple = (64, 3, 32, 32),
        num_steps: int = 5,
        device: str = "cuda",
    ) -> Tensor:
        """Generate samples using Dormand-Prince ODE solver.

        Args:
            model: Trained velocity prediction model.
            shape: Shape of samples (B, C, H, W).
            num_steps: Number of time steps for ODE solver.
            device: Device for computation.

        Returns:
            Tensor of shape (num_steps, B, C, H, W) with trajectories.
        """
        model.eval()

        x0 = torch.randn(shape, device=device)
        timesteps = torch.linspace(0.0, 1.0, num_steps, device=device)

        samples = odeint(
            func=lambda t, x: model(x, t.repeat(shape[0])),
            t=timesteps,
            y0=x0,
            method="dopri5",
            atol=1e-5,
            rtol=1e-5,
        )
        return samples
