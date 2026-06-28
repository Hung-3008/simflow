"""Flow factory and registry.

Usage:
    from src.flows import create_flow
    flow = create_flow(cfg)
"""

from src.utils.registry import Registry
from omegaconf import DictConfig

# Global flow registry
FLOW_REGISTRY = Registry("flow")


def create_flow(cfg: DictConfig):
    """Create a flow instance from config.

    Config format:
        flow:
          name: "optimal_transport"
          sigma_min: 1e-4
    """
    flow_cfg = dict(cfg.flow)
    name = flow_cfg.pop("name")
    return FLOW_REGISTRY.create(name, **flow_cfg)


# Import submodules to trigger registration
import src.flows.optimal_transport  # noqa: F401, E402
