"""Model factory and registry.

Usage:
    from src.model import create_model
    model = create_model(cfg)
"""

from src.utils.registry import Registry
from omegaconf import DictConfig

# Global model registry — models register themselves via @MODEL_REGISTRY.register()
MODEL_REGISTRY = Registry("model")


def create_model(cfg: DictConfig):
    """Create a model instance from config.

    Config format:
        model:
          name: "unet"
          params:
            ch: 128
            ...
    """
    params = dict(cfg.model.get("params", {}))
    return MODEL_REGISTRY.create(cfg.model.name, **params)


# Import submodules to trigger registration
import src.model.unet  # noqa: F401, E402
