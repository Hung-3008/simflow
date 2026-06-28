"""Generic registry pattern for factory-based component creation."""

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A generic registry that maps string names to classes.

    Usage:
        MODEL_REGISTRY = Registry("model")

        @MODEL_REGISTRY.register("unet")
        class Unet(nn.Module):
            ...

        model = MODEL_REGISTRY.create("unet", ch=128, ...)
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a class under a given name."""
        def decorator(cls: Type) -> Type:
            key = name or cls.__name__
            if key in self._registry:
                raise ValueError(
                    f"[{self._name}] '{key}' is already registered "
                    f"({self._registry[key].__name__}). Cannot register {cls.__name__}."
                )
            self._registry[key] = cls
            return cls
        return decorator

    def create(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered class by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"[{self._name}] '{name}' not found. Available: [{available}]"
            )
        return self._registry[name](**kwargs)

    def get(self, name: str) -> Type:
        """Get a registered class by name without instantiation."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"[{self._name}] '{name}' not found. Available: [{available}]"
            )
        return self._registry[name]

    def list(self) -> list:
        """Return list of registered names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, entries={self.list()})"
