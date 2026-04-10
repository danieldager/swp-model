from __future__ import annotations

from typing import Callable, Type

from swp.audio.models.base import AudioModel

_REGISTRY: dict[str, Type[AudioModel]] = {}


def register(name: str) -> Callable[[Type[AudioModel]], Type[AudioModel]]:
    """Class decorator that registers an AudioModel implementation.

    Usage:
        @register("encodec")
        class EnCodecModel:
            ...
    """
    def decorator(cls: Type[AudioModel]) -> Type[AudioModel]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> AudioModel:
    """Instantiate a registered AudioModel by name.

    Args:
        name:   model name as registered with @register
        **kwargs: forwarded to the model constructor

    Raises:
        KeyError: if name is not registered
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown model {name!r}. Available: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())