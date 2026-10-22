"""Pluggable training engines (PyTorch, MLX).

Both engines implement the same Engine protocol so the pool/comm layer
stays framework-agnostic. Gradients flow as numpy arrays.

Torch and MLX are BOTH optional at import time. Installing the core package
does not require either framework. `from macfleet.engines import TorchEngine`
only succeeds when `pip install macfleet[torch]` (or [all]) was run.
"""

from importlib.util import find_spec
from typing import TYPE_CHECKING

from macfleet.engines.base import (
    Engine,
    EngineCapabilities,
    EngineType,
    HardwareProfile,
    ThermalPressure,
    TrainingMetrics,
)

__all__ = [
    "Engine",
    "EngineCapabilities",
    "EngineType",
    "HardwareProfile",
    "ThermalPressure",
    "TrainingMetrics",
]

if TYPE_CHECKING:
    from macfleet.engines.mlx_engine import MLXEngine  # noqa: F401
    from macfleet.engines.torch_engine import TorchEngine  # noqa: F401

if find_spec("torch") is not None:
    __all__.append("TorchEngine")
__all__.append("MLXEngine")


def __getattr__(name: str):
    if name == "TorchEngine" and name in __all__:
        from macfleet.engines.torch_engine import TorchEngine
        return TorchEngine
    if name == "MLXEngine":
        from macfleet.engines.mlx_engine import MLXEngine
        return MLXEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
