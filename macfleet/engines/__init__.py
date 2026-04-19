"""Pluggable training engines (PyTorch, MLX).

Both engines implement the same Engine protocol so the pool/comm layer
stays framework-agnostic. Gradients flow as numpy arrays.

Torch and MLX are BOTH optional at import time. Installing the core package
does not require either framework. `from macfleet.engines import TorchEngine`
only succeeds when `pip install macfleet[torch]` (or [all]) was run.
"""

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

# PyTorch is optional — install via `pip install macfleet[torch]`
try:
    from macfleet.engines.torch_engine import TorchEngine  # noqa: F401

    __all__.append("TorchEngine")
except ImportError:
    pass

# MLX is optional — only available on Apple Silicon with mlx installed
try:
    from macfleet.engines.mlx_engine import MLXEngine  # noqa: F401

    __all__.append("MLXEngine")
except ImportError:
    pass
