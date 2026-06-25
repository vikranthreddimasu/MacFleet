"""MacFleet: Distributed ML training across Apple Silicon Macs over Thunderbolt."""

import logging

from macfleet.core.config import (
    ClusterConfig,
    ClusterState,
    NodeConfig,
    NodeRole,
    TrainingConfig,
)

__version__ = "0.2.0"

# Configure a NullHandler so library users can control logging.
# Applications (CLI, scripts) should call logging.basicConfig() to see output.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name: str):
    """Lazy imports for heavy classes (avoid importing torch/grpc at module load)."""
    if name == "Coordinator":
        from macfleet.core.coordinator import Coordinator
        return Coordinator
    if name == "Worker":
        from macfleet.core.worker import Worker
        return Worker
    if name == "Trainer":
        from macfleet.training.trainer import Trainer
        return Trainer
    raise AttributeError(f"module 'macfleet' has no attribute {name!r}")


__all__ = [
    "__version__",
    "ClusterConfig",
    "ClusterState",
    "NodeConfig",
    "NodeRole",
    "TrainingConfig",
    "Coordinator",
    "Worker",
    "Trainer",
]
