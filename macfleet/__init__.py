"""MacFleet v2: Pool Apple Silicon Macs into a distributed ML training cluster.

Zero-config discovery. Framework-agnostic engines. Adaptive networking.

    pip install macfleet && macfleet join
"""

import logging
from typing import TYPE_CHECKING

__version__ = "2.2.1"

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Type checkers see real symbols here; runtime uses __getattr__ below to
# keep heavy framework imports off the cold path.
if TYPE_CHECKING:
    from macfleet.compute.authz import TaskAuthorizationError, TaskAuthorizationPolicy
    from macfleet.compute.models import RemoteTaskError, TaskFuture
    from macfleet.compute.registry import task
    from macfleet.engines.mlx_engine import MLXEngine
    from macfleet.engines.torch_engine import TorchEngine
    from macfleet.sdk.decorators import distributed
    from macfleet.sdk.pool import Pool
    from macfleet.sdk.train import train
    from macfleet.training.data_parallel import DataParallel


def __getattr__(name: str):
    """Lazy imports for heavy modules (avoid importing torch/mlx at module load)."""
    if name == "Pool":
        from macfleet.sdk.pool import Pool
        return Pool
    if name == "train":
        from macfleet.sdk.train import train
        return train
    if name == "distributed":
        from macfleet.sdk.decorators import distributed
        return distributed
    if name == "DataParallel":
        from macfleet.training.data_parallel import DataParallel
        return DataParallel
    if name == "TorchEngine":
        from macfleet.engines.torch_engine import TorchEngine
        return TorchEngine
    if name == "MLXEngine":
        from macfleet.engines.mlx_engine import MLXEngine
        return MLXEngine
    if name == "TaskFuture":
        from macfleet.compute.models import TaskFuture
        return TaskFuture
    if name == "RemoteTaskError":
        from macfleet.compute.models import RemoteTaskError
        return RemoteTaskError
    if name == "TaskAuthorizationError":
        from macfleet.compute.authz import TaskAuthorizationError
        return TaskAuthorizationError
    if name == "TaskAuthorizationPolicy":
        from macfleet.compute.authz import TaskAuthorizationPolicy
        return TaskAuthorizationPolicy
    # v2.2 PR 7: @macfleet.task decorator
    if name == "task":
        from macfleet.compute.registry import task
        return task
    raise AttributeError(f"module 'macfleet' has no attribute {name!r}")


__all__ = [
    "__version__",
    "Pool",
    "train",
    "distributed",
    "DataParallel",
    "TorchEngine",
    "MLXEngine",
    "TaskFuture",
    "RemoteTaskError",
    "TaskAuthorizationError",
    "TaskAuthorizationPolicy",
    "task",
]
