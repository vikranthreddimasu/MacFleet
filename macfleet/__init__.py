"""MacFleet v2: Pool Apple Silicon Macs into a distributed ML training cluster.

Zero-config discovery. Framework-agnostic engines. Adaptive networking.

    pip install macfleet && macfleet join
"""

import logging

__version__ = "2.2.0rc1"

logging.getLogger(__name__).addHandler(logging.NullHandler())


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
    "task",
]
