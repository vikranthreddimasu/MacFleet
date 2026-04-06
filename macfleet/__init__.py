"""MacFleet v2: Pool Apple Silicon Macs into a distributed ML training cluster.

Zero-config discovery. Framework-agnostic engines. Adaptive networking.

    pip install macfleet && macfleet join
"""

import logging

__version__ = "2.0.0a1"

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name: str):
    """Lazy imports for heavy modules (avoid importing torch/mlx at module load)."""
    if name == "Pool":
        from macfleet.sdk.pool import Pool
        return Pool
    if name == "train":
        from macfleet.sdk.train import train
        return train
    if name == "DataParallel":
        from macfleet.training.data_parallel import DataParallel
        return DataParallel
    raise AttributeError(f"module 'macfleet' has no attribute {name!r}")


__all__ = [
    "__version__",
    "Pool",
    "train",
    "DataParallel",
]
