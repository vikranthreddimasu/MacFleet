"""MacFleet general-purpose distributed compute.

Submit arbitrary Python functions to the pool for execution across
Apple Silicon Macs. The compute layer sits alongside ML training —
both share the same transport infrastructure.

Usage (single-node or multi-node):
    with macfleet.Pool() as pool:
        results = pool.map(process_image, image_paths)
        future = pool.submit(expensive_fn, data)
        result = pool.run(analyze, dataset)
"""

from macfleet.compute.dispatch import TaskDispatcher
from macfleet.compute.models import (
    RemoteTaskError,
    TaskFuture,
    TaskResult,
    TaskSpec,
)
from macfleet.compute.worker import TaskWorker

__all__ = [
    "RemoteTaskError",
    "TaskDispatcher",
    "TaskFuture",
    "TaskResult",
    "TaskSpec",
    "TaskWorker",
]
