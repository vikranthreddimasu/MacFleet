"""N-node data parallel training strategy for MacFleet v2.

Synchronizes gradients across all pool members using AllReduce.
Framework-agnostic: works through the Engine protocol with numpy
as the intermediate representation.

Data parallel flow (each training step):
    1. Each node runs forward + backward on its weighted batch portion
    2. Gradients flattened to numpy array via engine.get_flat_gradients()
    3. AllReduced via CollectiveGroup (ring or direct exchange)
    4. Averaged gradients applied via engine.apply_flat_gradients()
    5. Each node runs optimizer.step() (identical updates → identical params)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from macfleet.comm.collectives import CollectiveGroup
from macfleet.engines.base import TrainingMetrics


@dataclass
class DataParallelConfig:
    """Configuration for data parallel training."""

    # Gradient sync
    bucket_size_mb: float = 25.0  # group gradients into communication buckets
    # Staleness tolerance for async gradient sync (0 = synchronous)
    max_staleness: int = 0
    # Broadcast parameters from coordinator on start
    broadcast_params_on_start: bool = True


class DataParallel:
    """N-node data parallel gradient synchronization.

    Ties together an engine (TorchEngine/MLXEngine) and a
    CollectiveGroup to synchronize gradients after each backward pass.

    The engine handles all framework-specific operations (forward, backward,
    optimizer). This class only touches gradients as numpy arrays.

    Usage:
        dp = DataParallel(engine, group)
        await dp.setup()                  # broadcast initial params
        # ... training loop ...
        await dp.sync_gradients()         # after backward, before step
        await dp.broadcast_parameters()   # explicit param sync
    """

    def __init__(
        self,
        engine: object,  # Engine protocol (TorchEngine or MLXEngine)
        group: CollectiveGroup,
        config: Optional[DataParallelConfig] = None,
    ):
        self.engine = engine
        self.group = group
        self.config = config or DataParallelConfig()
        self._step_count = 0
        self._sync_time_sec = 0.0

    @property
    def world_size(self) -> int:
        return self.group.world_size

    @property
    def rank(self) -> int:
        return self.group.rank

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    @property
    def avg_sync_time_sec(self) -> float:
        """Average gradient sync time over all steps."""
        if self._step_count == 0:
            return 0.0
        return self._sync_time_sec / self._step_count

    async def setup(self) -> None:
        """Initialize data parallel training.

        Broadcasts model parameters from rank 0 to all nodes so
        everyone starts from the same weights.
        """
        if self.config.broadcast_params_on_start and self.world_size > 1:
            await self.broadcast_parameters()

    async def sync_gradients(self) -> float:
        """AllReduce gradients across all nodes.

        Call after backward() and before step().

        Returns:
            Time spent in gradient sync (seconds).
        """
        if self.world_size == 1:
            return 0.0

        t0 = time.monotonic()

        # Extract gradients as flat numpy array
        flat_grads = self.engine.get_flat_gradients()

        # AllReduce (mean across all nodes)
        averaged = await self.group.allreduce(flat_grads, op="mean")

        # Write averaged gradients back to model
        self.engine.apply_flat_gradients(averaged)

        elapsed = time.monotonic() - t0
        self._step_count += 1
        self._sync_time_sec += elapsed
        return elapsed

    async def broadcast_parameters(self, src: int = 0) -> None:
        """Broadcast model parameters from src rank to all nodes.

        Ensures all nodes have identical weights before training starts.
        """
        if self.world_size == 1:
            return

        flat_params = self.engine.get_flat_parameters()
        synced = await self.group.broadcast(flat_params, src=src)
        self.engine.apply_flat_parameters(synced)

    def metrics(self) -> TrainingMetrics:
        """Get current training metrics."""
        return TrainingMetrics(
            step_time_sec=self.avg_sync_time_sec,
        )
