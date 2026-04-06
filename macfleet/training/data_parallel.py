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

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from macfleet.comm.collectives import CollectiveGroup
from macfleet.compression.adaptive import (
    AdaptiveCompressor,
    AdaptiveCompressionConfig,
    CompressedArray,
    CompressionLevel,
)
from macfleet.engines.base import TrainingMetrics
from macfleet.pool.network import LinkType
from macfleet.security.auth import GradientValidationError, validate_gradients

logger = logging.getLogger(__name__)


@dataclass
class DataParallelConfig:
    """Configuration for data parallel training."""

    # Gradient sync
    bucket_size_mb: float = 25.0  # group gradients into communication buckets
    # Staleness tolerance for async gradient sync (0 = synchronous)
    max_staleness: int = 0
    # Broadcast parameters from coordinator on start
    broadcast_params_on_start: bool = True
    # Compression
    compression: str = "none"  # "none", "light", "moderate", "aggressive", "adaptive"
    compression_warmup_steps: int = 0


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
        link_type: LinkType = LinkType.UNKNOWN,
    ):
        self.engine = engine
        self.group = group
        self.config = config or DataParallelConfig()
        self._step_count = 0
        self._sync_time_sec = 0.0
        self._bytes_sent = 0
        self._bytes_saved = 0

        # Setup compression
        self._compressor = self._make_compressor(link_type)

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

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio (bytes sent / bytes uncompressed)."""
        total = self._bytes_sent + self._bytes_saved
        if total == 0:
            return 1.0
        return self._bytes_sent / total

    def _make_compressor(self, link_type: LinkType) -> Optional[AdaptiveCompressor]:
        """Create compressor based on config."""
        comp = self.config.compression
        if comp == "none":
            return None

        if comp == "adaptive":
            return AdaptiveCompressor(
                link_type=link_type,
                config=AdaptiveCompressionConfig(
                    warmup_steps=self.config.compression_warmup_steps,
                ),
            )

        level_map = {
            "light": CompressionLevel.LIGHT,
            "moderate": CompressionLevel.MODERATE,
            "aggressive": CompressionLevel.AGGRESSIVE,
        }
        level = level_map.get(comp)
        if level is None:
            return None

        return AdaptiveCompressor(
            config=AdaptiveCompressionConfig(
                fixed_level=level,
                warmup_steps=self.config.compression_warmup_steps,
            ),
        )

    async def setup(self) -> None:
        """Initialize data parallel training.

        Broadcasts model parameters from rank 0 to all nodes so
        everyone starts from the same weights.
        """
        if self.config.broadcast_params_on_start and self.world_size > 1:
            await self.broadcast_parameters()

    async def sync_gradients(self) -> float:
        """AllReduce gradients across all nodes.

        If compression is enabled, gradients are compressed before
        allreduce and decompressed after. TopK + FP16 can reduce
        communication volume by 20-200x depending on settings.

        Call after backward() and before step().

        Returns:
            Time spent in gradient sync (seconds).
        """
        if self.world_size == 1:
            return 0.0

        t0 = time.monotonic()

        # Extract gradients as flat numpy array
        flat_grads = self.engine.get_flat_gradients()
        original_bytes = flat_grads.nbytes

        # Compress if active
        try:
            if self._compressor is not None and self._compressor.active:
                compressed = self._compressor.compress(flat_grads)

                if isinstance(compressed, CompressedArray):
                    sparse_grads = self._compressor.decompress(compressed)
                    averaged = await self.group.allreduce(sparse_grads, op="mean")
                    self._bytes_sent += compressed.compressed_size
                    self._bytes_saved += original_bytes - compressed.compressed_size
                else:
                    averaged = await self.group.allreduce(compressed, op="mean")
                    self._bytes_sent += original_bytes
            else:
                # No compression
                averaged = await self.group.allreduce(flat_grads, op="mean")
                self._bytes_sent += original_bytes
        except GradientValidationError as e:
            # SECURITY: Metadata bomb or corrupt compressed gradient from peer.
            # Fall back to local gradients rather than crashing.
            logger.error("Gradient deserialization failed: %s", e)
            logger.warning("Falling back to local gradients (discarding allreduce result)")
            averaged = flat_grads

        # SECURITY: Validate gradients before applying to model.
        # Prevents gradient poisoning attacks (NaN, Inf, extreme magnitudes).
        try:
            validate_gradients(averaged)
        except GradientValidationError as e:
            logger.error("Gradient validation failed: %s", e)
            logger.warning("Falling back to local gradients (discarding allreduce result)")
            averaged = flat_grads  # use own gradients only

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
