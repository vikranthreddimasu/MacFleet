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

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from macfleet.comm.collectives import CollectiveGroup
from macfleet.compression.adaptive import (
    AdaptiveCompressionConfig,
    AdaptiveCompressor,
    CompressedArray,
    CompressionLevel,
)
from macfleet.engines.base import Engine, TrainingMetrics
from macfleet.pool.network import LinkType
from macfleet.security.audit import audit_event
from macfleet.security.auth import GradientValidationError, validate_gradients

logger = logging.getLogger(__name__)

# Per-rank TopK before dense allreduce is not mathematically equivalent to
# allreduce-then-compress: ranks keep different index sets, so the averaged
# gradient is biased. Reject these until sparse-on-wire lands (TODOS Issue 3).
_TOPK_BEFORE_ALLREDUCE_UNSAFE = frozenset({"moderate", "aggressive"})

# Until sparse-on-wire, adaptive mode may only pick dense-safe levels.
_DENSE_SAFE_LINK_LEVELS: dict[LinkType, CompressionLevel] = {
    LinkType.THUNDERBOLT: CompressionLevel.NONE,
    LinkType.LOOPBACK: CompressionLevel.NONE,
    LinkType.ETHERNET: CompressionLevel.LIGHT,
    LinkType.WIFI: CompressionLevel.LIGHT,
    LinkType.UNKNOWN: CompressionLevel.LIGHT,
}


class SyncDegradedError(RuntimeError):
    """Raised when gradient sync fails and degraded fallback is disabled."""


@dataclass
class DataParallelConfig:
    """Configuration for data parallel training."""

    # Gradient sync
    bucket_size_mb: float = 25.0  # group gradients into communication buckets
    # Staleness tolerance for async gradient sync (0 = synchronous)
    max_staleness: int = 0
    # Broadcast parameters from coordinator on start
    broadcast_params_on_start: bool = True
    # Compression (dense-wire-safe: "none", "light", "adaptive")
    compression: str = "none"
    compression_warmup_steps: int = 0
    # When False (default), sync failures abort instead of applying local-only
    # gradients (which silently diverge models across the fleet).
    allow_degraded: bool = False


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
        engine: Engine,
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
        # _bytes_saved is reserved for the future sparse-on-wire path
        # (TODOS Issue 3). v2.2 transmits dense gradients, so it stays
        # 0. Kept on the instance so the existing compression_ratio
        # property stays defined; remove together with Issue 3 wiring.
        self._bytes_saved = 0
        self._expected_grad_size: Optional[int] = None
        self._unsynced_steps = 0
        self._validation_fallback_steps = 0
        self._last_sync_error: Optional[str] = None

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
    def unsynced_steps(self) -> int:
        """Steps that used local gradients because fleet sync failed."""
        return self._unsynced_steps

    @property
    def validation_fallback_steps(self) -> int:
        """Steps that discarded a synchronized gradient due to validation."""
        return self._validation_fallback_steps

    @property
    def degraded(self) -> bool:
        """True when this rank had to fall back from synchronized training."""
        return self._unsynced_steps > 0 or self._validation_fallback_steps > 0

    @property
    def last_sync_error(self) -> Optional[str]:
        """Last sync/degradation reason, if any."""
        return self._last_sync_error

    @property
    def avg_sync_time_sec(self) -> float:
        """Average gradient sync time over all steps."""
        if self._step_count == 0:
            return 0.0
        return self._sync_time_sec / self._step_count

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio (bytes sent / bytes uncompressed).

        v2.2 transmits dense gradients on the wire (sparse-on-wire is
        TODOS Issue 3). This property therefore returns 1.0 in v2.2 —
        compression is applied locally but the wire payload size is
        unchanged. Once sparse-on-wire lands, _bytes_saved will track
        the real reduction and this ratio will go below 1.0.
        """
        total = self._bytes_sent + self._bytes_saved
        if total == 0:
            return 1.0
        return self._bytes_sent / total

    def _make_compressor(self, link_type: LinkType) -> Optional[AdaptiveCompressor]:
        """Create compressor based on config.

        Only dense-safe modes are accepted until sparse-on-wire: TopK
        sparsification before allreduce corrupts the averaged gradient when
        ranks see different data.
        """
        comp = self.config.compression
        if comp == "none":
            return None

        if comp in _TOPK_BEFORE_ALLREDUCE_UNSAFE:
            raise ValueError(
                f"compression={comp!r} applies per-rank TopK before allreduce, "
                "which biases averaged gradients when ranks train on different "
                "batches. Until sparse-on-wire ships, use compression='none', "
                "'light' (FP16), or 'adaptive' (dense-safe FP16/none only)."
            )

        if comp == "adaptive":
            # Force dense-safe levels: WiFi/Ethernet would otherwise pick TopK.
            level = _DENSE_SAFE_LINK_LEVELS.get(link_type, CompressionLevel.LIGHT)
            if level is CompressionLevel.LIGHT:
                logger.warning(
                    "compression='adaptive' uses dense-safe FP16 only until "
                    "sparse-on-wire lands; TopK levels are disabled to protect "
                    "gradient correctness (link=%s).",
                    link_type.value,
                )
            return AdaptiveCompressor(
                config=AdaptiveCompressionConfig(
                    fixed_level=level,
                    warmup_steps=self.config.compression_warmup_steps,
                ),
            )

        if comp == "light":
            return AdaptiveCompressor(
                config=AdaptiveCompressionConfig(
                    fixed_level=CompressionLevel.LIGHT,
                    warmup_steps=self.config.compression_warmup_steps,
                ),
            )

        raise ValueError(
            f"Unknown compression={comp!r}. Valid: none, light, adaptive"
        )

    def _handle_sync_degradation(
        self,
        *,
        reason: str,
        audit_name: str,
        unsynced: bool = False,
        validation: bool = False,
        **audit_fields: object,
    ) -> None:
        """Record degradation and optionally abort the training step."""
        if unsynced:
            self._unsynced_steps += 1
        if validation:
            self._validation_fallback_steps += 1
        self._last_sync_error = reason
        audit_event(audit_name, rank=self.rank, reason=reason, **audit_fields)
        if not self.config.allow_degraded:
            raise SyncDegradedError(
                f"Gradient sync degraded ({reason}). Training aborted to avoid "
                "silent model divergence. Pass allow_degraded=True on "
                "DataParallelConfig to continue with local-only gradients."
            )

    async def setup(self) -> None:
        """Initialize data parallel training.

        Verifies all nodes have the same model architecture (param count),
        then broadcasts parameters from rank 0 so everyone starts from
        identical weights.
        """
        if self.world_size > 1:
            await self._validate_model_consistency()
            if self.config.broadcast_params_on_start:
                await self.broadcast_parameters()

    async def _validate_model_consistency(self) -> None:
        """Verify all nodes have the same model architecture.

        Every rank runs the collectives (gather + allreduce) in the
        same order so a peer raising on rank 0 can't deadlock the
        ring. Rank 0 captures the per-rank breakdown for a richer
        error message; all ranks check the cheaper allreduce-sum
        afterwards. Either rank 0's gather-based error or every
        rank's sum-mismatch error fires, never neither.
        """
        local_count = len(self.engine.get_flat_parameters())
        count_array = np.array([local_count], dtype=np.float64)

        # Step 1: gather counts to rank 0 (every rank participates).
        # Bound with a short timeout so a hung peer can't block the
        # whole validation. The richer per-rank breakdown is a "nice to
        # have"; the cheaper allreduce-sum still catches the mismatch
        # if gather can't finish in time.
        try:
            gathered = await asyncio.wait_for(
                self.group.gather(count_array, dst=0), timeout=10.0,
            )
        except (asyncio.TimeoutError, Exception):
            gathered = None

        # Step 2: allreduce-sum (every rank participates BEFORE any raise).
        summed = await self.group.allreduce(count_array, op="sum")
        expected = local_count * self.world_size

        # Step 3: rank 0 raises the rich error if gather caught a mismatch.
        if self.rank == 0 and gathered is not None:
            counts = [int(gathered[r][0]) for r in range(self.world_size)]
            if len(set(counts)) > 1:
                breakdown = ", ".join(
                    f"rank{r}={c}" for r, c in enumerate(counts)
                )
                raise RuntimeError(
                    f"Model architecture mismatch across ranks: {breakdown}. "
                    "All nodes must load the same model."
                )

        # Step 4: every rank raises on sum mismatch (covers gather failure path).
        if int(summed[0]) != expected:
            raise RuntimeError(
                f"Model architecture mismatch: this node (rank {self.rank}) has "
                f"{local_count} parameters, but the fleet total is {int(summed[0])} "
                f"(expected {expected} for {self.world_size} identical nodes). "
                f"All nodes must load the same model."
            )

    async def sync_gradients(self) -> float:
        """AllReduce gradients across all nodes.

        Dense-safe compression (``light`` / dense-safe ``adaptive``) may
        quantize locally before allreduce; the wire payload remains dense
        until sparse-on-wire. TopK levels are rejected because per-rank
        sparsification before allreduce biases the averaged gradient.

        On sync failure, raises :class:`SyncDegradedError` unless
        ``config.allow_degraded`` is True (local-only fallback).

        Call after backward() and before step().

        Returns:
            Time spent in gradient sync (seconds).
        """
        if self.world_size == 1:
            return 0.0

        t0 = time.monotonic()

        # Extract gradients as flat numpy array
        flat_grads = self.engine.get_flat_gradients()

        # Guard: empty gradients (no trainable params or all grads are None)
        if flat_grads.size == 0:
            logger.warning("Empty gradient array — skipping sync (no trainable params?)")
            self._step_count += 1
            return 0.0

        # Guard: NaN/Inf in local gradients before sending to peers.
        # A single node's NaN loss contaminates the entire fleet via allreduce.
        if not np.isfinite(flat_grads).all():
            logger.error(
                "Local gradients contain NaN/Inf (likely from NaN loss). "
                "Zeroing gradients for this step to avoid poisoning the fleet."
            )
            self._validation_fallback_steps += 1
            self._last_sync_error = "local_gradients_nan_or_inf"
            audit_event(
                "training.gradient_validation_failed",
                rank=self.rank,
                reason=self._last_sync_error,
            )
            flat_grads = np.zeros_like(flat_grads)

        # Record expected size on first call for shape validation
        if self._expected_grad_size is None:
            self._expected_grad_size = flat_grads.size

        original_bytes = flat_grads.nbytes

        # Compress if active. NOTE: until sparse-on-wire (TODOS Issue 3),
        # compressed gradients are decompressed locally before allreduce,
        # so the wire payload is dense regardless. _bytes_sent therefore
        # reflects ACTUAL wire bytes. _bytes_saved stays 0 in v2.2.
        try:
            if self._compressor is not None:
                compressed = self._compressor.compress(flat_grads)

                if isinstance(compressed, CompressedArray):
                    dense_grads = self._compressor.decompress(compressed)
                    averaged = await self.group.allreduce(dense_grads, op="mean")
                else:
                    averaged = await self.group.allreduce(compressed, op="mean")
                self._bytes_sent += original_bytes
            else:
                # No compression
                averaged = await self.group.allreduce(flat_grads, op="mean")
                self._bytes_sent += original_bytes
        except GradientValidationError as e:
            # SECURITY: Metadata bomb or corrupt compressed gradient from peer.
            logger.error("Gradient deserialization failed: %s", e)
            self._handle_sync_degradation(
                reason=f"gradient_deserialization_failed:{e}",
                audit_name="training.sync_degraded",
                unsynced=True,
            )
            logger.warning("Falling back to local gradients (discarding allreduce result)")
            averaged = flat_grads
        except (
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            ConnectionError,
            OSError,
            EOFError,
        ) as e:
            # Node dropout: a peer disconnected mid-allreduce (lid closed,
            # thermal shutdown, network failure).
            logger.error("Allreduce failed (node dropout?): %s", e)
            self._handle_sync_degradation(
                reason=f"allreduce_failed:{type(e).__name__}",
                audit_name="training.sync_degraded",
                unsynced=True,
                error_type=type(e).__name__,
            )
            logger.warning(
                "Falling back to local gradients. Training continues but "
                "this step is not synchronized across the fleet."
            )
            averaged = flat_grads

        # SECURITY: Validate gradients before applying to model.
        # Prevents gradient poisoning attacks (NaN, Inf, extreme magnitudes).
        try:
            validate_gradients(averaged)
        except GradientValidationError as e:
            logger.error("Gradient validation failed: %s", e)
            self._handle_sync_degradation(
                reason=f"gradient_validation_failed:{e}",
                audit_name="training.gradient_validation_failed",
                validation=True,
            )
            logger.warning("Falling back to local gradients (discarding allreduce result)")
            averaged = flat_grads  # use own gradients only

        # Guard: verify allreduce didn't corrupt the shape
        if averaged.size != self._expected_grad_size:
            logger.error(
                "Gradient shape mismatch after allreduce: expected %d, got %d. "
                "Falling back to local gradients.",
                self._expected_grad_size,
                averaged.size,
            )
            self._handle_sync_degradation(
                reason="gradient_shape_mismatch",
                audit_name="training.gradient_validation_failed",
                validation=True,
            )
            averaged = flat_grads

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
