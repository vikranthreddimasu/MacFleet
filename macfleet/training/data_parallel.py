"""Distributed Data Parallel wrapper for MacFleet.

Implements a DDP-style wrapper that synchronizes gradients
across nodes using AllReduce after the backward pass.
"""

import asyncio
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from macfleet.comm.collectives import AllReduce, CollectiveGroup
from macfleet.compression.pipeline import CompressionPipeline, create_pipeline


class MacFleetDDP(nn.Module):
    """Distributed Data Parallel wrapper for MacFleet.

    Wraps a PyTorch model and synchronizes gradients across nodes
    after each backward pass using AllReduce.

    Example:
        model = resnet50()
        ddp_model = MacFleetDDP(model, collective_group)

        # Training loop
        for batch in dataloader:
            loss = ddp_model(batch)
            loss.backward()
            ddp_model.sync_gradients()  # AllReduce gradients
            optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        collective_group: CollectiveGroup,
        compression_pipeline: Optional[CompressionPipeline] = None,
        broadcast_buffers: bool = True,
        bucket_size_mb: float = 25.0,
    ):
        """Initialize DDP wrapper.

        Args:
            module: PyTorch model to wrap.
            collective_group: Collective group for communication.
            compression_pipeline: Optional compression for gradients.
            broadcast_buffers: Whether to sync buffers (BatchNorm stats).
            bucket_size_mb: Size of gradient buckets for batching.
        """
        super().__init__()
        self.module = module
        self._group = collective_group
        self._compression = compression_pipeline
        self._broadcast_buffers = broadcast_buffers
        self._bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        # AllReduce operation
        if self._compression:
            self._allreduce = AllReduce(
                collective_group,
                compress_fn=self._compress_gradient,
                decompress_fn=self._decompress_gradient,
            )
        else:
            self._allreduce = AllReduce(collective_group)

        # Track parameters for gradient sync
        self._param_list = list(self.module.parameters())
        self._grad_ready = [False] * len(self._param_list)

        # For async gradient sync (optional optimization)
        self._pending_syncs: list[asyncio.Task] = []

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)

    def _compress_gradient(self, tensor: torch.Tensor) -> tuple:
        """Compress a gradient tensor."""
        compressed = self._compression.compress(tensor)
        return (
            compressed.indices,
            compressed.values,
            compressed.original_numel,
            compressed.original_dtype,
        )

    def _decompress_gradient(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        numel: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decompress a gradient tensor."""
        from macfleet.compression.pipeline import CompressedGradient

        compressed = CompressedGradient(
            indices=indices,
            values=values,
            original_numel=numel,
            original_dtype=dtype,
            is_sparse=True,
        )
        return self._compression.decompress(compressed)

    def _build_buckets(self) -> list[list[nn.Parameter]]:
        """Group parameters into communication buckets.

        Parameters are grouped in reverse order (matching PyTorch DDP convention,
        since later layers finish backward first) until bucket size is reached.
        """
        buckets: list[list[nn.Parameter]] = []
        current_bucket: list[nn.Parameter] = []
        current_size = 0

        for param in reversed(self._param_list):
            if param.grad is None:
                continue
            param_bytes = param.grad.numel() * param.grad.element_size()
            if current_bucket and current_size + param_bytes > self._bucket_size_bytes:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append(param)
            current_size += param_bytes

        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    async def sync_gradients(self) -> None:
        """Synchronize gradients across all nodes using AllReduce.

        Call this after loss.backward() and before optimizer.step().
        Gradients are grouped into buckets for overlapped communication.
        """
        if self._group.world_size == 1:
            return

        buckets = self._build_buckets()
        if not buckets:
            return

        for bucket in buckets:
            # Flatten bucket gradients
            flat = torch.cat([p.grad.data.flatten() for p in bucket])

            # AllReduce this bucket
            reduced = await self._allreduce(flat, op="mean")
            del flat

            # Copy back
            offset = 0
            for param in bucket:
                numel = param.grad.numel()
                param.grad.data.copy_(
                    reduced[offset:offset + numel].view_as(param.grad)
                )
                offset += numel
            del reduced

    def sync_gradients_sync(self) -> None:
        """Synchronous wrapper for sync_gradients.

        Only usable from a non-async context. For use within the async
        trainer, call sync_gradients() directly with await.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "sync_gradients_sync() cannot be called from a running event loop. "
                "Use 'await sync_gradients()' instead."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e).lower():
                raise
        asyncio.run(self.sync_gradients())

    async def sync_buffers(self) -> None:
        """Synchronize buffers (e.g., BatchNorm running stats) from rank 0.

        Called at the start of each forward pass if broadcast_buffers=True.
        """
        if not self._broadcast_buffers or self._group.world_size == 1:
            return

        from macfleet.comm.collectives import Broadcast

        broadcast = Broadcast(self._group)

        for buffer in self.module.buffers():
            synced = await broadcast(buffer, src=0)
            buffer.data.copy_(synced)

    async def broadcast_parameters(self) -> None:
        """Broadcast model parameters from rank 0 to all nodes.

        Call this at initialization to ensure all nodes start with
        the same model weights.
        """
        from macfleet.comm.collectives import Broadcast

        broadcast = Broadcast(self._group)

        for param in self.module.parameters():
            synced = await broadcast(param.data, src=0)
            param.data.copy_(synced)

    def get_gradient_norm(self) -> float:
        """Compute the total gradient norm (for debugging/logging)."""
        total_norm = 0.0
        for param in self._param_list:
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients on the wrapped module."""
        self.module.zero_grad(set_to_none=set_to_none)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return parameters of the wrapped module."""
        return self.module.parameters(recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """Return named parameters of the wrapped module."""
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return state dict of the wrapped module."""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load state dict into the wrapped module."""
        return self.module.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True) -> "MacFleetDDP":
        """Set training mode."""
        self.module.train(mode)
        return self

    def eval(self) -> "MacFleetDDP":
        """Set evaluation mode."""
        self.module.eval()
        return self


class GradientAccumulator:
    """Accumulate gradients over multiple micro-batches before sync.

    Useful for simulating larger batch sizes when memory is limited.
    """

    def __init__(
        self,
        ddp_model: MacFleetDDP,
        accumulation_steps: int = 1,
    ):
        """Initialize accumulator.

        Args:
            ddp_model: DDP-wrapped model.
            accumulation_steps: Number of micro-batches before sync.
        """
        self.model = ddp_model
        self.accumulation_steps = accumulation_steps
        self._step_count = 0

    async def step(self, loss: torch.Tensor) -> bool:
        """Process one micro-batch.

        Args:
            loss: Loss from this micro-batch.

        Returns:
            True if gradients were synced (accumulation complete).
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self._step_count += 1

        if self._step_count >= self.accumulation_steps:
            await self.model.sync_gradients()
            self._step_count = 0
            return True

        return False

    def reset(self) -> None:
        """Reset step counter."""
        self._step_count = 0


def wrap_model(
    model: nn.Module,
    collective_group: CollectiveGroup,
    compression: str = "topk_fp16",
    topk_ratio: float = 0.1,
) -> MacFleetDDP:
    """Convenience function to wrap a model for distributed training.

    Args:
        model: PyTorch model.
        collective_group: Collective group for this node.
        compression: Compression type ("none", "topk", "fp16", "topk_fp16").
        topk_ratio: Top-K ratio if using compression.

    Returns:
        DDP-wrapped model.
    """
    pipeline = create_pipeline(compression, topk_ratio) if compression != "none" else None

    return MacFleetDDP(
        module=model,
        collective_group=collective_group,
        compression_pipeline=pipeline,
    )
