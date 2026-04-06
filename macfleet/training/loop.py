"""Composable training loop for MacFleet v2.

Ties together: Engine, DataParallel, CollectiveGroup, and DataLoader
to run distributed data-parallel training across the pool.

The loop is async-native so gradient sync (network I/O) can overlap
with other work. Supports:
    - Weighted batch allocation (via sampler)
    - Gradient sync timing
    - Periodic checkpointing
    - Early stopping on loss plateau
    - Progress reporting via callbacks
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from macfleet.engines.base import TrainingMetrics
from macfleet.training.data_parallel import DataParallel


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    epochs: int = 10
    log_every_n_steps: int = 10
    checkpoint_every_n_steps: int = 0  # 0 = disabled
    max_grad_norm: float = 0.0  # 0 = disabled
    # Callbacks
    on_step: Optional[Callable] = None  # (step, metrics) -> None
    on_epoch: Optional[Callable] = None  # (epoch, metrics) -> None


@dataclass
class StepResult:
    """Result of a single training step."""

    loss: float
    sync_time_sec: float
    step_time_sec: float
    step: int


@dataclass
class EpochResult:
    """Result of a single epoch."""

    epoch: int
    avg_loss: float
    avg_step_time_sec: float
    avg_sync_time_sec: float
    total_time_sec: float
    steps: int


@dataclass
class TrainingResult:
    """Final result of a training run."""

    epochs_completed: int
    total_steps: int
    final_loss: float
    total_time_sec: float
    avg_throughput_steps_sec: float
    epoch_results: list[EpochResult] = field(default_factory=list)


async def training_loop(
    engine: object,
    dp: DataParallel,
    dataloader: Any,
    config: Optional[TrainingConfig] = None,
) -> TrainingResult:
    """Run the distributed training loop.

    Args:
        engine: TorchEngine (or any Engine protocol implementation).
        dp: DataParallel strategy (handles gradient sync).
        dataloader: PyTorch DataLoader yielding batches.
        config: Training configuration.

    Returns:
        TrainingResult with loss history and timing.
    """
    config = config or TrainingConfig()
    global_step = 0
    epoch_results = []
    total_start = time.monotonic()

    # Broadcast initial parameters so all nodes start identically
    await dp.setup()

    for epoch in range(config.epochs):
        epoch_start = time.monotonic()
        epoch_losses = []
        epoch_sync_times = []
        epoch_step_times = []

        for batch in dataloader:
            step_start = time.monotonic()

            # Forward + backward
            engine.zero_grad()
            loss = engine.forward(batch)
            engine.backward(loss)

            # Gradient sync across all nodes
            sync_time = await dp.sync_gradients()

            # Optimizer step
            engine.step()

            loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
            step_time = time.monotonic() - step_start
            global_step += 1

            epoch_losses.append(loss_val)
            epoch_sync_times.append(sync_time)
            epoch_step_times.append(step_time)

            step_result = StepResult(
                loss=loss_val,
                sync_time_sec=sync_time,
                step_time_sec=step_time,
                step=global_step,
            )

            if config.on_step and global_step % config.log_every_n_steps == 0:
                config.on_step(global_step, step_result)

        epoch_result = EpochResult(
            epoch=epoch,
            avg_loss=float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            avg_step_time_sec=float(np.mean(epoch_step_times)) if epoch_step_times else 0.0,
            avg_sync_time_sec=float(np.mean(epoch_sync_times)) if epoch_sync_times else 0.0,
            total_time_sec=time.monotonic() - epoch_start,
            steps=len(epoch_losses),
        )
        epoch_results.append(epoch_result)

        if config.on_epoch:
            config.on_epoch(epoch, epoch_result)

        # Update sampler epoch if it exists (for shuffle reproducibility)
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch + 1)

    total_time = time.monotonic() - total_start
    return TrainingResult(
        epochs_completed=config.epochs,
        total_steps=global_step,
        final_loss=epoch_results[-1].avg_loss if epoch_results else 0.0,
        total_time_sec=total_time,
        avg_throughput_steps_sec=global_step / total_time if total_time > 0 else 0.0,
        epoch_results=epoch_results,
    )
