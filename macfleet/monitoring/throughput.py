"""Training throughput tracking for MacFleet.

Tracks samples/sec, step times, communication overhead, and efficiency
across training runs. Used by the scheduler for re-profiling and
dynamic weight adjustment.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    samples: int
    compute_time_sec: float     # forward + backward
    sync_time_sec: float        # gradient allreduce
    step_time_sec: float        # total step time
    loss: float = 0.0
    timestamp: float = 0.0

    @property
    def throughput(self) -> float:
        """Samples per second for this step."""
        if self.step_time_sec <= 0:
            return 0.0
        return self.samples / self.step_time_sec

    @property
    def compute_pct(self) -> float:
        """Fraction of step time spent on compute."""
        if self.step_time_sec <= 0:
            return 0.0
        return self.compute_time_sec / self.step_time_sec

    @property
    def comm_pct(self) -> float:
        """Fraction of step time spent on communication."""
        if self.step_time_sec <= 0:
            return 0.0
        return self.sync_time_sec / self.step_time_sec


class ThroughputTracker:
    """Tracks training throughput with windowed statistics.

    Records per-step metrics and computes rolling averages for
    the scheduler's re-profiling decisions.

    Usage:
        tracker = ThroughputTracker()
        for step in range(N):
            with tracker.step(batch_size) as s:
                ... forward + backward ...
                s.compute_done()
                ... sync gradients ...
                s.sync_done()
                s.record_loss(loss_val)
    """

    def __init__(self, window_size: int = 50):
        self._window_size = window_size
        self._history: deque[StepMetrics] = deque(maxlen=window_size)
        self._total_steps = 0
        self._total_samples = 0
        self._start_time: Optional[float] = None

    def step(self, samples: int) -> StepContext:
        """Create a context manager for tracking a training step.

        Args:
            samples: Number of samples in this step's batch.

        Returns:
            StepContext to track timing within the step.
        """
        if self._start_time is None:
            self._start_time = time.monotonic()
        return StepContext(self, samples)

    def record(self, metrics: StepMetrics) -> None:
        """Record a completed step's metrics."""
        self._history.append(metrics)
        self._total_steps += 1
        self._total_samples += metrics.samples

    @property
    def avg_throughput(self) -> float:
        """Rolling average throughput (samples/sec)."""
        if not self._history:
            return 0.0
        total_samples = sum(m.samples for m in self._history)
        total_time = sum(m.step_time_sec for m in self._history)
        if total_time <= 0:
            return 0.0
        return total_samples / total_time

    @property
    def avg_step_time(self) -> float:
        """Rolling average step time (sec)."""
        if not self._history:
            return 0.0
        return sum(m.step_time_sec for m in self._history) / len(self._history)

    @property
    def avg_compute_time(self) -> float:
        """Rolling average compute time (sec)."""
        if not self._history:
            return 0.0
        return sum(m.compute_time_sec for m in self._history) / len(self._history)

    @property
    def avg_sync_time(self) -> float:
        """Rolling average sync time (sec)."""
        if not self._history:
            return 0.0
        return sum(m.sync_time_sec for m in self._history) / len(self._history)

    @property
    def comm_compute_ratio(self) -> float:
        """Communication-to-computation ratio.

        > 1.0 means network is the bottleneck.
        """
        compute = self.avg_compute_time
        if compute <= 0:
            return 0.0
        return self.avg_sync_time / compute

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def overall_throughput(self) -> float:
        """Throughput since start (samples/sec)."""
        if self._start_time is None or self._total_samples == 0:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._total_samples / elapsed

    @property
    def loss_history(self) -> list[float]:
        """Recent loss values."""
        return [m.loss for m in self._history if m.loss > 0]

    @property
    def throughput_history(self) -> list[float]:
        """Recent throughput values."""
        return [m.throughput for m in self._history]

    def summary(self) -> dict:
        """Summary statistics for display."""
        return {
            "total_steps": self._total_steps,
            "total_samples": self._total_samples,
            "avg_throughput": round(self.avg_throughput, 1),
            "avg_step_time_ms": round(self.avg_step_time * 1000, 1),
            "avg_compute_ms": round(self.avg_compute_time * 1000, 1),
            "avg_sync_ms": round(self.avg_sync_time * 1000, 1),
            "comm_compute_ratio": round(self.comm_compute_ratio, 2),
        }


class StepContext:
    """Context manager for tracking a single training step."""

    def __init__(self, tracker: ThroughputTracker, samples: int):
        self._tracker = tracker
        self._samples = samples
        self._start = 0.0
        self._compute_end = 0.0
        self._sync_end = 0.0
        self._loss = 0.0

    def __enter__(self) -> StepContext:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        end = time.monotonic()
        step_time = end - self._start
        compute_time = (self._compute_end - self._start) if self._compute_end > 0 else step_time
        sync_time = (self._sync_end - self._compute_end) if self._sync_end > 0 and self._compute_end > 0 else 0.0

        metrics = StepMetrics(
            step=self._tracker._total_steps,
            samples=self._samples,
            compute_time_sec=compute_time,
            sync_time_sec=sync_time,
            step_time_sec=step_time,
            loss=self._loss,
            timestamp=end,
        )
        self._tracker.record(metrics)

    def compute_done(self) -> None:
        """Mark end of compute phase (forward + backward)."""
        self._compute_end = time.monotonic()

    def sync_done(self) -> None:
        """Mark end of sync phase (gradient allreduce)."""
        self._sync_end = time.monotonic()

    def record_loss(self, loss: float) -> None:
        """Record the loss value for this step."""
        self._loss = loss
