"""Throughput calibration and monitoring for MacFleet.

Measures each node's training throughput to compute optimal
workload weights for data parallel training.
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from rich.console import Console

console = Console()


@dataclass
class ThroughputResult:
    """Result of throughput calibration."""
    samples_per_sec: float
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_used_mb: float
    device: str


def calibrate_throughput(
    model: nn.Module,
    batch_size: int,
    input_shape: tuple,
    device: str = "mps",
    num_warmup: int = 3,
    num_iterations: int = 10,
    dtype: torch.dtype = torch.float32,
) -> ThroughputResult:
    """Calibrate throughput by running forward/backward passes.

    Runs the model on dummy data to measure how fast this node
    can process training batches.

    Args:
        model: PyTorch model to benchmark.
        batch_size: Batch size to use.
        input_shape: Shape of input tensor (without batch dimension).
        device: Device to run on ("mps", "cuda", "cpu").
        num_warmup: Number of warmup iterations (not timed).
        num_iterations: Number of timed iterations.
        dtype: Data type for inputs.

    Returns:
        ThroughputResult with timing measurements.
    """
    console.print(f"[blue]Calibrating throughput on {device}...[/blue]")

    # Move model to device
    model = model.to(device)
    model.train()

    # Create dummy data
    full_shape = (batch_size,) + input_shape
    dummy_input = torch.randn(full_shape, dtype=dtype, device=device)
    dummy_target = torch.randint(0, 1000, (batch_size,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warmup
    console.print(f"  Warmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        optimizer.zero_grad()
        output = model(dummy_input)
        if output.shape[-1] != 1000:
            # Handle models with different output sizes
            dummy_target = torch.randint(0, output.shape[-1], (batch_size,), device=device)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()

        if device == "mps":
            torch.mps.synchronize()

    # Timed iterations
    console.print(f"  Timing ({num_iterations} iterations)...")
    forward_times = []
    backward_times = []
    total_times = []

    for i in range(num_iterations):
        # Fresh input each time to avoid caching effects
        dummy_input = torch.randn(full_shape, dtype=dtype, device=device)

        # Time forward pass
        if device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)

        if device == "mps":
            torch.mps.synchronize()
        forward_time = time.perf_counter() - start

        # Time backward pass
        start = time.perf_counter()
        loss.backward()
        optimizer.step()

        if device == "mps":
            torch.mps.synchronize()
        backward_time = time.perf_counter() - start

        forward_times.append(forward_time * 1000)  # Convert to ms
        backward_times.append(backward_time * 1000)
        total_times.append((forward_time + backward_time) * 1000)

    # Compute averages
    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)
    avg_total = sum(total_times) / len(total_times)

    # Compute samples per second
    samples_per_sec = (batch_size * 1000) / avg_total

    # Get memory usage
    if device == "mps":
        # MPS doesn't have a direct memory query, estimate from tensor sizes
        memory_used_mb = estimate_model_memory(model) + (
            dummy_input.numel() * dummy_input.element_size() / (1024 * 1024)
        )
    elif device == "cuda":
        memory_used_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        memory_used_mb = 0

    result = ThroughputResult(
        samples_per_sec=samples_per_sec,
        forward_time_ms=avg_forward,
        backward_time_ms=avg_backward,
        total_time_ms=avg_total,
        memory_used_mb=memory_used_mb,
        device=device,
    )

    console.print(f"  [green]Throughput: {samples_per_sec:.1f} samples/sec[/green]")
    console.print(f"  Forward: {avg_forward:.1f}ms, Backward: {avg_backward:.1f}ms")

    return result


def estimate_model_memory(model: nn.Module) -> float:
    """Estimate model memory usage in MB.

    Args:
        model: PyTorch model.

    Returns:
        Estimated memory in MB.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    # Gradients are same size as parameters
    grad_size = param_size

    # Optimizer states (Adam has 2x, SGD has ~0.5x)
    optimizer_size = param_size * 2  # Conservative estimate

    total = param_size + buffer_size + grad_size + optimizer_size
    return total / (1024 * 1024)


def compute_workload_weights(throughputs: list[float]) -> list[float]:
    """Compute workload weights from throughput measurements.

    Args:
        throughputs: List of samples/sec per node.

    Returns:
        Normalized weights for each node.
    """
    total = sum(throughputs)
    if total == 0:
        return [1.0 / len(throughputs)] * len(throughputs)
    return [t / total for t in throughputs]


class ThroughputMonitor:
    """Monitor training throughput in real-time."""

    def __init__(self, window_size: int = 100):
        """Initialize monitor.

        Args:
            window_size: Number of samples for moving average.
        """
        self.window_size = window_size
        self._times: list[float] = []
        self._batch_sizes: list[int] = []
        self._last_time: Optional[float] = None

    def start_batch(self) -> None:
        """Mark start of a batch."""
        self._last_time = time.perf_counter()

    def end_batch(self, batch_size: int) -> float:
        """Mark end of a batch and return current throughput.

        Args:
            batch_size: Number of samples in this batch.

        Returns:
            Current throughput in samples/sec.
        """
        if self._last_time is None:
            return 0.0

        elapsed = time.perf_counter() - self._last_time
        self._times.append(elapsed)
        self._batch_sizes.append(batch_size)

        # Keep window size
        if len(self._times) > self.window_size:
            self._times.pop(0)
            self._batch_sizes.pop(0)

        # Compute throughput
        total_time = sum(self._times)
        total_samples = sum(self._batch_sizes)

        if total_time > 0:
            return total_samples / total_time
        return 0.0

    def get_throughput(self) -> float:
        """Get current moving average throughput."""
        total_time = sum(self._times)
        total_samples = sum(self._batch_sizes)
        if total_time > 0:
            return total_samples / total_time
        return 0.0

    def reset(self) -> None:
        """Reset the monitor."""
        self._times.clear()
        self._batch_sizes.clear()
        self._last_time = None
