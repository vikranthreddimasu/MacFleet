"""Constraint-based heterogeneous workload scheduler.

Assigns batch proportions based on measured throughput, GPU cores,
and thermal state. Continuously adapts during training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from macfleet.pool.registry import ClusterRegistry


def _positive_finite_capacity(value: object) -> float:
    """Return a usable scheduling capacity, or zero for invalid telemetry."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        capacity = float(value)
    except (OverflowError, ValueError):
        return 0.0
    return capacity if math.isfinite(capacity) and capacity > 0 else 0.0


@dataclass
class WorkloadAssignment:
    """Workload assignment for a single node."""
    node_id: str
    rank: int
    weight: float           # fraction of total batch (0.0 - 1.0)
    batch_size: int         # actual samples for this node
    is_viable: bool = True  # False if batch too small to be useful

    @property
    def pct(self) -> str:
        return f"{self.weight * 100:.1f}%"


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    min_batch_per_node: int = 4       # Minimum viable batch size
    use_throughput: bool = True       # Use measured throughput if available
    rebalance_every_n_steps: int = 50 # Re-profile interval

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_batch_per_node, int)
            or isinstance(self.min_batch_per_node, bool)
            or self.min_batch_per_node < 1
        ):
            raise ValueError("min_batch_per_node must be a positive integer")
        if not isinstance(self.use_throughput, bool):
            raise ValueError("use_throughput must be a boolean")
        if (
            not isinstance(self.rebalance_every_n_steps, int)
            or isinstance(self.rebalance_every_n_steps, bool)
            or self.rebalance_every_n_steps < 1
        ):
            raise ValueError("rebalance_every_n_steps must be a positive integer")


class Scheduler:
    """Assigns workload across heterogeneous pool members.

    Uses a two-phase approach:
    1. Initial: weight by GPU cores
    2. Running: weight by measured throughput * thermal_factor

    The scheduler produces WorkloadAssignment objects that the
    DataParallel strategy uses to split batches.
    """

    def __init__(
        self,
        registry: ClusterRegistry,
        config: Optional[SchedulerConfig] = None,
    ):
        self.registry = registry
        self.config = config or SchedulerConfig()
        self._step_count = 0

    def compute_weights(self) -> dict[str, float]:
        """Compute normalized weights for all alive nodes.

        Returns:
            Dict mapping node_id to weight (0.0 - 1.0, sums to 1.0).
        """
        nodes = self.registry.alive_nodes
        if not nodes:
            return {}

        raw_weights: dict[str, float] = {}
        for node in nodes:
            throughput = _positive_finite_capacity(node.throughput_samples_sec)
            gpu_capacity = _positive_finite_capacity(node.hardware.gpu_cores)
            base = throughput if self.config.use_throughput and throughput else gpu_capacity

            factor = node.hardware.thermal_pressure.workload_multiplier
            raw_weights[node.node_id] = base * factor

        max_weight = max(raw_weights.values())
        if max_weight <= 0:
            # Equal split
            n = len(nodes)
            return {nid: 1.0 / n for nid in raw_weights}

        # Normalize after scaling by the largest capacity. Summing raw finite
        # values can still overflow (for example, two 1e308 throughput samples).
        scaled_weights = {nid: weight / max_weight for nid, weight in raw_weights.items()}
        total = sum(scaled_weights.values())
        return {nid: weight / total for nid, weight in scaled_weights.items()}

    def assign(self, global_batch_size: int) -> list[WorkloadAssignment]:
        """Produce workload assignments for all alive nodes.

        Args:
            global_batch_size: Total batch size to split across nodes.

        Returns:
            List of WorkloadAssignment, one per alive node.
        """
        if (
            not isinstance(global_batch_size, int)
            or isinstance(global_batch_size, bool)
            or global_batch_size < 1
        ):
            raise ValueError("global_batch_size must be a positive integer")
        weights = self.compute_weights()
        ranks = self.registry.get_ranks()

        assignments: list[WorkloadAssignment] = []
        sorted_nodes = sorted(weights.keys(), key=lambda nid: ranks.get(nid, 999))

        ideal_batches = {
            node_id: global_batch_size * weights[node_id]
            for node_id in sorted_nodes
        }
        batches = {
            node_id: int(ideal_batches[node_id])
            for node_id in sorted_nodes
        }
        remaining = global_batch_size - sum(batches.values())
        remainder_order = sorted(
            sorted_nodes,
            key=lambda node_id: (
                -(ideal_batches[node_id] - batches[node_id]),
                ranks.get(node_id, 999),
                node_id,
            ),
        )
        for node_id in remainder_order[:remaining]:
            batches[node_id] += 1

        for i, node_id in enumerate(sorted_nodes):
            weight = weights[node_id]
            rank = ranks.get(node_id, i)
            batch = batches[node_id]

            is_viable = batch >= self.config.min_batch_per_node

            assignments.append(WorkloadAssignment(
                node_id=node_id,
                rank=rank,
                weight=weight,
                batch_size=batch,
                is_viable=is_viable,
            ))

        return assignments

    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance weights."""
        self._step_count += 1
        return self._step_count % self.config.rebalance_every_n_steps == 0

    def get_non_viable_nodes(self, global_batch_size: int) -> list[str]:
        """Get node IDs that would have too-small batches."""
        assignments = self.assign(global_batch_size)
        return [a.node_id for a in assignments if not a.is_viable]
