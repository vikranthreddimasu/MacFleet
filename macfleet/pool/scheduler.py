"""Constraint-based heterogeneous workload scheduler.

Assigns batch proportions based on measured throughput, GPU cores,
and thermal state. Continuously adapts during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from macfleet.engines.base import ThermalPressure
from macfleet.pool.registry import ClusterRegistry, NodeRecord


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
    thermal_factors: dict[ThermalPressure, float] | None = None
    use_throughput: bool = True       # Use measured throughput if available
    rebalance_every_n_steps: int = 50 # Re-profile interval

    @property
    def _thermal_factors(self) -> dict[ThermalPressure, float]:
        if self.thermal_factors:
            return self.thermal_factors
        return {
            ThermalPressure.NOMINAL: 1.0,
            ThermalPressure.FAIR: 0.9,
            ThermalPressure.SERIOUS: 0.7,
            ThermalPressure.CRITICAL: 0.3,
        }


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

        thermal_factors = self.config._thermal_factors

        raw_weights: dict[str, float] = {}
        for node in nodes:
            if self.config.use_throughput and node.throughput_samples_sec > 0:
                base = node.throughput_samples_sec
            else:
                base = float(node.hardware.gpu_cores)

            factor = thermal_factors.get(node.hardware.thermal_pressure, 1.0)
            raw_weights[node.node_id] = base * factor

        total = sum(raw_weights.values())
        if total <= 0:
            # Equal split
            n = len(nodes)
            return {nid: 1.0 / n for nid in raw_weights}

        return {nid: w / total for nid, w in raw_weights.items()}

    def assign(self, global_batch_size: int) -> list[WorkloadAssignment]:
        """Produce workload assignments for all alive nodes.

        Args:
            global_batch_size: Total batch size to split across nodes.

        Returns:
            List of WorkloadAssignment, one per alive node.
        """
        weights = self.compute_weights()
        ranks = self.registry.get_ranks()

        assignments: list[WorkloadAssignment] = []
        remaining = global_batch_size

        sorted_nodes = sorted(weights.keys(), key=lambda nid: ranks.get(nid, 999))

        for i, node_id in enumerate(sorted_nodes):
            weight = weights[node_id]
            rank = ranks.get(node_id, i)

            if i == len(sorted_nodes) - 1:
                # Last node gets remainder
                batch = remaining
            else:
                batch = max(1, int(global_batch_size * weight))
                remaining -= batch

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
