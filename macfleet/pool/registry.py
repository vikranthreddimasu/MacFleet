"""Cluster registry: tracks pool members and elects coordinator.

No single master — the registry is maintained locally on each node,
synchronized via gossip. Coordinator is elected via bully algorithm
(highest compute_score wins).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from macfleet.engines.base import HardwareProfile, ThermalPressure
from macfleet.pool.heartbeat import NodeStatus


@dataclass
class NodeRecord:
    """Full record for a pool member.

    `port` is the heartbeat/discovery port (default 50051).
    `data_port` is the training transport port (default port + 1, i.e. 50052).
    Port split landed in v2.2 PR 2 (Issue 5).
    """
    node_id: str
    hostname: str
    ip_address: str
    port: int
    hardware: HardwareProfile
    data_port: int = 0  # 0 = derive as port + 1 (backward compat)
    status: NodeStatus = NodeStatus.ALIVE
    joined_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    throughput_samples_sec: float = 0.0
    current_weight: float = 0.0  # assigned by scheduler

    def __post_init__(self) -> None:
        if self.data_port == 0:
            self.data_port = self.port + 1

    @property
    def compute_score(self) -> float:
        return self.hardware.compute_score

    @property
    def is_alive(self) -> bool:
        return self.status == NodeStatus.ALIVE

    @property
    def is_coordinator_eligible(self) -> bool:
        return self.status in (NodeStatus.ALIVE,)


class ClusterRegistry:
    """Local view of the compute pool.

    Each node maintains its own registry, updated via heartbeat gossip.
    Coordinator election uses the bully algorithm: highest compute_score wins.
    """

    def __init__(self, local_node_id: str):
        self._local_node_id = local_node_id
        self._nodes: dict[str, NodeRecord] = {}
        self._lock = threading.Lock()
        self._coordinator_id: Optional[str] = None

    @property
    def local_node_id(self) -> str:
        return self._local_node_id

    @property
    def coordinator_id(self) -> Optional[str]:
        return self._coordinator_id

    @property
    def is_coordinator(self) -> bool:
        return self._coordinator_id == self._local_node_id

    @property
    def world_size(self) -> int:
        """Number of alive nodes."""
        with self._lock:
            return sum(1 for n in self._nodes.values() if n.is_alive)

    @property
    def alive_nodes(self) -> list[NodeRecord]:
        with self._lock:
            return [n for n in self._nodes.values() if n.is_alive]

    @property
    def all_nodes(self) -> list[NodeRecord]:
        with self._lock:
            return list(self._nodes.values())

    def register(self, record: NodeRecord) -> None:
        """Register or update a node in the registry."""
        with self._lock:
            self._nodes[record.node_id] = record
        self._elect_coordinator()

    def deregister(self, node_id: str) -> None:
        """Remove a node from the registry."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].status = NodeStatus.LEFT
        self._elect_coordinator()

    def mark_failed(self, node_id: str) -> None:
        """Mark a node as failed."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].status = NodeStatus.FAILED
        self._elect_coordinator()

    def mark_alive(self, node_id: str) -> None:
        """Mark a node as alive (recovered)."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].status = NodeStatus.ALIVE
                self._nodes[node_id].last_heartbeat = time.time()
        self._elect_coordinator()

    def update_throughput(self, node_id: str, throughput: float) -> None:
        """Update a node's measured throughput."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].throughput_samples_sec = throughput

    def update_thermal(self, node_id: str, pressure: ThermalPressure) -> None:
        """Update a node's thermal pressure."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].hardware.thermal_pressure = pressure

    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        with self._lock:
            return self._nodes.get(node_id)

    def get_ranks(self) -> dict[str, int]:
        """Assign ranks to alive nodes (sorted by compute_score desc, stable)."""
        alive = self.alive_nodes
        alive.sort(key=lambda n: (-n.compute_score, n.node_id))
        return {node.node_id: rank for rank, node in enumerate(alive)}

    def _elect_coordinator(self) -> None:
        """Bully algorithm: highest compute_score becomes coordinator."""
        with self._lock:
            eligible = [
                n for n in self._nodes.values() if n.is_coordinator_eligible
            ]
            if not eligible:
                self._coordinator_id = None
                return

            # Highest compute_score wins; tiebreak on node_id (lexicographic)
            winner = max(eligible, key=lambda n: (n.compute_score, n.node_id))
            self._coordinator_id = winner.node_id
