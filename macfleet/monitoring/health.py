"""Health monitoring for MacFleet distributed training.

Tracks node liveness via heartbeats and handles failure detection
and recovery for fault-tolerant training.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

from rich.console import Console

console = Console()


class NodeStatus(Enum):
    """Status of a node in the cluster."""
    UNKNOWN = "unknown"
    CONNECTED = "connected"
    TRAINING = "training"
    DISCONNECTED = "disconnected"
    RECOVERING = "recovering"


@dataclass
class NodeHealth:
    """Health information for a single node."""
    rank: int
    hostname: str
    status: NodeStatus = NodeStatus.UNKNOWN
    last_heartbeat: float = 0.0
    throughput: float = 0.0
    thermal_state: str = "nominal"
    current_step: int = 0
    consecutive_failures: int = 0

    @property
    def is_alive(self) -> bool:
        """Check if node is considered alive."""
        return self.status in (NodeStatus.CONNECTED, NodeStatus.TRAINING)

    @property
    def seconds_since_heartbeat(self) -> float:
        """Seconds since last heartbeat."""
        if self.last_heartbeat == 0:
            return float("inf")
        return time.time() - self.last_heartbeat


@dataclass
class HealthConfig:
    """Configuration for health monitoring."""
    heartbeat_interval_sec: float = 2.0
    heartbeat_timeout_sec: float = 6.0
    max_consecutive_failures: int = 3
    recovery_grace_period_sec: float = 10.0


class HealthMonitor:
    """Monitor health of all nodes in the cluster.

    Tracks heartbeats from each node and detects failures.
    When a node fails, it can trigger recovery actions.
    """

    def __init__(
        self,
        config: Optional[HealthConfig] = None,
        on_node_lost: Optional[Callable[[int], None]] = None,
        on_node_recovered: Optional[Callable[[int], None]] = None,
    ):
        """Initialize the health monitor.

        Args:
            config: Health monitoring configuration.
            on_node_lost: Callback when a node is detected as lost.
            on_node_recovered: Callback when a node recovers.
        """
        self._config = config or HealthConfig()
        self._nodes: Dict[int, NodeHealth] = {}
        self._on_node_lost = on_node_lost
        self._on_node_recovered = on_node_recovered
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    def register_node(
        self,
        rank: int,
        hostname: str,
    ) -> NodeHealth:
        """Register a node for health monitoring.

        Args:
            rank: Node's rank.
            hostname: Node's hostname.

        Returns:
            NodeHealth object for the node.
        """
        health = NodeHealth(
            rank=rank,
            hostname=hostname,
            status=NodeStatus.CONNECTED,
            last_heartbeat=time.time(),
        )
        self._nodes[rank] = health
        return health

    def unregister_node(self, rank: int) -> None:
        """Unregister a node from monitoring."""
        self._nodes.pop(rank, None)

    def record_heartbeat(
        self,
        rank: int,
        throughput: float = 0.0,
        thermal_state: str = "nominal",
        current_step: int = 0,
    ) -> None:
        """Record a heartbeat from a node.

        Args:
            rank: Node's rank.
            throughput: Reported throughput (samples/sec).
            thermal_state: Thermal state string.
            current_step: Current training step.
        """
        if rank not in self._nodes:
            return

        node = self._nodes[rank]
        was_disconnected = node.status == NodeStatus.DISCONNECTED

        node.last_heartbeat = time.time()
        node.throughput = throughput
        node.thermal_state = thermal_state
        node.current_step = current_step
        node.consecutive_failures = 0

        if was_disconnected:
            node.status = NodeStatus.RECOVERING
            console.print(f"[green]Node {rank} ({node.hostname}) reconnected[/green]")
            if self._on_node_recovered:
                self._on_node_recovered(rank)
        else:
            node.status = NodeStatus.TRAINING if current_step > 0 else NodeStatus.CONNECTED

    def get_node_health(self, rank: int) -> Optional[NodeHealth]:
        """Get health info for a specific node."""
        return self._nodes.get(rank)

    def get_all_health(self) -> Dict[int, NodeHealth]:
        """Get health info for all nodes."""
        return self._nodes.copy()

    def get_alive_nodes(self) -> list[int]:
        """Get list of ranks for alive nodes."""
        return [rank for rank, node in self._nodes.items() if node.is_alive]

    def get_dead_nodes(self) -> list[int]:
        """Get list of ranks for dead nodes."""
        return [rank for rank, node in self._nodes.items() if not node.is_alive]

    async def start(self) -> None:
        """Start the health monitoring loop."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """Main monitoring loop that checks for dead nodes."""
        while self._running:
            await self._check_nodes()
            await asyncio.sleep(self._config.heartbeat_interval_sec)

    async def _check_nodes(self) -> None:
        """Check all nodes for timeout."""
        now = time.time()
        timeout = self._config.heartbeat_timeout_sec

        for rank, node in list(self._nodes.items()):
            if rank == 0:  # Skip master
                continue

            if node.status == NodeStatus.DISCONNECTED:
                continue

            elapsed = now - node.last_heartbeat
            if elapsed > timeout:
                node.consecutive_failures += 1

                if node.consecutive_failures >= self._config.max_consecutive_failures:
                    self._mark_node_dead(rank)

    def _mark_node_dead(self, rank: int) -> None:
        """Mark a node as dead and trigger callback."""
        node = self._nodes.get(rank)
        if not node:
            return

        if node.status != NodeStatus.DISCONNECTED:
            node.status = NodeStatus.DISCONNECTED
            console.print(
                f"[bold red]Node {rank} ({node.hostname}) lost - "
                f"no heartbeat for {node.seconds_since_heartbeat:.1f}s[/bold red]"
            )
            if self._on_node_lost:
                self._on_node_lost(rank)

    @property
    def world_size(self) -> int:
        """Current number of registered nodes."""
        return len(self._nodes)

    @property
    def alive_count(self) -> int:
        """Number of alive nodes."""
        return len(self.get_alive_nodes())


class HeartbeatSender:
    """Send periodic heartbeats to the coordinator.

    Used by worker nodes to report their status.
    """

    def __init__(
        self,
        rank: int,
        send_fn: Callable,
        interval_sec: float = 2.0,
    ):
        """Initialize the heartbeat sender.

        Args:
            rank: This node's rank.
            send_fn: Function to send heartbeat (async or sync).
            interval_sec: Interval between heartbeats.
        """
        self._rank = rank
        self._send_fn = send_fn
        self._interval = interval_sec
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Metrics to include in heartbeat
        self.throughput = 0.0
        self.thermal_state = "nominal"
        self.current_step = 0

    async def start(self) -> None:
        """Start sending heartbeats."""
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop sending heartbeats."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _heartbeat_loop(self) -> None:
        """Main loop for sending heartbeats."""
        while self._running:
            try:
                await self._send_heartbeat()
            except Exception as e:
                console.print(f"[yellow]Heartbeat failed: {e}[/yellow]")

            await asyncio.sleep(self._interval)

    async def _send_heartbeat(self) -> None:
        """Send a single heartbeat."""
        result = self._send_fn(
            rank=self._rank,
            throughput=self.throughput,
            thermal_state=self.thermal_state,
            current_step=self.current_step,
        )

        # Handle both async and sync send functions
        if asyncio.iscoroutine(result):
            await result

    def update_metrics(
        self,
        throughput: Optional[float] = None,
        thermal_state: Optional[str] = None,
        current_step: Optional[int] = None,
    ) -> None:
        """Update metrics to include in next heartbeat."""
        if throughput is not None:
            self.throughput = throughput
        if thermal_state is not None:
            self.thermal_state = thermal_state
        if current_step is not None:
            self.current_step = current_step


def check_node_liveness(
    nodes: Dict[int, NodeHealth],
    timeout_sec: float = 6.0,
) -> tuple[list[int], list[int]]:
    """Check which nodes are alive and which are dead.

    Args:
        nodes: Dictionary of node health info.
        timeout_sec: Timeout threshold.

    Returns:
        Tuple of (alive_ranks, dead_ranks).
    """
    now = time.time()
    alive = []
    dead = []

    for rank, node in nodes.items():
        elapsed = now - node.last_heartbeat
        if elapsed <= timeout_sec:
            alive.append(rank)
        else:
            dead.append(rank)

    return alive, dead
