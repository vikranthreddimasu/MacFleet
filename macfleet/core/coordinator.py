"""Cluster coordinator (master node) for MacFleet.

The coordinator runs on the master node (MacBook Pro) and handles:
- Node registration and rank assignment
- Cluster state management
- Heartbeat monitoring
- Training coordination
- Workload balancing
"""

import asyncio
import logging
import threading
import time
from typing import Callable, Optional

from rich.console import Console
from rich.table import Table

from macfleet.comm.grpc_service import ClusterControlServicer, GRPCServer
from macfleet.core.config import (
    ClusterConfig,
    NodeConfig,
    NodeRole,
    TrainingConfig,
)
from macfleet.core.node import BaseNode

logger = logging.getLogger(__name__)
console = Console()  # Keep for Rich table output only


class Coordinator(BaseNode):
    """Master node coordinator for the MacFleet cluster.

    Handles:
    - gRPC server for control plane messages
    - Node registration and rank assignment
    - Heartbeat monitoring and health tracking
    - Workload rebalancing based on throughput/thermal state
    - Training start/stop coordination
    """

    def __init__(
        self,
        cluster_config: ClusterConfig,
        training_config: Optional[TrainingConfig] = None,
        node_config: Optional[NodeConfig] = None,
    ):
        """Initialize the coordinator.

        Args:
            cluster_config: Cluster configuration (role should be MASTER).
            training_config: Optional training configuration.
            node_config: Optional pre-configured node info.
        """
        # Ensure role is master
        if cluster_config.role != NodeRole.MASTER:
            cluster_config.role = NodeRole.MASTER

        super().__init__(cluster_config, node_config)

        self._training_config = training_config or TrainingConfig()
        self._grpc_server: Optional[GRPCServer] = None
        self._servicer: Optional[ClusterControlServicer] = None
        self._heartbeat_tracker: dict[int, float] = {}  # rank -> last_heartbeat_time
        self._throughput_tracker: dict[int, float] = {}  # rank -> samples/sec
        # Protects heartbeat/throughput trackers and cluster state.
        self._state_lock = threading.Lock()
        self._on_node_registered: Optional[Callable[[NodeConfig], None]] = None
        self._on_node_lost: Optional[Callable[[int], None]] = None

        # Register master as rank 0
        self._node_config.rank = 0
        self._node_config.workload_weight = 1.0  # Will be recalculated when workers join
        self._cluster_state.add_node(self._node_config)

    @property
    def training_config(self) -> TrainingConfig:
        """Get the training configuration."""
        return self._training_config

    @training_config.setter
    def training_config(self, config: TrainingConfig) -> None:
        """Set the training configuration."""
        self._training_config = config

    def set_on_node_registered(
        self,
        callback: Callable[[NodeConfig], None],
    ) -> None:
        """Set callback for when a node registers."""
        self._on_node_registered = callback

    def set_on_node_lost(
        self,
        callback: Callable[[int], None],
    ) -> None:
        """Set callback for when a node is lost."""
        self._on_node_lost = callback

    async def start(self) -> None:
        """Start the coordinator services."""
        self._running = True
        self._setup_signal_handlers()

        logger.info("Starting MacFleet Coordinator")
        logger.info("  Hostname: %s", self.hostname)
        logger.info("  IP Address: %s", self.ip_address)
        logger.info("  gRPC Port: %d", self._cluster_config.master_port)
        logger.info("  Tensor Port: %d", self._node_config.tensor_port)

        # Set up tensor transport
        await self._setup_transport()
        await self._transport.start_server()
        logger.info("Tensor transport started")

        # Set up service discovery
        await self._setup_discovery()
        if self._service_registry:
            logger.info("Service discovery started")

        # Create gRPC servicer and server
        self._servicer = ClusterControlServicer(
            cluster_state=self._cluster_state,
            tensor_addr=self.ip_address,
            tensor_port=self._node_config.tensor_port,
            on_register=self._handle_registration,
            on_heartbeat=self._handle_heartbeat,
            on_deregister=self._handle_deregistration,
            auth_token=self._cluster_config.auth_token,
        )

        self._grpc_server = GRPCServer(
            servicer=self._servicer,
            host="0.0.0.0",
            port=self._cluster_config.master_port,
        )
        self._grpc_server.start()
        logger.info("gRPC server started")
        logger.info("Coordinator ready, waiting for workers...")

    async def stop(self) -> None:
        """Stop the coordinator services."""
        logger.info("Shutting down coordinator...")

        self._running = False

        # Stop gRPC server
        if self._grpc_server:
            self._grpc_server.stop()
            self._grpc_server = None
            logger.info("gRPC server stopped")

        # Tear down transport
        await self._teardown_transport()
        logger.info("Tensor transport stopped")

        # Tear down discovery
        await self._teardown_discovery()
        logger.info("Service discovery stopped")

        logger.info("Coordinator shutdown complete")

    async def run(self) -> None:
        """Main run loop for the coordinator.

        Monitors heartbeats and handles node failures.
        """
        await self.start()

        try:
            while self._running:
                # Check for lost nodes
                await self._check_heartbeats()

                # Display status periodically
                if self._cluster_state.world_size > 1:
                    self._print_cluster_status()

                await asyncio.sleep(self._cluster_config.heartbeat_interval_sec)

        finally:
            await self.stop()

    def _handle_registration(self, node: NodeConfig) -> None:
        """Handle a new node registration.

        Called from gRPC thread pool — must use _state_lock for shared state.
        """
        logger.info(
            "Worker registered: %s at %s, rank %d",
            node.hostname, node.ip_address, node.rank,
        )
        logger.info("  GPU Cores: %d, RAM: %d GB, Weight: %.2f%%",
                     node.gpu_cores, node.ram_gb, node.workload_weight * 100)

        with self._state_lock:
            # Initialize heartbeat tracking
            self._heartbeat_tracker[node.rank] = time.time()
            self._throughput_tracker[node.rank] = 0.0

            # Recalculate master's weight
            self._recalculate_weights()

        if self._on_node_registered:
            self._on_node_registered(node)

    def _handle_deregistration(self, rank: int, reason: str) -> None:
        """Handle a graceful node deregistration.

        Called from gRPC thread pool — must use _state_lock for shared state.
        """
        with self._state_lock:
            self._heartbeat_tracker.pop(rank, None)
            self._throughput_tracker.pop(rank, None)

        logger.info("Node deregistered: rank=%d (%s)", rank, reason)

        if self._on_node_lost:
            self._on_node_lost(rank)

    def _handle_heartbeat(
        self,
        rank: int,
        throughput: float,
        thermal_state: str,
    ) -> None:
        """Handle a heartbeat from a worker.

        Called from gRPC thread pool — must use _state_lock for shared state.
        """
        with self._state_lock:
            self._heartbeat_tracker[rank] = time.time()
            self._throughput_tracker[rank] = throughput

        # Check for thermal throttling
        if thermal_state in ("serious", "critical"):
            node = self._cluster_state.get_node(rank)
            if node:
                logger.warning(
                    "Node %d (%s) thermal state: %s",
                    rank, node.hostname, thermal_state,
                )

    async def _check_heartbeats(self) -> None:
        """Check for nodes that haven't sent heartbeats."""
        now = time.time()
        timeout = self._cluster_config.heartbeat_timeout_sec
        lost_ranks = []

        with self._state_lock:
            for rank, last_time in list(self._heartbeat_tracker.items()):
                if rank == 0:  # Skip master
                    continue

                if now - last_time > timeout:
                    lost_ranks.append(rank)

            for rank in lost_ranks:
                node = self._cluster_state.get_node(rank)
                if node:
                    logger.error("Node lost: rank=%d (%s)", rank, node.hostname)

                # Remove from tracking
                self._heartbeat_tracker.pop(rank, None)
                self._throughput_tracker.pop(rank, None)
                self._cluster_state.remove_node(rank)

                # Recalculate weights
                self._recalculate_weights()

        # Return ranks to servicer OUTSIDE _state_lock to avoid lock ordering
        # inversion (_state_lock -> _servicer._lock vs _servicer._lock -> _state_lock)
        if self._servicer and lost_ranks:
            with self._servicer._lock:
                for rank in lost_ranks:
                    self._servicer._free_ranks.append(rank)
                    self._servicer._base_weights.pop(rank, None)

        # Callbacks outside lock
        for rank in lost_ranks:
            if self._on_node_lost:
                self._on_node_lost(rank)

    def _recalculate_weights(self) -> None:
        """Recalculate workload weights based on GPU cores.

        Caller must hold _state_lock.
        """
        total_cores = sum(n.gpu_cores for n in self._cluster_state.nodes.values())
        if total_cores == 0:
            return

        for node in self._cluster_state.nodes.values():
            node.workload_weight = node.gpu_cores / total_cores

    def _print_cluster_status(self) -> None:
        """Print cluster status to console."""
        table = Table(title="Cluster Status")
        table.add_column("Rank", style="cyan")
        table.add_column("Hostname", style="green")
        table.add_column("GPU Cores", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("Throughput", justify="right")

        with self._state_lock:
            nodes = sorted(
                self._cluster_state.nodes.values(),
                key=lambda n: n.rank,
            )
            for node in nodes:
                throughput = self._throughput_tracker.get(node.rank, 0.0)
                table.add_row(
                    str(node.rank),
                    node.hostname,
                    str(node.gpu_cores),
                    f"{node.workload_weight:.1%}",
                    f"{throughput:.1f} samples/s" if throughput > 0 else "-",
                )

        console.print(table)

    def get_worker_addresses(self) -> list[tuple[str, int]]:
        """Get (ip, tensor_port) for all workers."""
        return [
            (node.ip_address, node.tensor_port)
            for node in self._cluster_state.nodes.values()
            if node.rank != 0
        ]

    def get_all_addresses(self) -> list[tuple[str, int]]:
        """Get (ip, tensor_port) for all nodes including master."""
        return [
            (node.ip_address, node.tensor_port)
            for node in self._cluster_state.nodes.values()
        ]

    async def wait_for_workers(
        self,
        min_workers: int = 1,
        timeout: float = 60.0,
    ) -> bool:
        """Wait for a minimum number of workers to register.

        Args:
            min_workers: Minimum number of workers required.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if enough workers registered, False if timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            # Count workers (world_size - 1 for master)
            num_workers = self._cluster_state.world_size - 1
            if num_workers >= min_workers:
                return True
            await asyncio.sleep(0.5)
        return False

    async def broadcast_tensor(
        self,
        tensor,
        msg_type=None,
    ) -> None:
        """Broadcast a tensor to all workers.

        Args:
            tensor: Tensor to broadcast.
            msg_type: Optional message type.
        """
        from macfleet.utils.tensor_utils import MessageType

        if msg_type is None:
            msg_type = MessageType.TENSOR_WEIGHTS

        for ip, port in self.get_worker_addresses():
            conn_key = f"{ip}:{port}"
            if conn_key not in self._transport._connections:
                await self._transport.connect(ip, port)
            await self._transport.send_tensor(tensor, conn_key, msg_type)


async def run_coordinator(
    cluster_config: Optional[ClusterConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> None:
    """Run the coordinator node.

    Convenience function for starting a coordinator.

    Args:
        cluster_config: Optional cluster configuration.
        training_config: Optional training configuration.
    """
    if cluster_config is None:
        cluster_config = ClusterConfig(role=NodeRole.MASTER)

    coordinator = Coordinator(
        cluster_config=cluster_config,
        training_config=training_config,
    )

    await coordinator.run()
