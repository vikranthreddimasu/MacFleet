"""Worker node for MacFleet distributed training.

The worker runs on secondary nodes (e.g., MacBook Air) and handles:
- Registration with the coordinator
- Heartbeat reporting
- Training execution
- Gradient communication
"""

import asyncio
import logging
import subprocess
from typing import Optional

from macfleet.comm.discovery import discover_master
from macfleet.comm.grpc_service import ClusterControlClient
from macfleet.core.config import (
    ClusterConfig,
    NodeConfig,
    NodeRole,
)
from macfleet.core.node import BaseNode

logger = logging.getLogger(__name__)


class Worker(BaseNode):
    """Worker node for the MacFleet cluster.

    Handles:
    - Connection and registration with coordinator
    - Periodic heartbeat sending
    - Tensor communication with coordinator
    - Local training execution
    """

    def __init__(
        self,
        cluster_config: ClusterConfig,
        node_config: Optional[NodeConfig] = None,
        auto_discover: bool = True,
    ):
        """Initialize the worker.

        Args:
            cluster_config: Cluster configuration (role should be WORKER).
            node_config: Optional pre-configured node info.
            auto_discover: Whether to auto-discover the master if not specified.
        """
        # Ensure role is worker
        if cluster_config.role != NodeRole.WORKER:
            cluster_config.role = NodeRole.WORKER

        super().__init__(cluster_config, node_config)

        self._auto_discover = auto_discover
        self._grpc_client: Optional[ClusterControlClient] = None
        self._master_tensor_addr: Optional[str] = None
        self._master_tensor_port: Optional[int] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._current_throughput = 0.0
        self._current_step = 0

    @property
    def master_address(self) -> str:
        """Get the master's gRPC address."""
        return self._cluster_config.master_grpc_address

    async def start(self) -> None:
        """Start the worker and connect to the coordinator."""
        self._running = True
        self._setup_signal_handlers()

        logger.info("Starting MacFleet Worker")
        logger.info("  Hostname: %s", self.hostname)
        logger.info("  IP Address: %s", self.ip_address)
        logger.info("  Tensor Port: %d", self._node_config.tensor_port)

        # Try to auto-discover master if enabled
        if self._auto_discover and not self._cluster_config.master_addr:
            logger.info("Searching for master node...")
            master = await asyncio.to_thread(discover_master, 10.0)
            if master:
                self._cluster_config.master_addr = master.ip_address
                self._cluster_config.master_port = master.grpc_port
                logger.info(
                    "Found master: %s (%s:%d)",
                    master.hostname, master.ip_address, master.grpc_port,
                )
            else:
                logger.warning("No master found via discovery")

        logger.info("  Master: %s", self.master_address)

        # Set up tensor transport
        await self._setup_transport()
        await self._transport.start_server()
        logger.info("Tensor transport started")

        # Set up service discovery
        await self._setup_discovery()
        if self._service_registry:
            logger.info("Service discovery started")

        # Connect to coordinator
        await self._connect_to_coordinator()

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("Worker ready (rank=%d)", self.rank)

    async def stop(self) -> None:
        """Stop the worker and disconnect from coordinator."""
        logger.info("Shutting down worker...")

        self._running = False

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Deregister from coordinator before disconnecting
        if self._grpc_client:
            if self.rank >= 0:
                self._grpc_client.deregister(self.rank, "Graceful shutdown")
            self._grpc_client.disconnect()
            self._grpc_client = None
            logger.info("Deregistered and disconnected from coordinator")

        # Tear down transport
        await self._teardown_transport()
        logger.info("Tensor transport stopped")

        # Tear down discovery
        await self._teardown_discovery()
        logger.info("Service discovery stopped")

        logger.info("Worker shutdown complete")

    async def run(self) -> None:
        """Main run loop for the worker.

        Waits for commands from the coordinator.
        """
        await self.start()

        try:
            # Main loop - wait for shutdown or commands
            while self._running:
                await asyncio.sleep(1.0)

        finally:
            await self.stop()

    async def _connect_to_coordinator(self) -> None:
        """Connect to the coordinator and register."""
        logger.info("Connecting to coordinator...")

        self._grpc_client = ClusterControlClient(
            master_addr=self._cluster_config.master_addr,
            master_port=self._cluster_config.master_port,
            auth_token=self._cluster_config.auth_token,
            auth_token_env=self._cluster_config.auth_token_env,
        )

        # Try to connect with exponential backoff
        import random

        max_retries = 15
        base_delay = 1.0
        max_delay = 30.0

        for attempt in range(max_retries):
            try:
                self._grpc_client.connect()

                # Register with coordinator
                (
                    rank,
                    weight,
                    world_size,
                    master_tensor_addr,
                    master_tensor_port,
                ) = self._grpc_client.register(
                    hostname=self._node_config.hostname,
                    ip_address=self._node_config.ip_address,
                    gpu_cores=self._node_config.gpu_cores,
                    ram_gb=self._node_config.ram_gb,
                    memory_bandwidth_gbps=self._node_config.memory_bandwidth_gbps,
                    tensor_port=self._node_config.tensor_port,
                )

                # Update node config
                self._node_config.rank = rank
                self._node_config.workload_weight = weight
                self._master_tensor_addr = master_tensor_addr
                self._master_tensor_port = master_tensor_port

                logger.info("Registered with master, assigned rank %d", rank)
                logger.info("  Workload weight: %.1f%%", weight * 100)
                logger.info("  World size: %d", world_size)

                # Connect to master's tensor channel
                await self._transport.connect(
                    master_tensor_addr,
                    master_tensor_port,
                )
                logger.info("Connected to master tensor channel")

                return

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay *= 0.5 + random.random()  # Add jitter
                    logger.warning(
                        "Attempt %d/%d failed: %s (retry in %.1fs)",
                        attempt + 1, max_retries, e, delay,
                    )
                    self._grpc_client.disconnect()
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(
                        f"Failed to connect to coordinator after {max_retries} attempts"
                    )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the coordinator."""
        interval = self._cluster_config.heartbeat_interval_sec

        while self._running:
            try:
                thermal = self._get_thermal_state()
                acknowledged, new_weight, should_stop = self._grpc_client.heartbeat(
                    rank=self.rank,
                    throughput=self._current_throughput,
                    thermal_state=thermal,
                    current_step=self._current_step,
                )

                # Update weight if changed
                if new_weight != self._node_config.workload_weight:
                    old_weight = self._node_config.workload_weight
                    self._node_config.workload_weight = new_weight
                    logger.info(
                        "Workload weight changed: %.1f%% -> %.1f%%",
                        old_weight * 100, new_weight * 100,
                    )

                # Check for stop signal
                if should_stop:
                    logger.info("Received stop signal from coordinator")
                    self._running = False
                    break

            except Exception as e:
                logger.error("Heartbeat failed: %s", e)
                # Only reconnect the gRPC channel, do NOT re-register
                # (re-registration would assign a new rank and break training state)
                try:
                    if self._grpc_client:
                        self._grpc_client.disconnect()
                        self._grpc_client.connect()
                except Exception:
                    pass

            await asyncio.sleep(interval)

    def _get_thermal_state(self) -> str:
        """Get the current thermal state from macOS.

        Returns one of: "nominal", "fair", "serious", "critical"
        """
        try:
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                output = result.stdout.lower()
                if "critical" in output:
                    return "critical"
                elif "serious" in output:
                    return "serious"
                elif "fair" in output:
                    return "fair"

            return "nominal"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "nominal"

    def update_throughput(self, samples_per_sec: float) -> None:
        """Update the current throughput measurement.

        Args:
            samples_per_sec: Current training throughput.
        """
        self._current_throughput = samples_per_sec

    def update_step(self, step: int) -> None:
        """Update the current training step.

        Args:
            step: Current step number.
        """
        self._current_step = step

    async def send_tensor_to_master(
        self,
        tensor,
        msg_type=None,
    ) -> None:
        """Send a tensor to the master node.

        Args:
            tensor: Tensor to send.
            msg_type: Optional message type.
        """
        from macfleet.utils.tensor_utils import MessageType

        if msg_type is None:
            msg_type = MessageType.TENSOR_GRADIENT

        conn_key = f"{self._master_tensor_addr}:{self._master_tensor_port}"
        await self._transport.send_tensor(tensor, conn_key, msg_type)

    async def recv_tensor_from_master(
        self,
        device: Optional[str] = None,
    ):
        """Receive a tensor from the master node.

        Args:
            device: Target device for the tensor.

        Returns:
            Tuple of (tensor, message_type).
        """
        conn_key = f"{self._master_tensor_addr}:{self._master_tensor_port}"
        return await self._transport.recv_tensor(conn_key, device)

    async def sync_barrier(self, barrier_id: str) -> bool:
        """Wait at a synchronization barrier.

        Args:
            barrier_id: Unique identifier for this barrier.

        Returns:
            True when all nodes have reached the barrier.
        """
        if not self._grpc_client:
            raise RuntimeError("Not connected to coordinator")

        # Poll until barrier is released
        while True:
            proceed, nodes = self._grpc_client.sync_barrier(
                rank=self.rank,
                step=self._current_step,
                barrier_id=barrier_id,
            )

            if proceed:
                return True

            await asyncio.sleep(0.01)  # Small delay between polls


async def run_worker(
    cluster_config: Optional[ClusterConfig] = None,
    master_addr: Optional[str] = None,
    master_port: int = 50051,
) -> None:
    """Run a worker node.

    Convenience function for starting a worker.

    Args:
        cluster_config: Optional cluster configuration.
        master_addr: Master node address (overrides config).
        master_port: Master node port (overrides config).
    """
    if cluster_config is None:
        cluster_config = ClusterConfig(
            role=NodeRole.WORKER,
            master_addr=master_addr or "10.0.0.1",
            master_port=master_port,
        )

    if master_addr:
        cluster_config.master_addr = master_addr
    if master_port:
        cluster_config.master_port = master_port

    worker = Worker(cluster_config=cluster_config)
    await worker.run()
