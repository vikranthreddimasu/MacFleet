"""Base node class for MacFleet distributed training.

Provides common functionality shared by coordinator and worker nodes,
including network setup, tensor transport, and service discovery.
"""

import asyncio
import logging
import signal
from abc import ABC, abstractmethod
from typing import Optional

import torch

from macfleet.comm.discovery import ServiceRegistry
from macfleet.comm.transport import TensorTransport
from macfleet.core.config import (
    ClusterConfig,
    ClusterState,
    NodeConfig,
    NodeRole,
)
from macfleet.utils.network import (
    get_gpu_info,
    get_hostname,
    get_local_ip,
    get_memory_bandwidth,
    get_memory_info,
)

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for all MacFleet nodes.

    Provides common functionality for both coordinator (master)
    and worker nodes including:
    - Network and transport setup
    - Service discovery
    - Node information gathering
    - Graceful shutdown handling
    """

    def __init__(
        self,
        cluster_config: ClusterConfig,
        node_config: Optional[NodeConfig] = None,
    ):
        """Initialize the node.

        Args:
            cluster_config: Cluster configuration.
            node_config: Optional pre-configured node info.
                        If not provided, will be auto-detected.
        """
        self._cluster_config = cluster_config
        self._node_config = node_config or self._detect_node_config()

        # Override IP if host was explicitly specified
        if cluster_config.host:
            self._node_config.ip_address = cluster_config.host

        self._cluster_state = ClusterState()
        self._transport: Optional[TensorTransport] = None
        self._service_registry: Optional[ServiceRegistry] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

    def _detect_node_config(self) -> NodeConfig:
        """Auto-detect node configuration from system info."""
        hostname = get_hostname()
        ip_address = get_local_ip()
        gpu_info = get_gpu_info()
        memory_info = get_memory_info()
        memory_bandwidth = get_memory_bandwidth()

        return NodeConfig(
            hostname=hostname,
            ip_address=ip_address,
            gpu_cores=gpu_info.get("gpu_cores", 10),
            ram_gb=memory_info.get("total_gb", 16),
            memory_bandwidth_gbps=memory_bandwidth,
            tensor_port=self._cluster_config.tensor_port,
            rank=-1,  # Will be assigned by coordinator
            workload_weight=0.0,  # Will be assigned by coordinator
        )

    @property
    def hostname(self) -> str:
        """Get the node hostname."""
        return self._node_config.hostname

    @property
    def ip_address(self) -> str:
        """Get the node IP address."""
        return self._node_config.ip_address

    @property
    def rank(self) -> int:
        """Get the node rank."""
        return self._node_config.rank

    @property
    def workload_weight(self) -> float:
        """Get the workload weight."""
        return self._node_config.workload_weight

    @property
    def is_master(self) -> bool:
        """Check if this is the master node."""
        return self._cluster_config.role == NodeRole.MASTER

    @property
    def is_running(self) -> bool:
        """Check if the node is running."""
        return self._running

    @property
    def cluster_state(self) -> ClusterState:
        """Get the cluster state."""
        return self._cluster_state

    @property
    def node_config(self) -> NodeConfig:
        """Get the node configuration."""
        return self._node_config

    @property
    def cluster_config(self) -> ClusterConfig:
        """Get the cluster configuration."""
        return self._cluster_config

    async def _setup_transport(self) -> None:
        """Set up the tensor transport layer."""
        self._transport = TensorTransport(
            host=self._node_config.ip_address,
            port=self._node_config.tensor_port,
        )

    async def _setup_discovery(self) -> None:
        """Set up service discovery using async zeroconf."""
        if self._cluster_config.discovery_enabled:
            self._service_registry = ServiceRegistry()

            try:
                gpu_info = get_gpu_info()
                memory_info = get_memory_info()

                await self._service_registry.async_register_service(
                    hostname=self._node_config.hostname,
                    ip_address=self._node_config.ip_address,
                    grpc_port=self._cluster_config.master_port,
                    tensor_port=self._node_config.tensor_port,
                    role=self._cluster_config.role.value,
                    gpu_cores=gpu_info.get("gpu_cores", 10),
                    ram_gb=memory_info.get("total_gb", 16),
                )
            except Exception as e:
                logger.warning("Service discovery registration failed: %s", e)
                self._service_registry = None

    async def _teardown_transport(self) -> None:
        """Tear down the tensor transport layer."""
        if self._transport:
            await self._transport.stop_server()
            self._transport = None

    async def _teardown_discovery(self) -> None:
        """Tear down service discovery."""
        if self._service_registry:
            try:
                await self._service_registry.async_stop()
            except Exception:
                pass
            self._service_registry = None

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown.

        Uses loop.call_soon_threadsafe to safely set the asyncio Event
        from a signal handler context (which runs outside the event loop).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def signal_handler(sig, frame):
            logger.info("Received signal %s, shutting down...", sig)
            self._running = False
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(self._shutdown_event.set)
            else:
                self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    @abstractmethod
    async def start(self) -> None:
        """Start the node.

        Subclasses must implement this to start their specific
        services (gRPC server for master, client for worker).
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the node.

        Subclasses must implement this to stop their specific
        services cleanly.
        """
        pass

    @abstractmethod
    async def run(self) -> None:
        """Main run loop for the node.

        Subclasses must implement this to define their main
        operational loop.
        """
        pass

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def get_device(self) -> str:
        """Get the PyTorch device for this node.

        Returns "mps" if available, otherwise "cpu".
        """
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def sync_device(self) -> None:
        """Synchronize the MPS device.

        Important for timing and ensuring computations are complete
        before sending data over the network.
        """
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        """Empty the MPS cache to free memory.

        Helps prevent memory fragmentation on the MacBook Air.
        """
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def __repr__(self) -> str:
        role = "Master" if self.is_master else "Worker"
        return (
            f"{role}Node("
            f"rank={self.rank}, "
            f"host={self.hostname}, "
            f"ip={self.ip_address}, "
            f"gpu_cores={self._node_config.gpu_cores}, "
            f"ram_gb={self._node_config.ram_gb})"
        )
