"""Bonjour/zeroconf service discovery for MacFleet.

Enables automatic discovery of MacFleet nodes on the local network
using mDNS/DNS-SD (Bonjour on macOS).
"""

import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from zeroconf import ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf
from zeroconf.asyncio import AsyncZeroconf

# Service type for MacFleet
MACFLEET_SERVICE_TYPE = "_macfleet._tcp.local."

# Default TTL for service registration
DEFAULT_TTL = 120


@dataclass
class DiscoveredNode:
    """Information about a discovered MacFleet node."""
    hostname: str
    ip_address: str
    grpc_port: int
    tensor_port: int
    role: str
    gpu_cores: int
    ram_gb: int


class MacFleetServiceListener(ServiceListener):
    """Listener for MacFleet service discovery events."""

    def __init__(
        self,
        on_add: Optional[Callable[[DiscoveredNode], None]] = None,
        on_remove: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[DiscoveredNode], None]] = None,
    ):
        """Initialize the listener.

        Args:
            on_add: Callback when a node is discovered.
            on_remove: Callback when a node is removed.
            on_update: Callback when a node is updated.
        """
        self._on_add = on_add
        self._on_remove = on_remove
        self._on_update = on_update
        self._zeroconf: Optional[Zeroconf] = None

    def set_zeroconf(self, zc: Zeroconf) -> None:
        """Set the Zeroconf instance for service info lookup."""
        self._zeroconf = zc

    def add_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Called when a service is discovered."""
        info = zc.get_service_info(service_type, name)
        if info and self._on_add:
            node = self._parse_service_info(info)
            if node:
                self._on_add(node)

    def remove_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Called when a service is removed."""
        if self._on_remove:
            # Extract hostname from service name
            hostname = name.replace(f".{service_type}", "")
            self._on_remove(hostname)

    def update_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Called when a service is updated."""
        info = zc.get_service_info(service_type, name)
        if info and self._on_update:
            node = self._parse_service_info(info)
            if node:
                self._on_update(node)

    def _parse_service_info(self, info: ServiceInfo) -> Optional[DiscoveredNode]:
        """Parse service info into a DiscoveredNode."""
        try:
            # Get IP address
            if info.addresses:
                ip_address = socket.inet_ntoa(info.addresses[0])
            else:
                return None

            # Parse properties
            props = info.properties
            grpc_port = info.port
            tensor_port = int(props.get(b"tensor_port", b"50052").decode())
            role = props.get(b"role", b"worker").decode()
            gpu_cores = int(props.get(b"gpu_cores", b"0").decode())
            ram_gb = int(props.get(b"ram_gb", b"0").decode())

            # Get hostname from server name
            hostname = info.server.rstrip(".")

            return DiscoveredNode(
                hostname=hostname,
                ip_address=ip_address,
                grpc_port=grpc_port,
                tensor_port=tensor_port,
                role=role,
                gpu_cores=gpu_cores,
                ram_gb=ram_gb,
            )
        except (ValueError, AttributeError):
            return None


class ServiceRegistry:
    """Register and discover MacFleet services using Bonjour/zeroconf."""

    def __init__(self):
        """Initialize the registry."""
        self._zeroconf: Optional[Zeroconf] = None
        self._async_zeroconf: Optional[AsyncZeroconf] = None
        self._service_info: Optional[ServiceInfo] = None
        self._browser: Optional[ServiceBrowser] = None
        self._listener: Optional[MacFleetServiceListener] = None
        self._discovered_nodes: dict[str, DiscoveredNode] = {}
        self._nodes_lock = threading.Lock()  # Protects _discovered_nodes from Zeroconf threads

    def start(self) -> None:
        """Start the zeroconf instance (synchronous)."""
        if not self._zeroconf:
            self._zeroconf = Zeroconf()

    async def async_start(self) -> None:
        """Start the zeroconf instance (async-safe)."""
        if not self._async_zeroconf:
            self._async_zeroconf = AsyncZeroconf()
            self._zeroconf = self._async_zeroconf.zeroconf

    def stop(self) -> None:
        """Stop the zeroconf instance and cleanup (synchronous)."""
        if self._browser:
            self._browser.cancel()
            self._browser = None

        if self._service_info and self._zeroconf:
            self._zeroconf.unregister_service(self._service_info)
            self._service_info = None

        if self._zeroconf:
            self._zeroconf.close()
            self._zeroconf = None

        self._async_zeroconf = None
        with self._nodes_lock:
            self._discovered_nodes.clear()

    async def async_stop(self) -> None:
        """Stop the zeroconf instance and cleanup (async-safe)."""
        if self._browser:
            self._browser.cancel()
            self._browser = None

        if self._service_info and self._async_zeroconf:
            await self._async_zeroconf.async_unregister_service(self._service_info)
            self._service_info = None

        if self._async_zeroconf:
            await self._async_zeroconf.async_close()
            self._async_zeroconf = None
            self._zeroconf = None

        with self._nodes_lock:
            self._discovered_nodes.clear()

    def _build_service_info(
        self,
        hostname: str,
        ip_address: str,
        grpc_port: int,
        tensor_port: int,
        role: str,
        gpu_cores: int,
        ram_gb: int,
    ) -> ServiceInfo:
        """Build a ServiceInfo object for registration."""
        service_name = f"{hostname}.{MACFLEET_SERVICE_TYPE}"

        properties = {
            b"role": role.encode(),
            b"tensor_port": str(tensor_port).encode(),
            b"gpu_cores": str(gpu_cores).encode(),
            b"ram_gb": str(ram_gb).encode(),
        }

        return ServiceInfo(
            MACFLEET_SERVICE_TYPE,
            service_name,
            addresses=[socket.inet_aton(ip_address)],
            port=grpc_port,
            properties=properties,
            server=f"{hostname}.local.",
        )

    def register_service(
        self,
        hostname: str,
        ip_address: str,
        grpc_port: int,
        tensor_port: int,
        role: str,
        gpu_cores: int,
        ram_gb: int,
    ) -> None:
        """Register this node as a MacFleet service (synchronous).

        Args:
            hostname: This node's hostname.
            ip_address: This node's IP address.
            grpc_port: gRPC port for control messages.
            tensor_port: Port for tensor transfers.
            role: Node role ("master" or "worker").
            gpu_cores: Number of GPU cores.
            ram_gb: RAM in gigabytes.
        """
        if not self._zeroconf:
            self.start()

        self._service_info = self._build_service_info(
            hostname, ip_address, grpc_port, tensor_port, role, gpu_cores, ram_gb,
        )
        self._zeroconf.register_service(self._service_info, ttl=DEFAULT_TTL)

    async def async_register_service(
        self,
        hostname: str,
        ip_address: str,
        grpc_port: int,
        tensor_port: int,
        role: str,
        gpu_cores: int,
        ram_gb: int,
    ) -> None:
        """Register this node as a MacFleet service (async-safe).

        Args:
            hostname: This node's hostname.
            ip_address: This node's IP address.
            grpc_port: gRPC port for control messages.
            tensor_port: Port for tensor transfers.
            role: Node role ("master" or "worker").
            gpu_cores: Number of GPU cores.
            ram_gb: RAM in gigabytes.
        """
        if not self._async_zeroconf:
            await self.async_start()

        self._service_info = self._build_service_info(
            hostname, ip_address, grpc_port, tensor_port, role, gpu_cores, ram_gb,
        )
        await self._async_zeroconf.async_register_service(
            self._service_info, ttl=DEFAULT_TTL,
        )

    def unregister_service(self) -> None:
        """Unregister this node's service."""
        if self._service_info and self._zeroconf:
            self._zeroconf.unregister_service(self._service_info)
            self._service_info = None

    def start_discovery(
        self,
        on_add: Optional[Callable[[DiscoveredNode], None]] = None,
        on_remove: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[DiscoveredNode], None]] = None,
    ) -> None:
        """Start discovering MacFleet services on the network.

        Args:
            on_add: Callback when a node is discovered.
            on_remove: Callback when a node is removed.
            on_update: Callback when a node is updated.
        """
        if not self._zeroconf:
            self.start()

        # Create listener with thread-safe node tracking
        def track_add(node: DiscoveredNode) -> None:
            with self._nodes_lock:
                self._discovered_nodes[node.hostname] = node
            if on_add:
                on_add(node)

        def track_remove(hostname: str) -> None:
            with self._nodes_lock:
                self._discovered_nodes.pop(hostname, None)
            if on_remove:
                on_remove(hostname)

        def track_update(node: DiscoveredNode) -> None:
            with self._nodes_lock:
                self._discovered_nodes[node.hostname] = node
            if on_update:
                on_update(node)

        self._listener = MacFleetServiceListener(
            on_add=track_add,
            on_remove=track_remove,
            on_update=track_update,
        )

        self._browser = ServiceBrowser(
            self._zeroconf,
            MACFLEET_SERVICE_TYPE,
            self._listener,
        )

    def stop_discovery(self) -> None:
        """Stop service discovery."""
        if self._browser:
            self._browser.cancel()
            self._browser = None
            self._listener = None

    def get_discovered_nodes(self) -> list[DiscoveredNode]:
        """Get list of currently discovered nodes."""
        with self._nodes_lock:
            return list(self._discovered_nodes.values())

    def find_master(self, timeout: float = 5.0) -> Optional[DiscoveredNode]:
        """Find the master node on the network.

        Args:
            timeout: Maximum time to search in seconds.

        Returns:
            The master node if found, None otherwise.
        """
        if not self._zeroconf:
            self.start()

        master_node: Optional[DiscoveredNode] = None

        def on_add(node: DiscoveredNode) -> None:
            nonlocal master_node
            if node.role == "master":
                master_node = node

        # Start discovery
        listener = MacFleetServiceListener(on_add=on_add)
        browser = ServiceBrowser(self._zeroconf, MACFLEET_SERVICE_TYPE, listener)

        # Wait for master or timeout
        start = time.time()
        while time.time() - start < timeout:
            if master_node:
                break
            time.sleep(0.1)

        browser.cancel()
        return master_node

    @property
    def is_registered(self) -> bool:
        """Check if this node is registered."""
        return self._service_info is not None

    @property
    def is_discovering(self) -> bool:
        """Check if discovery is active."""
        return self._browser is not None


def discover_master(
    timeout: float = 5.0,
) -> Optional[DiscoveredNode]:
    """Convenience function to find the master node.

    Args:
        timeout: Maximum time to search.

    Returns:
        The master node if found.
    """
    registry = ServiceRegistry()
    try:
        registry.start()
        return registry.find_master(timeout)
    finally:
        registry.stop()


def register_and_discover(
    hostname: str,
    ip_address: str,
    grpc_port: int,
    tensor_port: int,
    role: str,
    gpu_cores: int,
    ram_gb: int,
    on_node_found: Optional[Callable[[DiscoveredNode], None]] = None,
) -> ServiceRegistry:
    """Register this node and start discovery.

    Returns the registry for later cleanup.

    Args:
        hostname: This node's hostname.
        ip_address: This node's IP address.
        grpc_port: gRPC port.
        tensor_port: Tensor transfer port.
        role: Node role.
        gpu_cores: GPU cores.
        ram_gb: RAM in GB.
        on_node_found: Callback for discovered nodes.

    Returns:
        The ServiceRegistry instance (caller must call stop() when done).
    """
    registry = ServiceRegistry()
    registry.start()
    registry.register_service(
        hostname=hostname,
        ip_address=ip_address,
        grpc_port=grpc_port,
        tensor_port=tensor_port,
        role=role,
        gpu_cores=gpu_cores,
        ram_gb=ram_gb,
    )
    registry.start_discovery(on_add=on_node_found)
    return registry
