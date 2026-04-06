"""Bonjour/zeroconf service discovery for MacFleet v2.

Ported from v1 comm/discovery.py with extended properties for pool metadata:
- node_id, chip_name, link_types, pool_version added to mDNS properties
- Removed master/worker role distinction (v2 uses peer model with elected coordinator)
"""

import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from zeroconf import ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf
from zeroconf.asyncio import AsyncZeroconf

import macfleet
from macfleet.security.auth import DEFAULT_SERVICE_TYPE, SecurityConfig

MACFLEET_SERVICE_TYPE = DEFAULT_SERVICE_TYPE
DEFAULT_TTL = 120


@dataclass
class DiscoveredNode:
    """A node discovered via mDNS."""
    hostname: str
    node_id: str
    ip_address: str
    port: int  # main communication port
    gpu_cores: int
    ram_gb: int
    chip_name: str
    link_types: str  # comma-separated: "wifi,ethernet,thunderbolt"
    pool_version: str
    compute_score: float = 0.0

    @property
    def link_type_list(self) -> list[str]:
        return [lt.strip() for lt in self.link_types.split(",") if lt.strip()]


class PoolServiceListener(ServiceListener):
    """Listener for MacFleet pool service discovery events."""

    def __init__(
        self,
        on_add: Optional[Callable[[DiscoveredNode], None]] = None,
        on_remove: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[DiscoveredNode], None]] = None,
    ):
        self._on_add = on_add
        self._on_remove = on_remove
        self._on_update = on_update

    def add_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        info = zc.get_service_info(service_type, name)
        if info and self._on_add:
            node = self._parse_service_info(info)
            if node:
                self._on_add(node)

    def remove_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        if self._on_remove:
            hostname = name.replace(f".{service_type}", "")
            self._on_remove(hostname)

    def update_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        info = zc.get_service_info(service_type, name)
        if info and self._on_update:
            node = self._parse_service_info(info)
            if node:
                self._on_update(node)

    def _parse_service_info(self, info: ServiceInfo) -> Optional[DiscoveredNode]:
        try:
            if info.addresses:
                ip_address = socket.inet_ntoa(info.addresses[0])
            else:
                return None

            props = info.properties
            hostname = info.server.rstrip(".")
            node_id = props.get(b"node_id", b"").decode() or hostname
            gpu_cores = int(props.get(b"gpu_cores", b"0").decode())
            ram_gb = int(props.get(b"ram_gb", b"0").decode())
            chip_name = props.get(b"chip_name", b"unknown").decode()
            link_types = props.get(b"link_types", b"").decode()
            pool_version = props.get(b"pool_version", b"0.0.0").decode()
            compute_score = float(props.get(b"compute_score", b"0").decode())

            return DiscoveredNode(
                hostname=hostname,
                node_id=node_id,
                ip_address=ip_address,
                port=info.port,
                gpu_cores=gpu_cores,
                ram_gb=ram_gb,
                chip_name=chip_name,
                link_types=link_types,
                pool_version=pool_version,
                compute_score=compute_score,
            )
        except (ValueError, AttributeError):
            return None


class ServiceRegistry:
    """Register and discover MacFleet pool members using Bonjour/zeroconf."""

    def __init__(self, security: Optional[SecurityConfig] = None):
        self._security = security or SecurityConfig()
        self._service_type = self._security.mdns_service_type
        self._zeroconf: Optional[Zeroconf] = None
        self._async_zeroconf: Optional[AsyncZeroconf] = None
        self._service_info: Optional[ServiceInfo] = None
        self._browser: Optional[ServiceBrowser] = None
        self._listener: Optional[PoolServiceListener] = None
        self._discovered_nodes: dict[str, DiscoveredNode] = {}
        self._nodes_lock = threading.Lock()

    def start(self) -> None:
        if not self._zeroconf:
            self._zeroconf = Zeroconf()

    async def async_start(self) -> None:
        if not self._async_zeroconf:
            self._async_zeroconf = AsyncZeroconf()
            self._zeroconf = self._async_zeroconf.zeroconf

    def stop(self) -> None:
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

    def _build_properties(
        self,
        node_id: str,
        gpu_cores: int,
        ram_gb: int,
        chip_name: str,
        link_types: str,
        compute_score: float,
    ) -> dict[bytes, bytes]:
        """Build mDNS service properties.

        SECURITY: When fleet is token-protected, minimize broadcast info.
        Only broadcast node_id, port, and version. Hardware details
        (GPU cores, RAM, chip) are exchanged AFTER authenticated connection.
        """
        if self._security.is_secure:
            return {
                b"node_id": node_id.encode(),
                b"pool_version": macfleet.__version__.encode(),
            }
        return {
            b"node_id": node_id.encode(),
            b"gpu_cores": str(gpu_cores).encode(),
            b"ram_gb": str(ram_gb).encode(),
            b"chip_name": chip_name.encode(),
            b"link_types": link_types.encode(),
            b"pool_version": macfleet.__version__.encode(),
            b"compute_score": f"{compute_score:.1f}".encode(),
        }

    def register_node(
        self,
        hostname: str,
        node_id: str,
        ip_address: str,
        port: int,
        gpu_cores: int,
        ram_gb: int,
        chip_name: str = "unknown",
        link_types: str = "",
        compute_score: float = 0.0,
    ) -> None:
        """Register this node in the pool via mDNS."""
        if not self._zeroconf:
            self.start()

        service_name = f"{node_id}.{self._service_type}"
        properties = self._build_properties(
            node_id, gpu_cores, ram_gb, chip_name, link_types, compute_score,
        )

        self._service_info = ServiceInfo(
            self._service_type,
            service_name,
            addresses=[socket.inet_aton(ip_address)],
            port=port,
            properties=properties,
            server=f"{hostname}.local.",
        )
        self._zeroconf.register_service(self._service_info, ttl=DEFAULT_TTL)

    async def async_register_node(
        self,
        hostname: str,
        node_id: str,
        ip_address: str,
        port: int,
        gpu_cores: int,
        ram_gb: int,
        chip_name: str = "unknown",
        link_types: str = "",
        compute_score: float = 0.0,
    ) -> None:
        """Register this node in the pool via mDNS (async)."""
        if not self._async_zeroconf:
            await self.async_start()

        service_name = f"{node_id}.{self._service_type}"
        properties = self._build_properties(
            node_id, gpu_cores, ram_gb, chip_name, link_types, compute_score,
        )

        self._service_info = ServiceInfo(
            self._service_type,
            service_name,
            addresses=[socket.inet_aton(ip_address)],
            port=port,
            properties=properties,
            server=f"{hostname}.local.",
        )
        await self._async_zeroconf.async_register_service(
            self._service_info, ttl=DEFAULT_TTL,
        )

    def start_discovery(
        self,
        on_add: Optional[Callable[[DiscoveredNode], None]] = None,
        on_remove: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[DiscoveredNode], None]] = None,
    ) -> None:
        """Start discovering pool members on the network."""
        if not self._zeroconf:
            self.start()

        def track_add(node: DiscoveredNode) -> None:
            with self._nodes_lock:
                self._discovered_nodes[node.node_id] = node
            if on_add:
                on_add(node)

        def track_remove(hostname: str) -> None:
            with self._nodes_lock:
                self._discovered_nodes.pop(hostname, None)
            if on_remove:
                on_remove(hostname)

        def track_update(node: DiscoveredNode) -> None:
            with self._nodes_lock:
                self._discovered_nodes[node.node_id] = node
            if on_update:
                on_update(node)

        self._listener = PoolServiceListener(
            on_add=track_add, on_remove=track_remove, on_update=track_update,
        )
        self._browser = ServiceBrowser(
            self._zeroconf, self._service_type, self._listener,
        )

    def stop_discovery(self) -> None:
        if self._browser:
            self._browser.cancel()
            self._browser = None
            self._listener = None

    def get_discovered_nodes(self) -> list[DiscoveredNode]:
        with self._nodes_lock:
            return list(self._discovered_nodes.values())

    def find_peers(self, timeout: float = 5.0) -> list[DiscoveredNode]:
        """Block until timeout, collecting all discovered peers."""
        if not self._zeroconf:
            self.start()

        found: list[DiscoveredNode] = []
        lock = threading.Lock()

        def on_add(node: DiscoveredNode) -> None:
            with lock:
                found.append(node)

        listener = PoolServiceListener(on_add=on_add)
        browser = ServiceBrowser(self._zeroconf, self._service_type, listener)

        time.sleep(timeout)
        browser.cancel()
        return found

    @property
    def is_registered(self) -> bool:
        return self._service_info is not None

    @property
    def is_discovering(self) -> bool:
        return self._browser is not None
