"""Bonjour/zeroconf service discovery for MacFleet v2.

Ported from v1 comm/discovery.py with extended properties for pool metadata:
- node_id, chip_name, link_types, pool_version added to mDNS properties
- Removed master/worker role distinction (v2 uses peer model with elected coordinator)
"""

import asyncio
import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from zeroconf import ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf
from zeroconf.asyncio import AsyncZeroconf

import macfleet
from macfleet.pool.network import NetworkLink
from macfleet.pool.topology import deserialize_network_links, serialize_network_links
from macfleet.security.auth import DEFAULT_SERVICE_TYPE, SecurityConfig

MACFLEET_SERVICE_TYPE = DEFAULT_SERVICE_TYPE
DEFAULT_TTL = 120


@dataclass
class DiscoveredNode:
    """A node discovered via mDNS.

    v2.2 PR 2 introduced a port split: `port` is the heartbeat/discovery port
    (default 50051), `data_port` is the training transport port (default 50052).
    `port` remains the mDNS ServiceInfo.port for backward compat with 2.1.x
    peers. `data_port` rides in a TXT property and falls back to `port + 1`
    when a 2.1.x peer is discovered without advertising it.
    """
    hostname: str
    node_id: str
    ip_address: str
    port: int  # heartbeat / discovery / control (default 50051)
    gpu_cores: int
    ram_gb: int
    chip_name: str
    link_types: str  # comma-separated: "wifi,ethernet,thunderbolt"
    pool_version: str
    compute_score: float = 0.0
    data_port: int = 0  # training transport port (default 50052, 0 = not advertised)
    network_links: tuple[NetworkLink, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not (0 < self.port < 65536):
            raise ValueError("port must be between 1 and 65535")
        if (
            isinstance(self.compute_score, bool)
            or not isinstance(self.compute_score, (int, float))
            or not math.isfinite(float(self.compute_score))
            or self.compute_score < 0
        ):
            raise ValueError("compute_score must be finite and non-negative")
        # Backward compat: 2.1.x peers don't advertise data_port. Fall back to
        # heartbeat_port + 1 — matches the 50051/50052 convention.
        if self.data_port == 0:
            self.data_port = self.port + 1
        if (
            isinstance(self.data_port, bool)
            or not isinstance(self.data_port, int)
            or not (0 < self.data_port < 65536)
        ):
            raise ValueError("data_port must be between 1 and 65535")

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
        loop = getattr(zc, "loop", None)
        if loop is not None and loop.is_running():
            # When AsyncZeroconf attaches to a running event loop (e.g. inside
            # an async Pool), the synchronous get_service_info() submits a
            # coroutine via run_coroutine_threadsafe and then blocks on
            # .result(timeout) from the ServiceBrowser thread.  If the loop is
            # busy the future times out and raises EventLoopBlocked, silently
            # dropping the peer.  Use the async variant instead and let the
            # loop process it without blocking the browser thread.
            on_add = self._on_add
            parse = self._parse_service_info

            async def _async_add() -> None:
                node_info = await zc.async_get_service_info(service_type, name)
                if node_info and on_add:
                    node = parse(node_info)
                    if node:
                        on_add(node)

            asyncio.run_coroutine_threadsafe(_async_add(), loop)
            return
        info = zc.get_service_info(service_type, name)
        if info and self._on_add:
            node = self._parse_service_info(info)
            if node:
                self._on_add(node)

    def remove_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        if self._on_remove:
            # Service instance name is "{node_id}.{service_type}".
            suffix = f".{service_type}"
            node_id = name[: -len(suffix)] if name.endswith(suffix) else name
            self._on_remove(node_id)

    def update_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        loop = getattr(zc, "loop", None)
        if loop is not None and loop.is_running():
            on_update = self._on_update
            parse = self._parse_service_info

            async def _async_update() -> None:
                node_info = await zc.async_get_service_info(service_type, name)
                if node_info and on_update:
                    node = parse(node_info)
                    if node:
                        on_update(node)

            asyncio.run_coroutine_threadsafe(_async_update(), loop)
            return
        info = zc.get_service_info(service_type, name)
        if info and self._on_update:
            node = self._parse_service_info(info)
            if node:
                self._on_update(node)

    def _parse_service_info(self, info: ServiceInfo) -> Optional[DiscoveredNode]:
        try:
            if not info.addresses:
                return None
            # Pick the first parseable address. inet_ntoa rejects IPv6
            # (16-byte) entries — fall through to inet_ntop. Skip any
            # malformed address rather than failing the whole record.
            ip_address: Optional[str] = None
            for raw in info.addresses:
                if len(raw) == 4:
                    ip_address = socket.inet_ntop(socket.AF_INET, raw)
                    break
                if len(raw) == 16:
                    ip_address = socket.inet_ntop(socket.AF_INET6, raw)
                    break
            if ip_address is None:
                return None

            if info.server is None:
                return None

            props = info.properties
            hostname = info.server.rstrip(".")

            def _prop(key: bytes, default: bytes) -> bytes:
                val = props.get(key)
                return val if isinstance(val, bytes) else default

            node_id = _prop(b"node_id", b"").decode() or hostname
            gpu_cores = int(_prop(b"gpu_cores", b"0").decode())
            ram_gb = int(_prop(b"ram_gb", b"0").decode())
            chip_name = _prop(b"chip_name", b"unknown").decode()
            link_types = _prop(b"link_types", b"").decode()
            pool_version = _prop(b"pool_version", b"0.0.0").decode()
            compute_score = float(_prop(b"compute_score", b"0").decode())
            # data_port advertised since v2.2; 2.1.x peers lack this. DiscoveredNode
            # __post_init__ falls back to heartbeat port + 1 when 0.
            data_port = int(_prop(b"data_port", b"0").decode())
            network_links = deserialize_network_links(
                _prop(b"network_links", b"").decode()
            )

            if info.port is None:
                return None

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
                data_port=data_port,
                network_links=network_links,
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
            unregister_broadcast = await self._async_zeroconf.async_unregister_service(
                self._service_info
            )
            if unregister_broadcast is not None:
                await unregister_broadcast
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
        data_port: int,
        network_links: tuple[NetworkLink, ...] = (),
    ) -> dict[bytes, bytes]:
        """Build mDNS service properties.

        SECURITY: When fleet is token-protected, minimize broadcast info.
        Only broadcast node_id, data_port, and version. Hardware details
        (GPU cores, RAM, chip) are exchanged AFTER authenticated connection.

        data_port is always broadcast (even in secure mode) because it's not
        sensitive — it's just the TCP port the transport listens on. Peers
        need it to initiate the authenticated handshake.
        """
        if self._security.is_secure:
            return {
                b"node_id": node_id.encode(),
                b"pool_version": macfleet.__version__.encode(),
                b"data_port": str(data_port).encode(),
            }
        return {
            b"node_id": node_id.encode(),
            b"gpu_cores": str(gpu_cores).encode(),
            b"ram_gb": str(ram_gb).encode(),
            b"chip_name": chip_name.encode(),
            b"link_types": link_types.encode(),
            b"network_links": serialize_network_links(network_links).encode(),
            b"pool_version": macfleet.__version__.encode(),
            b"compute_score": f"{compute_score:.1f}".encode(),
            b"data_port": str(data_port).encode(),
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
        data_port: int = 0,
        network_links: tuple[NetworkLink, ...] = (),
    ) -> None:
        """Register this node in the pool via mDNS.

        `port` is the heartbeat/discovery port (default 50051).
        `data_port` is the training transport port (default port + 1).
        """
        if not self._zeroconf:
            self.start()
        if data_port == 0:
            data_port = port + 1

        service_name = f"{node_id}.{self._service_type}"
        properties = self._build_properties(
            node_id, gpu_cores, ram_gb, chip_name, link_types, compute_score,
            data_port, network_links,
        )

        self._service_info = ServiceInfo(
            self._service_type,
            service_name,
            addresses=[socket.inet_aton(ip_address)],
            port=port,
            properties=properties,
            server=f"{hostname}.local.",
        )
        self._zeroconf.register_service(self._service_info, ttl=DEFAULT_TTL)  # type: ignore[union-attr]

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
        data_port: int = 0,
        network_links: tuple[NetworkLink, ...] = (),
    ) -> None:
        """Register this node in the pool via mDNS (async).

        `port` is the heartbeat/discovery port (default 50051).
        `data_port` is the training transport port (default port + 1).
        """
        if not self._async_zeroconf:
            await self.async_start()
        if data_port == 0:
            data_port = port + 1

        service_name = f"{node_id}.{self._service_type}"
        properties = self._build_properties(
            node_id, gpu_cores, ram_gb, chip_name, link_types, compute_score,
            data_port, network_links,
        )

        self._service_info = ServiceInfo(
            self._service_type,
            service_name,
            addresses=[socket.inet_aton(ip_address)],
            port=port,
            properties=properties,
            server=f"{hostname}.local.",
        )
        await self._async_zeroconf.async_register_service(  # type: ignore[union-attr]
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

        track_add, track_remove, track_update = self._tracking_callbacks(
            on_add=on_add, on_remove=on_remove, on_update=on_update,
        )
        self._listener = PoolServiceListener(
            on_add=track_add, on_remove=track_remove, on_update=track_update,
        )
        assert self._zeroconf is not None
        self._browser = ServiceBrowser(
            self._zeroconf, self._service_type, self._listener,
        )

    def _tracking_callbacks(
        self,
        on_add: Optional[Callable[[DiscoveredNode], None]] = None,
        on_remove: Optional[Callable[[str], None]] = None,
        on_update: Optional[Callable[[DiscoveredNode], None]] = None,
    ) -> tuple[
        Callable[[DiscoveredNode], None],
        Callable[[str], None],
        Callable[[DiscoveredNode], None],
    ]:
        """Wrap callbacks so every discovery path maintains the node cache."""
        def track_add(node: DiscoveredNode) -> None:
            with self._nodes_lock:
                self._discovered_nodes[node.node_id] = node
            if on_add:
                on_add(node)

        def track_remove(node_id: str) -> None:
            with self._nodes_lock:
                removed = self._discovered_nodes.pop(node_id, None)
                if removed is None:
                    # Fallback when TXT node_id differed from the mDNS
                    # instance name used at registration time.
                    stale = [
                        nid
                        for nid, node in self._discovered_nodes.items()
                        if node.hostname == node_id or nid == node_id
                    ]
                    for nid in stale:
                        self._discovered_nodes.pop(nid, None)
            if on_remove:
                on_remove(node_id)

        def track_update(node: DiscoveredNode) -> None:
            with self._nodes_lock:
                self._discovered_nodes[node.node_id] = node
            if on_update:
                on_update(node)

        return track_add, track_remove, track_update

    def stop_discovery(self) -> None:
        if self._browser:
            self._browser.cancel()
            self._browser = None
            self._listener = None

    def get_discovered_nodes(self) -> list[DiscoveredNode]:
        with self._nodes_lock:
            return list(self._discovered_nodes.values())

    def find_peers(self, timeout: float = 5.0) -> list[DiscoveredNode]:
        """Block until timeout, collecting all discovered peers.

        WARNING: this is synchronous (uses time.sleep). Calling from an
        async context blocks the event loop for the full timeout. Use
        async_find_peers() instead from coroutines.
        """
        if not self._zeroconf:
            self.start()

        found: dict[str, DiscoveredNode] = {}
        lock = threading.Lock()

        def remember(node: DiscoveredNode) -> None:
            with lock:
                found[node.node_id] = node

        track_add, _, track_update = self._tracking_callbacks(
            on_add=remember, on_update=remember,
        )
        listener = PoolServiceListener(on_add=track_add, on_update=track_update)
        assert self._zeroconf is not None
        browser = ServiceBrowser(self._zeroconf, self._service_type, listener)

        time.sleep(timeout)
        browser.cancel()
        with lock:
            return list(found.values())

    async def async_find_peers(self, timeout: float = 5.0) -> list[DiscoveredNode]:
        """Async variant of find_peers — yields the loop while waiting."""
        if not self._zeroconf:
            await self.async_start()

        found: dict[str, DiscoveredNode] = {}
        lock = threading.Lock()

        def remember(node: DiscoveredNode) -> None:
            with lock:
                found[node.node_id] = node

        track_add, _, track_update = self._tracking_callbacks(
            on_add=remember, on_update=remember,
        )
        listener = PoolServiceListener(on_add=track_add, on_update=track_update)
        assert self._zeroconf is not None
        browser = ServiceBrowser(self._zeroconf, self._service_type, listener)

        try:
            await asyncio.sleep(timeout)
        finally:
            browser.cancel()
        with lock:
            return list(found.values())

    @property
    def is_registered(self) -> bool:
        return self._service_info is not None

    @property
    def is_discovering(self) -> bool:
        return self._browser is not None
