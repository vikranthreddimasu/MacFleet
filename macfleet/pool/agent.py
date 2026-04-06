"""Pool agent: the daemon that runs on every participating Mac.

Handles: hardware profiling, mDNS registration, peer discovery,
heartbeat gossip, coordinator election, and work assignment.

    macfleet join  →  starts the agent
    macfleet leave →  stops the agent gracefully
"""

from __future__ import annotations

import asyncio
import logging
import platform
import secrets as secrets_mod
import socket
import subprocess
from typing import Optional

from rich.console import Console

logger = logging.getLogger(__name__)

from macfleet.engines.base import HardwareProfile, ThermalPressure
from macfleet.monitoring.thermal import get_thermal_state
from macfleet.pool.discovery import ServiceRegistry, DiscoveredNode
from macfleet.pool.heartbeat import GossipHeartbeat, HeartbeatConfig
from macfleet.pool.network import LinkType, detect_interfaces, get_network_topology
from macfleet.pool.registry import ClusterRegistry, NodeRecord
from macfleet.security.auth import (
    SecurityConfig,
    create_client_ssl_context,
    create_server_ssl_context,
    sign_heartbeat,
    verify_heartbeat,
)

console = Console()


def _detect_chip_name() -> str:
    """Detect the Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "Apple Silicon"


def _detect_gpu_cores() -> int:
    """Detect GPU core count on Apple Silicon."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Total Number of Cores" in line:
                    try:
                        return int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 8  # safe default for Apple Silicon


def _detect_ram_gb() -> float:
    """Detect total RAM in GB."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return 8.0


def _detect_memory_bandwidth() -> float:
    """Estimate memory bandwidth in GB/s based on chip."""
    chip = _detect_chip_name().lower()
    # Rough estimates based on known Apple Silicon specs
    if "ultra" in chip:
        return 800.0
    elif "max" in chip:
        return 400.0
    elif "pro" in chip:
        return 200.0
    else:
        return 100.0  # base M-series


def _check_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) backend is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def _check_mlx_available() -> bool:
    """Check if MLX is installed and functional."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def profile_hardware() -> HardwareProfile:
    """Profile the local Mac's hardware capabilities."""
    hostname = socket.gethostname()
    node_id = f"{hostname}-{secrets_mod.token_hex(4)}"

    return HardwareProfile(
        hostname=hostname,
        node_id=node_id,
        gpu_cores=_detect_gpu_cores(),
        ram_gb=_detect_ram_gb(),
        memory_bandwidth_gbps=_detect_memory_bandwidth(),
        has_ane=True,  # All Apple Silicon has ANE
        chip_name=_detect_chip_name(),
        thermal_pressure=get_thermal_state().pressure,
        mps_available=_check_mps_available(),
        mlx_available=_check_mlx_available(),
    )


class PoolAgent:
    """The main agent daemon that manages pool membership.

    Lifecycle:
        agent = PoolAgent()
        await agent.start()   # profile, register, discover, heartbeat
        ...                    # training happens via pool
        await agent.stop()    # graceful departure
    """

    def __init__(
        self,
        name: Optional[str] = None,
        port: int = 50051,
        token: Optional[str] = None,
        fleet_id: Optional[str] = None,
        tls: bool = False,
    ):
        self.port = port
        self.token = token
        self._security = SecurityConfig(token=token, fleet_id=fleet_id, tls=tls)

        # Profiled at start()
        self.hardware: Optional[HardwareProfile] = None
        self._name_override = name

        # Components (pass security config)
        self._discovery = ServiceRegistry(security=self._security)
        self._heartbeat: Optional[GossipHeartbeat] = None
        self._registry: Optional[ClusterRegistry] = None

        self._running = False
        self._heartbeat_server: Optional[asyncio.Server] = None
        self._heartbeat_ssl_ctx = None
        if self._security.tls:
            self._heartbeat_ssl_ctx = create_server_ssl_context()

    @property
    def node_id(self) -> str:
        if self.hardware:
            return self.hardware.node_id
        return "unknown"

    @property
    def registry(self) -> Optional[ClusterRegistry]:
        return self._registry

    @property
    def is_coordinator(self) -> bool:
        return self._registry.is_coordinator if self._registry else False

    async def start(self) -> None:
        """Start the pool agent: profile, register, discover, heartbeat."""
        # 1. Profile hardware
        self.hardware = profile_hardware()
        if self._name_override:
            self.hardware.hostname = self._name_override

        console.print(f"[bold blue]MacFleet[/bold blue] agent starting on {self.hardware.hostname}")
        console.print(f"  Chip: {self.hardware.chip_name}")
        console.print(f"  GPU cores: {self.hardware.gpu_cores}")
        console.print(f"  RAM: {self.hardware.ram_gb:.0f} GB")
        console.print(f"  MPS: {'yes' if self.hardware.mps_available else 'no'}")
        console.print(f"  MLX: {'yes' if self.hardware.mlx_available else 'no'}")
        if self._security.is_secure:
            fleet_label = self._security.fleet_id or "default"
            console.print(f"  Fleet: {fleet_label} [bold green](token-protected)[/bold green]")
            if self._security.tls:
                console.print(f"  TLS: [bold green]enabled[/bold green]")

        # 2. Detect network
        topology = get_network_topology()
        link_types_str = ",".join(
            l.link_type.value for l in topology.links if l.link_type != LinkType.LOOPBACK
        )
        console.print(f"  Network: {link_types_str or 'none detected'}")

        # Pick the best IP to advertise
        best = topology.best_link
        ip_address = best.ip_address if best else "127.0.0.1"

        # 3. Initialize registry
        self._registry = ClusterRegistry(self.hardware.node_id)

        # Register self
        self._registry.register(NodeRecord(
            node_id=self.hardware.node_id,
            hostname=self.hardware.hostname,
            ip_address=ip_address,
            port=self.port,
            hardware=self.hardware,
        ))

        # 4. Register via mDNS
        self._discovery.register_node(
            hostname=self.hardware.hostname,
            node_id=self.hardware.node_id,
            ip_address=ip_address,
            port=self.port,
            gpu_cores=self.hardware.gpu_cores,
            ram_gb=int(self.hardware.ram_gb),
            chip_name=self.hardware.chip_name,
            link_types=link_types_str,
            compute_score=self.hardware.compute_score,
        )

        # 5. Start discovery
        self._discovery.start_discovery(
            on_add=self._on_peer_discovered,
            on_remove=self._on_peer_removed,
        )

        # 6. Start heartbeat
        self._heartbeat = GossipHeartbeat(
            node_id=self.hardware.node_id,
            config=HeartbeatConfig(interval_sec=1.0, suspicion_rounds=3, failure_timeout_sec=10.0),
            security=self._security,
            on_suspected=self._on_peer_suspected,
            on_failed=self._on_peer_failed,
            on_recovered=self._on_peer_recovered,
        )

        # Start heartbeat responder server (TLS when auth is enabled)
        self._heartbeat_server = await asyncio.start_server(
            self._handle_heartbeat_ping, "0.0.0.0", self.port,
            ssl=self._heartbeat_ssl_ctx,
        )

        await self._heartbeat.start()
        self._running = True

        console.print(f"[green]Joined pool[/green] as {self.hardware.node_id} on {ip_address}:{self.port}")
        if self._registry.is_coordinator:
            console.print("[bold yellow]This node is the coordinator[/bold yellow]")

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False

        if self._heartbeat:
            await self._heartbeat.stop()
        if self._heartbeat_server:
            self._heartbeat_server.close()
            await self._heartbeat_server.wait_closed()

        self._discovery.stop()
        console.print(f"[yellow]Left pool[/yellow]")

    async def _handle_heartbeat_ping(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Respond to heartbeat pings from peers."""
        fleet_key = self._security.fleet_key
        try:
            data = await asyncio.wait_for(reader.readline(), timeout=5.0)

            if fleet_key and data.startswith(b"APING"):
                # Authenticated heartbeat
                parts = data.decode().strip().split(" ")
                if len(parts) == 4:
                    _, peer_node_id, nonce_hex, sig_hex = parts
                    nonce = bytes.fromhex(nonce_hex)
                    sig = bytes.fromhex(sig_hex)
                    if verify_heartbeat(fleet_key, peer_node_id, nonce, sig):
                        # Send authenticated PONG
                        resp_nonce = secrets_mod.token_bytes(16)
                        resp_sig = sign_heartbeat(fleet_key, self.node_id, resp_nonce)
                        writer.write(
                            f"APONG {self.node_id} {resp_nonce.hex()} {resp_sig.hex()}\n".encode()
                        )
                        await writer.drain()
                    else:
                        logger.debug("Heartbeat auth failed from peer %s", peer_node_id)
            elif not fleet_key and data.startswith(b"PING"):
                # Open heartbeat (backward compatible)
                writer.write(f"PONG {self.node_id}\n".encode())
                await writer.drain()
            # If secure but received plain PING, or open but received APING: silently ignore

        except (asyncio.TimeoutError, ConnectionResetError, ValueError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    def _on_peer_discovered(self, node: DiscoveredNode) -> None:
        """Called when a new peer is discovered via mDNS."""
        if node.node_id == self.node_id:
            return  # Ignore self

        # SECURITY: Create HardwareProfile from discovery data and compute
        # score locally from reported hardware specs. Never trust the
        # broadcast compute_score — a rogue node could inflate it to
        # win coordinator election.
        hw = HardwareProfile(
            hostname=node.hostname,
            node_id=node.node_id,
            gpu_cores=node.gpu_cores,
            ram_gb=float(node.ram_gb),
            memory_bandwidth_gbps=0.0,  # Unknown until profiled
            has_ane=True,
            chip_name=node.chip_name,
        )

        self._registry.register(NodeRecord(
            node_id=node.node_id,
            hostname=node.hostname,
            ip_address=node.ip_address,
            port=node.port,
            hardware=hw,
        ))

        # Use locally-computed score, not the broadcast one
        self._heartbeat.add_peer(node.node_id, node.ip_address, node.port, hw.compute_score)

        console.print(f"[cyan]Discovered[/cyan] {node.hostname} ({node.chip_name}, {node.gpu_cores} GPU cores)")

        if self._registry.is_coordinator:
            console.print(f"[bold yellow]This node is the coordinator[/bold yellow] (world_size={self._registry.world_size})")

    def _on_peer_removed(self, hostname: str) -> None:
        """Called when a peer's mDNS registration expires."""
        console.print(f"[yellow]Peer removed:[/yellow] {hostname}")

    def _on_peer_suspected(self, node_id: str) -> None:
        """Called when heartbeat suspects a peer is down."""
        console.print(f"[yellow]Peer suspected:[/yellow] {node_id}")

    def _on_peer_failed(self, node_id: str) -> None:
        """Called when a peer is confirmed failed."""
        self._registry.mark_failed(node_id)
        console.print(f"[red]Peer failed:[/red] {node_id} (world_size={self._registry.world_size})")

    def _on_peer_recovered(self, node_id: str) -> None:
        """Called when a suspected peer recovers."""
        self._registry.mark_alive(node_id)
        console.print(f"[green]Peer recovered:[/green] {node_id}")
