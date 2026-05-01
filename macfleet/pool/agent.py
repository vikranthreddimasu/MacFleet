"""Pool agent: the daemon that runs on every participating Mac.

Handles: hardware profiling, mDNS registration, peer discovery,
heartbeat gossip, coordinator election, and work assignment.

    macfleet join  →  starts the agent
    macfleet leave →  stops the agent gracefully
"""

from __future__ import annotations

import asyncio
import logging
import secrets as secrets_mod
import socket
import ssl
import subprocess
from typing import Optional

from rich.console import Console

logger = logging.getLogger(__name__)

from macfleet.comm.transport import HardwareExchange
from macfleet.compute.worker import TaskWorker
from macfleet.engines.base import HardwareProfile
from macfleet.monitoring.thermal import get_thermal_state
from macfleet.pool.discovery import DiscoveredNode, ServiceRegistry
from macfleet.pool.heartbeat import GossipHeartbeat, HeartbeatConfig
from macfleet.pool.network import LinkType, get_network_topology
from macfleet.pool.registry import ClusterRegistry, NodeRecord
from macfleet.security.auth import (
    HW_HANDSHAKE_MAX_JSON_BYTES,
    AuthRateLimiter,
    HandshakeHwValidationError,
    SecurityConfig,
    create_client_ssl_context,
    create_server_ssl_context,
    sign_heartbeat,
    sign_heartbeat_with_hw,
    verify_heartbeat,
    verify_heartbeat_with_hw,
)

# v2.2 PR 6 (Issue 22): heartbeat read timeout tightened from 5s to 1s.
# Rationale: a legitimate ping completes in <50ms on WiFi. The old 5s
# timeout meant a burst of 100 attacker connections could stall the
# event loop for 500s worth of pending reads. 1s is still generous for
# a cross-continent round-trip but slams the door on slowloris-style DoS.
HEARTBEAT_READ_TIMEOUT_SEC = 1.0

console = Console()


def _pick_ephemeral_port(exclude: int = 0) -> int:
    """Return a kernel-assigned ephemeral port that's not `exclude`.

    v2.2 post-rc port-flake fix: when PoolAgent is constructed with
    port=0 data_port=0, we bind the heartbeat server first to lock
    `self.port`, then call this to pick `self.data_port`. Rare 1-in-60k
    case of the kernel handing back the same port is handled by a retry.
    """
    for _ in range(16):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        if port != exclude:
            return port
    raise RuntimeError(f"could not pick an ephemeral port distinct from {exclude} after 16 tries")


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
        data_port: Optional[int] = None,
        token: Optional[str] = None,
        fleet_id: Optional[str] = None,
        tls: bool = False,
        peers: Optional[list[str]] = None,
    ):
        # Port split (v2.2 PR 2): heartbeat/discovery on `port` (50051 default),
        # data transport (future PeerTransport from Issue 1a) on `data_port`
        # (50052 default). Must be distinct ports — heartbeat uses line-delimited
        # APING/PONG; transport uses WireMessage binary protocol. Sharing one
        # port means a transport handshake looks like a malformed APING and vice
        # versa. See docs/designs/v3-cathedral.md Issue 5.
        self.port = port
        # v2.2 post-rc: port=0 means "kernel picks ephemeral" for both
        # heartbeat and data_port. We rewrite self.port to the actually-bound
        # value after asyncio.start_server returns in start(). data_port=0
        # gets resolved in _resolve_data_port() via a throwaway bind.
        if port == 0:
            self.data_port = 0 if data_port is None else data_port
        else:
            self.data_port = data_port if data_port is not None else port + 1
        # Collision check only applies to explicit nonzero pairs; (0, 0) is
        # valid (kernel picks two distinct ephemeral ports).
        if self.port != 0 and self.data_port == self.port:
            raise ValueError(
                f"heartbeat port and data port must differ "
                f"(both set to {self.port}). Set --data-port or pick distinct ports."
            )
        self.token = token
        self._security = SecurityConfig(token=token, fleet_id=fleet_id, tls=tls)
        self._manual_peers = peers or []  # ["ip:port", ...]

        # Profiled at start()
        self.hardware: Optional[HardwareProfile] = None
        self._name_override = name

        # Components (pass security config)
        self._discovery = ServiceRegistry(security=self._security)
        self._heartbeat: Optional[GossipHeartbeat] = None
        self._registry: Optional[ClusterRegistry] = None

        self._running = False
        self._task_worker: Optional[TaskWorker] = None
        self._heartbeat_server: Optional[asyncio.Server] = None
        self._heartbeat_ssl_ctx = None
        if self._security.tls:
            self._heartbeat_ssl_ctx = create_server_ssl_context()

        # v2.2 PR 6 (Issue 22): per-IP rate limiter shared with the transport
        # layer pattern — banned IPs get dropped without reading, repeat
        # offenders see exponential backoff. Prevents a rogue node from
        # brute-forcing fleet tokens through the heartbeat port.
        self._heartbeat_rate_limiter = AuthRateLimiter()

    @property
    def node_id(self) -> str:
        if self.hardware:
            return self.hardware.node_id
        return "unknown"

    def _local_hw_exchange(self) -> HardwareExchange:
        """Build a HardwareExchange from the local HardwareProfile.

        Used by the v2.2 manual-peer bootstrap (Issue 6) so the APING/APONG
        round-trip carries real hardware instead of zero-scored placeholders.
        Returns an empty exchange if hardware hasn't been profiled yet.
        """
        hw = self.hardware
        if hw is None:
            return HardwareExchange()
        return HardwareExchange(
            gpu_cores=hw.gpu_cores,
            ram_gb=hw.ram_gb,
            memory_bandwidth_gbps=hw.memory_bandwidth_gbps,
            chip_name=hw.chip_name,
            has_ane=hw.has_ane,
            mps_available=hw.mps_available,
            mlx_available=hw.mlx_available,
            data_port=self.data_port,
        )

    @staticmethod
    def _hw_from_exchange(peer_id: str, peer_hw: HardwareExchange) -> HardwareProfile:
        """Build a HardwareProfile for the registry from a peer's HardwareExchange."""
        return HardwareProfile(
            hostname=peer_id,
            node_id=peer_id,
            gpu_cores=peer_hw.gpu_cores,
            ram_gb=peer_hw.ram_gb,
            memory_bandwidth_gbps=peer_hw.memory_bandwidth_gbps,
            has_ane=peer_hw.has_ane,
            chip_name=peer_hw.chip_name,
            mps_available=peer_hw.mps_available,
            mlx_available=peer_hw.mlx_available,
        )

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
            # Replace both hostname AND node_id so the mDNS service name
            # stays under the RFC 6763 63-byte limit. Some CI / cloud hosts
            # have hostnames like
            # "sat12-bq147-a49fb4f3-b5ae-471f-aa6f-0eed0a917324-167552B41E56.local"
            # which would blow past zeroconf's name validation.
            self.hardware.hostname = self._name_override
            self.hardware.node_id = f"{self._name_override}-{secrets_mod.token_hex(4)}"

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
                console.print("  TLS: [bold green]enabled[/bold green]")

        # 2. Detect network
        topology = get_network_topology()
        link_types_str = ",".join(
            l.link_type.value for l in topology.links if l.link_type != LinkType.LOOPBACK
        )
        console.print(f"  Network: {link_types_str or 'none detected'}")

        # Pick the best IP to advertise
        best = topology.best_link
        ip_address = best.ip_address if best else "127.0.0.1"

        # 3. Start heartbeat responder server FIRST so we know the actual bound
        # port before anything else (registry / mDNS) needs it. When the caller
        # passed port=0, the kernel picks an ephemeral port; we rewrite
        # self.port to reflect reality. reuse_address=True survives TIME_WAIT
        # from a previous clean stop, which is what CI test rapid-fire needs.
        self._heartbeat_server = await asyncio.start_server(
            self._handle_heartbeat_ping, "0.0.0.0", self.port,
            ssl=self._heartbeat_ssl_ctx,
            reuse_address=True,
        )
        bound_port = self._heartbeat_server.sockets[0].getsockname()[1]
        if self.port == 0:
            self.port = bound_port
            # If data_port was also 0, resolve it now via a throwaway bind.
            # The transport layer hasn't bound yet, so we're free to pick an
            # ephemeral port the kernel will hand back (and which won't
            # collide with self.port since each bind to 0 gets a distinct one).
            if self.data_port == 0:
                self.data_port = _pick_ephemeral_port(exclude=self.port)

        # 4. Initialize registry (now that ports are final)
        self._registry = ClusterRegistry(self.hardware.node_id)
        self._registry.register(NodeRecord(
            node_id=self.hardware.node_id,
            hostname=self.hardware.hostname,
            ip_address=ip_address,
            port=self.port,
            data_port=self.data_port,
            hardware=self.hardware,
        ))

        # 5. Register via mDNS (async to avoid EventLoopBlocked)
        await self._discovery.async_register_node(
            hostname=self.hardware.hostname,
            node_id=self.hardware.node_id,
            ip_address=ip_address,
            port=self.port,
            gpu_cores=self.hardware.gpu_cores,
            ram_gb=int(self.hardware.ram_gb),
            chip_name=self.hardware.chip_name,
            link_types=link_types_str,
            compute_score=self.hardware.compute_score,
            data_port=self.data_port,
        )

        # 6. Start discovery
        self._discovery.start_discovery(
            on_add=self._on_peer_discovered,
            on_remove=self._on_peer_removed,
        )

        # 7. Start heartbeat gossip. In secure mode the gossip pings carry
        # signed HW (APING v2) so mDNS-discovered peers refresh from a
        # zero-score placeholder to their real compute_score on the first
        # successful round.
        self._heartbeat = GossipHeartbeat(
            node_id=self.hardware.node_id,
            config=HeartbeatConfig(interval_sec=1.0, suspicion_rounds=3, failure_timeout_sec=10.0),
            security=self._security,
            on_suspected=self._on_peer_suspected,
            on_failed=self._on_peer_failed,
            on_recovered=self._on_peer_recovered,
            local_hw_provider=self._gossip_local_hw_bytes if self._security.is_secure else None,
            on_peer_hw=self._on_peer_hw_received if self._security.is_secure else None,
        )

        await self._heartbeat.start()
        self._running = True

        console.print(
            f"[green]Joined pool[/green] as {self.hardware.node_id} "
            f"on {ip_address} (heartbeat :{self.port}, data :{self.data_port})"
        )

        # 7. Connect to manually specified peers (bypasses mDNS)
        for peer_addr in self._manual_peers:
            await self._add_manual_peer(peer_addr)

        if self._registry.is_coordinator:
            console.print("[bold yellow]This node is the coordinator[/bold yellow]")

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False

        if self._task_worker:
            await self._task_worker.stop()
            self._task_worker = None

        if self._heartbeat:
            await self._heartbeat.stop()
        if self._heartbeat_server:
            self._heartbeat_server.close()
            await self._heartbeat_server.wait_closed()

        await self._discovery.async_stop()
        console.print("[yellow]Left pool[/yellow]")

    async def _handle_heartbeat_ping(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Respond to heartbeat pings from peers.

        v2.1 format (4 fields): `APING {node_id} {nonce_hex} {sig_hex}`
        v2.2 format (5 fields): `APING {node_id} {nonce_hex} {sig_hex} {hw_json_hex}`

        The 5-field variant is used by `--peer` manual-peer bootstrap (Issue 6)
        so manual peers can exchange real hardware info without mDNS TXT records.
        Server replies in matching format: 4-in → APONG 4-out, 5-in → APONG 5-out.

        v2.2 PR 6 (Issue 22): per-IP rate limiting + tightened 1s read timeout.
        Each failed auth increments a per-IP counter; 5 failures in a row earns
        a 5-minute ban. A banned IP is disconnected before we even read.
        """
        fleet_key = self._security.fleet_key

        # Per-IP rate limiting: reject banned IPs before any I/O.
        peername = writer.get_extra_info("peername")
        peer_ip = peername[0] if peername else "unknown"
        if self._heartbeat_rate_limiter.is_banned(peer_ip):
            logger.warning("Heartbeat: rate-limited banned IP %s", peer_ip)
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ssl.SSLError, BrokenPipeError, ConnectionResetError):
                pass
            return

        # Exponential backoff for IPs with recent failures — slows down
        # brute-force attempts without rejecting outright.
        delay = self._heartbeat_rate_limiter.get_delay(peer_ip)
        if delay > 0:
            await asyncio.sleep(delay)

        try:
            data = await asyncio.wait_for(
                reader.readline(), timeout=HEARTBEAT_READ_TIMEOUT_SEC,
            )

            if fleet_key and data.startswith(b"APING"):
                # Authenticated heartbeat
                parts = data.decode().strip().split(" ")
                if len(parts) == 4:
                    _, peer_node_id, nonce_hex, sig_hex = parts
                    try:
                        nonce = bytes.fromhex(nonce_hex)
                        sig = bytes.fromhex(sig_hex)
                    except ValueError:
                        self._heartbeat_rate_limiter.record_failure(peer_ip)
                        logger.debug("APING v1 malformed hex from %s", peer_ip)
                        return
                    if verify_heartbeat(fleet_key, peer_node_id, nonce, sig):
                        self._heartbeat_rate_limiter.record_success(peer_ip)
                        # Send authenticated PONG (v2.1 legacy format)
                        resp_nonce = secrets_mod.token_bytes(16)
                        resp_sig = sign_heartbeat(fleet_key, self.node_id, resp_nonce)
                        writer.write(
                            f"APONG {self.node_id} {resp_nonce.hex()} {resp_sig.hex()}\n".encode()
                        )
                        await writer.drain()
                    else:
                        self._heartbeat_rate_limiter.record_failure(peer_ip)
                        logger.debug("Heartbeat auth failed from peer %s", peer_node_id)
                elif len(parts) == 5:
                    # v2.2: carries peer HW profile
                    _, peer_node_id, nonce_hex, sig_hex, hw_hex = parts
                    try:
                        nonce = bytes.fromhex(nonce_hex)
                        sig = bytes.fromhex(sig_hex)
                        hw_json = bytes.fromhex(hw_hex)
                    except ValueError:
                        self._heartbeat_rate_limiter.record_failure(peer_ip)
                        logger.debug("APING v2 from %s: malformed hex", peer_node_id)
                        return
                    if len(hw_json) > HW_HANDSHAKE_MAX_JSON_BYTES:
                        self._heartbeat_rate_limiter.record_failure(peer_ip)
                        logger.debug(
                            "APING v2 from %s: HW payload %dB exceeds max %dB",
                            peer_node_id, len(hw_json), HW_HANDSHAKE_MAX_JSON_BYTES,
                        )
                        return
                    if not verify_heartbeat_with_hw(
                        fleet_key, peer_node_id, nonce, hw_json, sig,
                    ):
                        self._heartbeat_rate_limiter.record_failure(peer_ip)
                        logger.debug("APING v2 auth failed from peer %s", peer_node_id)
                        return
                    self._heartbeat_rate_limiter.record_success(peer_ip)
                    # Reply with APONG v2 (5-field) carrying local HW
                    resp_nonce = secrets_mod.token_bytes(16)
                    try:
                        local_hw_json = self._local_hw_exchange().to_json_bytes()
                    except Exception as e:
                        logger.debug("APING v2 reply: local HW serialization failed: %s", e)
                        return
                    resp_sig = sign_heartbeat_with_hw(
                        fleet_key, self.node_id, resp_nonce, local_hw_json,
                    )
                    writer.write(
                        (
                            f"APONG {self.node_id} {resp_nonce.hex()} "
                            f"{resp_sig.hex()} {local_hw_json.hex()}\n"
                        ).encode()
                    )
                    await writer.drain()
                else:
                    self._heartbeat_rate_limiter.record_failure(peer_ip)
                    logger.debug("APING malformed (%d fields) from %s", len(parts), peer_ip)
            elif not fleet_key and data.startswith(b"PING"):
                # Open heartbeat (backward compatible)
                writer.write(f"PONG {self.node_id}\n".encode())
                await writer.drain()
            elif fleet_key:
                # Secure server got a non-APING message → treat as a failed
                # auth attempt. Plain PINGs against an auth'd fleet are
                # either a misconfigured client or an attacker probing.
                self._heartbeat_rate_limiter.record_failure(peer_ip)
                logger.debug(
                    "Heartbeat from %s: secure server received non-APING (%r)",
                    peer_ip, data[:32],
                )
            # Else: open server got a non-PING message — silently ignore (no auth loss)

        except asyncio.TimeoutError:
            # Slow/stalled connection — treat as a failure to deter slowloris.
            self._heartbeat_rate_limiter.record_failure(peer_ip)
        except (ConnectionResetError, ValueError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ssl.SSLError, BrokenPipeError, ConnectionResetError):
                pass

    async def _add_manual_peer(self, peer_addr: str) -> None:
        """Connect to a manually specified peer (bypasses mDNS).

        Used when mDNS is blocked (e.g. enterprise WiFi with client isolation).
        Sends a heartbeat ping to verify the peer is reachable and running.

        v2.2 Issue 6: when authenticated, sends APING v2 (5-field with HW payload)
        and parses APONG v2 to extract the peer's real HardwareProfile + data_port.
        If the peer responds with APONG v1 (4-field), falls back to zero-HW
        placeholder for v2.1 compat.
        """
        writer: Optional[asyncio.StreamWriter] = None
        try:
            if ":" in peer_addr:
                host, port_str = peer_addr.rsplit(":", 1)
                port = int(port_str)
            else:
                host = peer_addr
                port = self.port  # default to same port

            # Ping the peer to verify it's alive and get its node_id
            fleet_key = self._security.fleet_key
            ssl_ctx = create_client_ssl_context() if self._security.tls else None

            console.print(f"[dim]Connecting to peer {host}:{port}...[/dim]")

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, ssl=ssl_ctx),
                timeout=10.0,
            )

            local_hw_json: Optional[bytes] = None
            if fleet_key:
                # v2.2: send APING v2 with local HW so peer can register us with
                # real compute_score + data_port on their side. Peer responds with
                # APONG v2 (or APONG v1 if older).
                nonce = secrets_mod.token_bytes(16)
                local_hw_json = self._local_hw_exchange().to_json_bytes()
                sig = sign_heartbeat_with_hw(
                    fleet_key, self.node_id, nonce, local_hw_json,
                )
                writer.write(
                    (
                        f"APING {self.node_id} {nonce.hex()} "
                        f"{sig.hex()} {local_hw_json.hex()}\n"
                    ).encode()
                )
                await writer.drain()
                response = await asyncio.wait_for(reader.readline(), timeout=10.0)
            else:
                writer.write(f"PING {self.node_id}\n".encode())
                await writer.drain()
                response = await asyncio.wait_for(reader.readline(), timeout=10.0)

            peer_hw: Optional[HardwareExchange] = None
            peer_data_port = port + 1

            # Parse response
            if fleet_key:
                if not response.startswith(b"APONG"):
                    console.print(
                        f"[red]Peer {peer_addr}: no authenticated response "
                        f"(got: {response[:50]!r})[/red]"
                    )
                    return
                parts = response.decode().strip().split(" ")
                if len(parts) == 5:
                    # APONG v2: peer returned signed HW
                    _, peer_node_id, resp_nonce_hex, resp_sig_hex, peer_hw_hex = parts
                    try:
                        resp_nonce = bytes.fromhex(resp_nonce_hex)
                        resp_sig = bytes.fromhex(resp_sig_hex)
                        peer_hw_json = bytes.fromhex(peer_hw_hex)
                    except ValueError:
                        console.print(f"[red]Peer {peer_addr}: malformed hex in APONG v2[/red]")
                        return
                    if len(peer_hw_json) > HW_HANDSHAKE_MAX_JSON_BYTES:
                        console.print(
                            f"[red]Peer {peer_addr}: HW payload "
                            f"{len(peer_hw_json)}B exceeds max[/red]"
                        )
                        return
                    if not verify_heartbeat_with_hw(
                        fleet_key, peer_node_id, resp_nonce, peer_hw_json, resp_sig,
                    ):
                        console.print(
                            f"[red]Peer {peer_addr}: authentication failed "
                            f"(wrong token?)[/red]"
                        )
                        return
                    try:
                        peer_hw = HardwareExchange.from_json_bytes(peer_hw_json)
                    except HandshakeHwValidationError as e:
                        console.print(f"[red]Peer {peer_addr}: bad HW payload: {e}[/red]")
                        return
                    if peer_hw.data_port > 0:
                        peer_data_port = peer_hw.data_port
                elif len(parts) == 4:
                    # APONG v1: v2.1 peer — fall back to zero-HW placeholder
                    _, peer_node_id, resp_nonce_hex, resp_sig_hex = parts
                    if not verify_heartbeat(
                        fleet_key, peer_node_id,
                        bytes.fromhex(resp_nonce_hex), bytes.fromhex(resp_sig_hex),
                    ):
                        console.print(
                            f"[red]Peer {peer_addr}: authentication failed "
                            f"(wrong token?)[/red]"
                        )
                        return
                    logger.info(
                        "Peer %s responded with APONG v1 (no HW) — likely v2.1. "
                        "Registering with zero compute_score.", peer_node_id,
                    )
                else:
                    console.print(
                        f"[red]Peer {peer_addr}: malformed response "
                        f"({len(parts)} fields)[/red]"
                    )
                    return
            else:
                if not response.startswith(b"PONG"):
                    console.print(f"[red]Peer {peer_addr}: no response (got: {response[:50]!r})[/red]")
                    return
                parts = response.decode().strip().split(" ")
                peer_node_id = parts[1] if len(parts) >= 2 else f"peer-{host}"

            # Register the peer. With v2.2 APONG we have real HW; without it we
            # fall back to a zero-score placeholder so the peer still shows up in
            # the registry (just loses coordinator election until discovered via
            # mDNS or re-pinged).
            if peer_hw is not None:
                hw = self._hw_from_exchange(peer_node_id, peer_hw)
            else:
                hw = HardwareProfile(
                    hostname=peer_node_id,
                    node_id=peer_node_id,
                    gpu_cores=0,
                    ram_gb=0.0,
                    memory_bandwidth_gbps=0.0,
                    has_ane=True,
                    chip_name="unknown (manual peer)",
                )
            self._registry.register(NodeRecord(
                node_id=peer_node_id,
                hostname=peer_node_id,
                ip_address=host,
                port=port,
                data_port=peer_data_port,
                hardware=hw,
            ))
            self._heartbeat.add_peer(peer_node_id, host, port, hw.compute_score)

            hw_label = (
                f"{hw.chip_name}, {hw.gpu_cores} GPU cores, {hw.ram_gb:.0f} GB"
                if peer_hw is not None
                else "no HW info"
            )
            console.print(
                f"[cyan]Connected to peer[/cyan] {peer_node_id} at {host} "
                f"(heartbeat :{port}, data :{peer_data_port}, {hw_label})"
            )

        except (OSError, ssl.SSLError, asyncio.TimeoutError, ConnectionRefusedError, ValueError) as e:
            console.print(f"[red]Failed to connect to peer {peer_addr}: {type(e).__name__}: {e}[/red]")
            console.print("[dim]Make sure the peer is running 'macfleet join' and is reachable[/dim]")
        finally:
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except (OSError, ssl.SSLError, BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                    pass

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
            data_port=node.data_port,
            hardware=hw,
        ))

        # Use locally-computed score, not the broadcast one
        self._heartbeat.add_peer(node.node_id, node.ip_address, node.port, hw.compute_score)

        console.print(f"[cyan]Discovered[/cyan] {node.hostname} ({node.chip_name}, {node.gpu_cores} GPU cores)")

        if self._registry.is_coordinator:
            console.print(
                f"[bold yellow]This node is the coordinator[/bold yellow] "
                f"(world_size={self._registry.world_size})"
            )

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

    def _gossip_local_hw_bytes(self) -> Optional[bytes]:
        """Return the local HW exchange JSON for gossip APING v2, or None."""
        if self.hardware is None:
            return None
        try:
            return self._local_hw_exchange().to_json_bytes()
        except Exception as e:
            logger.debug("gossip local HW serialization failed: %s", e)
            return None

    def _on_peer_hw_received(self, peer_node_id: str, peer_hw_json: bytes) -> None:
        """Refresh registry HW for a peer using HW JSON received via APONG v2.

        Closes the secure-mode coordinator-election gap: mDNS broadcasts
        elide hardware fields, so peers initially register with
        compute_score=0. After one successful APONG v2, this callback
        replaces the placeholder profile with the peer's real specs.
        """
        if self._registry is None:
            return
        try:
            peer_hw = HardwareExchange.from_json_bytes(peer_hw_json)
        except HandshakeHwValidationError as e:
            logger.debug("APONG v2 HW from %s: parse failed: %s", peer_node_id, e)
            return
        record = self._registry.get_node(peer_node_id)
        if record is None:
            return
        new_hw = self._hw_from_exchange(peer_node_id, peer_hw)
        # Preserve thermal pressure (registry is authoritative on liveness state).
        new_hw.thermal_pressure = record.hardware.thermal_pressure
        record.hardware = new_hw
        if peer_hw.data_port > 0:
            record.data_port = peer_hw.data_port
        # Re-elect coordinator with the refreshed compute score.
        self._registry._elect_coordinator()
