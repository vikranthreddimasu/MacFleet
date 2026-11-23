"""Pool agent: the daemon that runs on every participating Mac.

Handles: hardware profiling, mDNS registration, peer discovery,
heartbeat gossip, coordinator election, and work assignment.

    macfleet join  →  starts the agent (Ctrl+C stops it gracefully)
    Pool.leave()   →  stops an in-process agent from Python
"""

from __future__ import annotations

import asyncio
import logging
import secrets as secrets_mod
import socket
import ssl
import subprocess
import sys
from functools import lru_cache
from typing import Optional

from rich.console import Console

logger = logging.getLogger(__name__)

from macfleet.comm.transport import HardwareExchange
from macfleet.compute.worker import TaskWorker
from macfleet.engines.base import HardwareProfile
from macfleet.monitoring.thermal import get_thermal_state
from macfleet.pool.discovery import (
    MAX_MDNS_INSTANCE_NAME_BYTES,
    DiscoveredNode,
    ServiceRegistry,
)
from macfleet.pool.heartbeat import GossipHeartbeat, HeartbeatConfig, NodeStatus, request_heartbeat
from macfleet.pool.network import LinkType, get_network_topology
from macfleet.pool.registry import ClusterRegistry, NodeRecord
from macfleet.security.auth import (
    HW_HANDSHAKE_MAX_JSON_BYTES,
    MAX_NODE_ID_BYTES,
    AuthRateLimiter,
    HandshakeHwValidationError,
    SecurityConfig,
    create_client_ssl_context,
    create_server_tls_context,
    heartbeat_channel_key,
    sign_heartbeat_response,
    verify_heartbeat,
    verify_heartbeat_with_hw,
)

# v2.2 PR 6 (Issue 22): heartbeat read timeout tightened from 5s to 1s.
# Rationale: a legitimate ping completes in <50ms on WiFi. The old 5s
# timeout meant a burst of 100 attacker connections could stall the
# event loop for 500s worth of pending reads. 1s is still generous for
# a cross-continent round-trip but slams the door on slowloris-style DoS.
HEARTBEAT_READ_TIMEOUT_SEC = 1.0
# Cap how long a write to a slow / lagging peer can hold the
# heartbeat handler. Without this, a slowloris-on-the-receive-side
# attacker could pin handlers indefinitely on writer.drain().
HEARTBEAT_WRITE_TIMEOUT_SEC = 1.0
MAX_REVERSE_PEER_PROBES = 32

console = Console()


def _pick_ephemeral_port(exclude: int = 0) -> int:
    """Return a kernel-assigned ephemeral port that's not `exclude`.

    v2.2 post-rc port-flake fix: when PoolAgent is constructed with
    port=0 data_port=0, we bind the heartbeat server first to lock
    `self.port`, then call this to pick `self.data_port`. Rare 1-in-60k
    case of the kernel handing back the same port is handled by a retry.

    Returns the first port distinct from `exclude` from up to 32 tries
    (doubled from 16 to give the kernel breathing room under contention
    on tightly-loaded CI runners).
    """
    last_seen = -1
    for _ in range(32):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        last_seen = port
        if port != exclude:
            return port  # type: ignore[no-any-return]
    raise RuntimeError(
        f"could not pick an ephemeral port distinct from {exclude} after 32 tries "
        f"(last kernel-assigned port: {last_seen})"
    )


def _detect_chip_name() -> str:
    """Detect the Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except (subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


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
    except (subprocess.TimeoutExpired, OSError):
        pass
    return 0


def _detect_ram_gb() -> float:
    """Detect total RAM in GB."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return 0.0


def _detect_memory_bandwidth() -> float:
    """Estimate memory bandwidth in GB/s based on chip."""
    chip = _detect_chip_name().lower()
    # Rough estimates based on known Apple Silicon specs
    if not any(chip == f"apple m{n}" or chip.startswith(f"apple m{n} ") for n in range(1, 5)):
        return 0.0
    if "ultra" in chip:
        return 800.0
    elif "max" in chip:
        return 400.0
    elif "pro" in chip:
        return 200.0
    else:
        return 100.0  # base M-series


@lru_cache(maxsize=2)
def _probe_framework(script: str) -> bool:
    """Probe optional runtimes without loading them into the pool daemon."""
    try:
        return subprocess.run(
            [sys.executable, "-c", script], capture_output=True, timeout=10,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _check_mps_available() -> bool:
    """Check the installed Torch runtime's MPS support in a separate process."""
    return _probe_framework("import torch; raise SystemExit(0 if torch.backends.mps.is_available() else 1)")


def _check_mlx_available() -> bool:
    """Check if MLX is installed and functional."""
    return _probe_framework("import mlx.core")


def _generate_node_id(hostname: str) -> str:
    """Keep the readable prefix and complete random suffix within an mDNS label."""
    suffix = f"-{secrets_mod.token_hex(4)}"
    prefix_bytes = hostname.encode("utf-8")[:MAX_MDNS_INSTANCE_NAME_BYTES - len(suffix)]
    # Slicing can split a multibyte character; discard only that partial tail.
    return prefix_bytes.decode("utf-8", errors="ignore") + suffix


def profile_hardware() -> HardwareProfile:
    """Profile the local Mac's hardware capabilities."""
    hostname = socket.gethostname()
    node_id = _generate_node_id(hostname)

    return HardwareProfile(
        hostname=hostname,
        node_id=node_id,
        gpu_cores=_detect_gpu_cores(),
        ram_gb=_detect_ram_gb(),
        memory_bandwidth_gbps=_detect_memory_bandwidth(),
        has_ane=_detect_chip_name().lower().startswith("apple m"),
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
        for option, value in (("port", port), ("data_port", data_port)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535):
                raise ValueError(f"{option} must be an integer between 0 and 65535")
        if port is None:
            raise ValueError("port must be an integer between 0 and 65535")
        if port == 65535 and data_port is None:
            raise ValueError("port=65535 requires an explicit data_port below 65536")
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
        self._peer_probes: dict[str, asyncio.Task] = {}

        self._running = False
        self._task_worker: Optional[TaskWorker] = None
        self._heartbeat_server: Optional[asyncio.Server] = None
        self._heartbeat_cert = b""
        self._heartbeat_ssl_ctx = None
        if self._security.tls:
            self._heartbeat_ssl_ctx, self._heartbeat_cert = create_server_tls_context()

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
            heartbeat_port=self.port,
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
        if self._running:
            raise RuntimeError("Pool agent is already running")
        try:
            await self._start()
        except BaseException:
            await self.stop()
            raise

    async def _start(self) -> None:
        # 1. Profile hardware
        self.hardware = profile_hardware()
        if self._name_override:
            self.hardware.hostname = self._name_override
            self.hardware.node_id = _generate_node_id(self._name_override)

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
        link_types_str = ",".join(sorted({
            link.link_type.value for link in topology.links if link.link_type != LinkType.LOOPBACK
        }))
        console.print(f"  Network: {link_types_str or 'none detected'}")

        # Pick the best IP to advertise
        best = topology.best_link
        ip_address = best.ip_address if best else "127.0.0.1"
        if best is None:
            console.print("[yellow]No usable IPv4 interface; this pool is reachable only on this Mac.[/yellow]")

        # 3. Start heartbeat responder server FIRST so we know the actual bound
        # port before anything else (registry / mDNS) needs it. When the caller
        # passed port=0, the kernel picks an ephemeral port; we rewrite
        # self.port to reflect reality. reuse_address=True survives TIME_WAIT
        # from a previous clean stop, which is what CI test rapid-fire needs.
        self._heartbeat_server = await asyncio.start_server(
            self._handle_heartbeat_ping, "0.0.0.0", self.port,
            ssl=self._heartbeat_ssl_ctx,
            ssl_handshake_timeout=3.0 if self._heartbeat_ssl_ctx else None,
            ssl_shutdown_timeout=3.0 if self._heartbeat_ssl_ctx else None,
            reuse_address=True,
        )
        bound_port = self._heartbeat_server.sockets[0].getsockname()[1]
        if self.port == 0:
            self.port = bound_port
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
            network_links=tuple(topology.links),
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
            network_links=tuple(topology.links),
        )

        # 7. Start heartbeat gossip. In secure mode the gossip pings carry
        # signed hardware so mDNS-discovered peers refresh from a
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

        # 6. Start discovery
        loop = asyncio.get_running_loop()

        def discovered(node: DiscoveredNode) -> None:
            loop.call_soon_threadsafe(self._on_peer_discovered, node)

        def removed(node_id: str) -> None:
            loop.call_soon_threadsafe(self._on_peer_removed, node_id)

        self._discovery.start_discovery(on_add=discovered, on_remove=removed, on_update=discovered)
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

        probes = list(self._peer_probes.values())
        for probe in probes:
            probe.cancel()
        await asyncio.gather(*probes, return_exceptions=True)
        self._peer_probes.clear()

        if self._task_worker:
            await self._task_worker.stop()
            self._task_worker = None

        if self._heartbeat:
            await self._heartbeat.stop()
            self._heartbeat = None
        if self._heartbeat_server:
            self._heartbeat_server.close()
            await self._heartbeat_server.wait_closed()
            self._heartbeat_server = None

        await self._discovery.async_stop()
        self._registry = None
        console.print("[yellow]Left pool[/yellow]")

    async def _handle_heartbeat_ping(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Verify a TLS-bound client proof before returning signed hardware and accepting a reverse probe."""
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
                try:
                    verb, peer_id, nonce_hex, proof_hex = data.decode().split()
                    nonce = bytes.fromhex(nonce_hex)
                    if writer.get_extra_info("ssl_object") is None:
                        raise ValueError("Missing heartbeat TLS connection")
                    key = heartbeat_channel_key(fleet_key, self._heartbeat_cert)
                    if (verb != "APING" or len(peer_id.encode()) > MAX_NODE_ID_BYTES or len(nonce) != 16
                            or not verify_heartbeat(key, peer_id, nonce, bytes.fromhex(proof_hex))):
                        raise ValueError("Invalid heartbeat proof")
                except ValueError:
                    self._heartbeat_rate_limiter.record_failure(peer_ip)
                    return
                self._heartbeat_rate_limiter.record_success(peer_ip)
                heartbeat = self._heartbeat
                prior_peer = heartbeat.get_peer(peer_id) if heartbeat is not None else None
                server_nonce = secrets_mod.token_bytes(16)
                hardware = self._local_hw_exchange().to_json_bytes()
                proof = sign_heartbeat_response(key, self.node_id, server_nonce, nonce, hardware)
                writer.write(f"APONG {self.node_id} {server_nonce.hex()} {proof.hex()} {hardware.hex()}\n".encode())
                await asyncio.wait_for(writer.drain(), timeout=HEARTBEAT_WRITE_TIMEOUT_SEC)
                reply = await asyncio.wait_for(reader.readline(), timeout=HEARTBEAT_READ_TIMEOUT_SEC)
                if reply:
                    verb, signature_hex, hw_hex = reply.decode().split()
                    hw = bytes.fromhex(hw_hex)
                    if (verb != "AHW" or len(hw) > HW_HANDSHAKE_MAX_JSON_BYTES
                            or not verify_heartbeat_with_hw(
                                key, peer_id, server_nonce, hw, bytes.fromhex(signature_hex))):
                        self._heartbeat_rate_limiter.record_failure(peer_ip)
                        return
                    # Discovery can replace this member while its inbound handshake is suspended.
                    if self._heartbeat is heartbeat and (
                        heartbeat is None or heartbeat.get_peer(peer_id) is prior_peer
                    ):
                        self._queue_peer_probe(peer_id, peer_ip, HardwareExchange.from_json_bytes(hw))
            elif not fleet_key and data.startswith(b"PING"):
                # Open heartbeat (backward compatible)
                writer.write(f"PONG {self.node_id}\n".encode())
                try:
                    await asyncio.wait_for(
                        writer.drain(), timeout=HEARTBEAT_WRITE_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    logger.debug("PONG drain timed out for %s", peer_ip)
            elif fleet_key:
                # Secure server got a non-APING message. Empty data
                # (immediate EOF from a port scanner / TCP RST) is benign
                # noise; only count an actual non-empty payload as a
                # failed auth attempt. This avoids banning a colleague's
                # `nc localhost 50051` after a few exploratory connects.
                if data:
                    self._heartbeat_rate_limiter.record_failure(peer_ip)
                    logger.debug(
                        "Heartbeat from %s: secure server received non-APING (%r)",
                        peer_ip, data[:32],
                    )
                else:
                    logger.debug(
                        "Heartbeat from %s: empty payload, not counted as failure",
                        peer_ip,
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

    def _queue_peer_probe(self, peer_id: str, peer_ip: str, hw: HardwareExchange) -> None:
        """Verify an inbound caller's listening endpoint before admitting it."""
        registry = self._registry
        if (not self._running or not self._security.is_secure or registry is None
                or not hw.heartbeat_port or peer_id == self.node_id or not peer_id
                or len(peer_id.encode("utf-8")) > MAX_NODE_ID_BYTES or peer_ip == "unknown"):
            return
        known = registry.get_node(peer_id)
        if (known is not None and known.is_alive
                and known.ip_address == peer_ip and known.port == hw.heartbeat_port):
            return
        if peer_id in self._peer_probes or len(self._peer_probes) >= MAX_REVERSE_PEER_PROBES:
            return

        # APING can be replayed; require a fresh APONG from the claimed source endpoint.
        async def verify_peer() -> None:
            async with asyncio.timeout(10.0):
                await self._add_manual_peer(f"{peer_ip}:{hw.heartbeat_port}", expected_peer_id=peer_id)

        probe = asyncio.create_task(verify_peer())
        self._peer_probes[peer_id] = probe

        def finished(done: asyncio.Task) -> None:
            if self._peer_probes.get(peer_id) is done:
                self._peer_probes.pop(peer_id)
            if not done.cancelled() and (error := done.exception()) is not None:
                logger.debug("Reverse peer probe failed for %s: %s", peer_id, type(error).__name__)

        probe.add_done_callback(finished)

    def _cancel_peer_probe(self, node_id: str) -> None:
        probe = self._peer_probes.pop(node_id, None)
        if probe is not None:
            probe.cancel()

    async def _add_manual_peer(self, peer_addr: str, *, expected_peer_id: Optional[str] = None) -> None:
        """Connect to a manually specified peer (bypasses mDNS).

        Used when mDNS is blocked (e.g. enterprise WiFi with client isolation).
        Sends a heartbeat ping to verify the peer is reachable and running.

        Secure peers exchange hardware after verifying the endpoint's TLS-bound proof.
        """
        # The responder's ID is learned from the verified reply. Snapshot tracker
        # identities before the await so a departed or replaced peer cannot be restored.
        tracked = self._heartbeat.peers if self._security.is_secure and self._heartbeat is not None else {}
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
            assert writer is not None  # narrow Optional for mypy

            peer_hw: Optional[HardwareExchange] = None
            peer_data_port = port + 1
            if fleet_key:
                peer_node_id, hw_json = await asyncio.wait_for(request_heartbeat(
                    reader, writer, fleet_key, self.node_id,
                    self._local_hw_exchange().to_json_bytes(), expected_peer_id,
                ), timeout=10.0)
                if hw_json is not None:
                    peer_hw = HardwareExchange.from_json_bytes(hw_json)
                    if peer_hw.data_port:
                        peer_data_port = peer_hw.data_port
            else:
                writer.write(f"PING {self.node_id}\n".encode())
                await asyncio.wait_for(writer.drain(), timeout=HEARTBEAT_WRITE_TIMEOUT_SEC)
                response = await asyncio.wait_for(reader.readline(), timeout=10.0)
                if not response.startswith(b"PONG"):
                    console.print(f"[red]Peer {peer_addr}: no response (got: {response[:50]!r})[/red]")
                    return
                parts = response.decode().strip().split(" ")
                peer_node_id = parts[1] if len(parts) >= 2 else f"peer-{host}"

            if expected_peer_id is not None and peer_node_id != expected_peer_id:
                logger.warning(
                    "Manual peer identity mismatch: expected %s, received %s", expected_peer_id, peer_node_id,
                )
                return
            if peer_node_id == self.node_id:
                return

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
            registry = self._registry
            heartbeat = self._heartbeat
            if registry is None or heartbeat is None:
                raise RuntimeError("PoolAgent manual peer registration requires start() first")
            if self._security.is_secure and heartbeat.get_peer(peer_node_id) is not tracked.get(peer_node_id):
                return
            # Store the address that answered (a manual hostname can resolve to multiple IPs).
            peername = writer.get_extra_info("peername") if self._security.is_secure else None
            address = peername[0] if isinstance(peername, tuple) and peername else host
            previous = registry.get_node(peer_node_id)
            endpoint = (address, port, peer_data_port)
            same_endpoint = (previous is not None and previous.status != NodeStatus.LEFT
                             and (previous.ip_address, previous.port, previous.data_port) == endpoint)
            if self._security.is_secure and same_endpoint and tracked.get(peer_node_id) is not None:
                if peer_hw is not None:
                    registry.update_hardware(peer_node_id, hw)
                registry.mark_alive(peer_node_id)
                heartbeat.record_heartbeat(peer_node_id)
            else:
                registry.register(NodeRecord(
                    node_id=peer_node_id,
                    hostname=peer_node_id,
                    ip_address=address,
                    port=port,
                    data_port=peer_data_port,
                    hardware=hw,
                ))
                heartbeat.add_peer(peer_node_id, address, port, hw.compute_score)

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
            console.print(
                "[dim]Check the peer's heartbeat address and port. "
                "For SDK training, start the same training script on every Mac.[/dim]"
            )
        finally:
            if writer is not None:
                try:
                    writer.close()
                    await asyncio.wait_for(writer.wait_closed(), timeout=HEARTBEAT_WRITE_TIMEOUT_SEC)
                except (OSError, ssl.SSLError, BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                    pass

    def _on_peer_discovered(self, node: DiscoveredNode) -> None:
        """Called when a new peer is discovered via mDNS."""
        if node.node_id == self.node_id:
            return  # Ignore self
        registry = self._registry
        if registry is None:
            logger.debug("Ignoring discovered peer %s before registry initialization", node.node_id)
            return

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

        registered = registry.register_discovered(NodeRecord(
            node_id=node.node_id,
            hostname=node.hostname,
            ip_address=node.ip_address,
            port=node.port,
            data_port=node.data_port,
            hardware=hw,
            network_links=node.network_links,
            status=NodeStatus.SUSPECTED if self._security.is_secure else NodeStatus.ALIVE,
        ))

        if not registered:
            return
        self._cancel_peer_probe(node.node_id)

        # Use locally-computed score, not the broadcast one. Guard against the
        # discovery callback firing before _heartbeat is constructed in start().
        if self._heartbeat is not None:
            self._heartbeat.add_peer(node.node_id, node.ip_address, node.port, hw.compute_score)

        if self._security.is_secure:
            details = "hardware details omitted from secure discovery"
        else:
            cores = f"{node.gpu_cores} GPU cores" if node.gpu_cores else "GPU cores unknown"
            details = f"{node.chip_name}, {cores}"
        console.print(f"[cyan]Discovered[/cyan] {node.hostname} ({details})")

        if registry.is_coordinator:
            console.print(
                f"[bold yellow]This node is the coordinator[/bold yellow] "
                f"(world_size={registry.world_size})"
            )

    def _on_peer_removed(self, node_id: str) -> None:
        """Called when a peer's mDNS registration expires."""
        self._cancel_peer_probe(node_id)
        registry = self._registry
        if registry is not None:
            registry.deregister(node_id)
        heartbeat = self._heartbeat
        if heartbeat is not None:
            heartbeat.remove_peer(node_id)
        console.print(f"[yellow]Peer removed:[/yellow] {node_id}")

    def _on_peer_suspected(self, node_id: str) -> None:
        """Called when heartbeat suspects a peer is down."""
        console.print(f"[yellow]Peer suspected:[/yellow] {node_id}")

    def _on_peer_failed(self, node_id: str) -> None:
        """Called when a peer is confirmed failed."""
        registry = self._registry
        if registry is None:
            logger.debug("Ignoring failed peer %s before registry initialization", node_id)
            return
        registry.mark_failed(node_id)
        console.print(f"[red]Peer failed:[/red] {node_id} (world_size={registry.world_size})")

    def _on_peer_recovered(self, node_id: str) -> None:
        """Called when a suspected peer recovers."""
        registry = self._registry
        if registry is None:
            logger.debug("Ignoring recovered peer %s before registry initialization", node_id)
            return
        registry.mark_alive(node_id)
        console.print(f"[green]Peer recovered:[/green] {node_id}")

    def _gossip_local_hw_bytes(self) -> Optional[bytes]:
        """Return the local HW exchange JSON for authenticated gossip, or None."""
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
        registry = self._registry
        if registry is None:
            return
        try:
            peer_hw = HardwareExchange.from_json_bytes(peer_hw_json)
        except HandshakeHwValidationError as e:
            logger.debug("APONG v2 HW from %s: parse failed: %s", peer_node_id, e)
            return
        new_hw = self._hw_from_exchange(peer_node_id, peer_hw)
        # Hand the swap to the registry so (hardware, data_port) are
        # mutated atomically under self._registry._lock and the coordinator
        # re-election runs in the same critical section.
        new_data_port = peer_hw.data_port if peer_hw.data_port > 0 else None
        registry.update_hardware(peer_node_id, new_hw, new_data_port=new_data_port)
        registry.mark_alive(peer_node_id)
