"""Adaptive TCP transport for MacFleet v2.

Manages per-peer TCP connections with:
- Handshake protocol for peer identification (v2.2: carries signed HW profile)
- Adaptive buffer sizes based on link type (WiFi/Ethernet/TB4)
- Per-connection send/recv locks for safe concurrent operations
- WireMessage protocol with CRC32 verification
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from macfleet.comm.protocol import (
    HEADER_SIZE,
    MessageFlags,
    MessageType,
    WireMessage,
)
from macfleet.pool.network import LinkType
from macfleet.security.auth import (
    CHALLENGE_SIZE,
    HW_HANDSHAKE_MAX_JSON_BYTES,
    HW_HANDSHAKE_WIRE_VERSION,
    AuthRateLimiter,
    HandshakeHwValidationError,
    SecurityConfig,
    compute_response,
    create_client_ssl_context,
    create_server_ssl_context,
    generate_challenge,
    sign_hw_profile,
    verify_hw_profile,
    verify_response,
)

logger = logging.getLogger(__name__)


@dataclass
class TransportConfig:
    """Transport layer configuration."""

    recv_timeout_sec: float = 120.0
    connect_timeout_sec: float = 10.0
    # Buffer sizes tuned per link type
    wifi_buffer_bytes: int = 1_048_576  # 1 MB
    ethernet_buffer_bytes: int = 2_097_152  # 2 MB
    thunderbolt_buffer_bytes: int = 4_194_304  # 4 MB
    default_buffer_bytes: int = 1_048_576  # 1 MB

    def buffer_size_for_link(self, link_type: LinkType) -> int:
        """Return optimal buffer size for the given link type."""
        return {
            LinkType.THUNDERBOLT: self.thunderbolt_buffer_bytes,
            LinkType.ETHERNET: self.ethernet_buffer_bytes,
            LinkType.WIFI: self.wifi_buffer_bytes,
        }.get(link_type, self.default_buffer_bytes)


@dataclass
class HardwareExchange:
    """Hardware profile that peers exchange during the v2.2 authenticated handshake.

    Signed with the fleet HMAC key (bound to the recipient's challenge nonce),
    so a peer can't lie about hardware to win coordinator election and also
    can't replay a captured HW payload from another session.

    Fields mirror the subset of `HardwareProfile` that's relevant for
    election + capability checks. Thermal pressure is NOT included — it's
    dynamic and travels via gossip heartbeats instead.
    """

    wire_version: int = HW_HANDSHAKE_WIRE_VERSION
    gpu_cores: int = 0
    ram_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    chip_name: str = "unknown"
    has_ane: bool = False
    mps_available: bool = False
    mlx_available: bool = False
    data_port: int = 0

    def to_json_bytes(self) -> bytes:
        """Stable JSON encoding (sort_keys so HMAC is deterministic)."""
        return json.dumps(asdict(self), sort_keys=True).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "HardwareExchange":
        """Parse JSON bytes back into a HardwareExchange.

        Raises HandshakeHwValidationError on malformed input so the handshake
        path can reject without leaking JSON errors upstream.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
            if not isinstance(payload, dict):
                raise HandshakeHwValidationError("HW payload not a JSON object")
            # Only accept known fields; ignore extras for forward compat
            known = {f for f in cls.__dataclass_fields__}
            filtered = {k: v for k, v in payload.items() if k in known}
            return cls(**filtered)
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            raise HandshakeHwValidationError(f"HW payload deserialization failed: {e}") from e


def _pack_hw_suffix(
    fleet_key: bytes, local_id: str, hw: HardwareExchange, peer_challenge: bytes,
) -> bytes:
    """Build the v2.2-handshake HW-exchange suffix appended to the ACK and RESP.

    Wire layout (designed for right-to-left peeling off a variable-length base):

        wire_version (1B)
        hw_json_len  (2B BE)
        hw_json      (variable)
        hmac         (32B) — signed over (wire || peer_challenge || local_id || hw_json)
        block_size   (2B BE) — TRAILING total length of everything above

    The trailing `block_size` lets the receiver peel the suffix off an
    ACK whose prefix (`peer_id`) has unknown length without needing to
    reparse from the left.

    HMAC is bound to `peer_challenge` (the challenge this peer sent to us)
    so the suffix can't be replayed against another session — A5.
    """
    hw_json = hw.to_json_bytes()
    if len(hw_json) > HW_HANDSHAKE_MAX_JSON_BYTES:
        raise HandshakeHwValidationError(
            f"HW payload {len(hw_json)}B exceeds max {HW_HANDSHAKE_MAX_JSON_BYTES}B"
        )
    sig = sign_hw_profile(
        fleet_key, hw.wire_version, peer_challenge, local_id, hw_json,
    )
    body = struct.pack("!BH", hw.wire_version & 0xFF, len(hw_json)) + hw_json + sig
    return body + struct.pack("!H", len(body))


def _peel_hw_suffix(
    fleet_key: bytes, peer_id: str, payload: bytes, sent_challenge: bytes,
) -> tuple[bytes, HardwareExchange]:
    """Peel a v2.2 HW suffix off the right of `payload`. Returns (base, hw).

    Verifies the HMAC against `sent_challenge` (the challenge WE originally
    sent to this peer — replay protection). Raises HandshakeHwValidationError
    on any structural or cryptographic failure.
    """
    if len(payload) < 2:
        raise HandshakeHwValidationError("payload too short to carry HW suffix length")
    (block_size,) = struct.unpack("!H", payload[-2:])
    # Sanity: block_size = 1 (wire) + 2 (hw_len) + hw_json_len + 32 (hmac)
    if block_size < 3 + 32 or block_size > 3 + HW_HANDSHAKE_MAX_JSON_BYTES + 32:
        raise HandshakeHwValidationError(
            f"HW suffix block_size {block_size} outside valid range"
        )
    suffix_start = len(payload) - 2 - block_size
    if suffix_start < 0:
        raise HandshakeHwValidationError(
            f"HW suffix block_size {block_size} exceeds payload length {len(payload)}"
        )
    block = payload[suffix_start : suffix_start + block_size]
    base = payload[:suffix_start]

    wire_version, hw_len = struct.unpack_from("!BH", block, 0)
    if hw_len > HW_HANDSHAKE_MAX_JSON_BYTES:
        raise HandshakeHwValidationError(
            f"peer HW payload length {hw_len} exceeds max {HW_HANDSHAKE_MAX_JSON_BYTES}"
        )
    expected_total = 3 + hw_len + 32
    if block_size != expected_total:
        raise HandshakeHwValidationError(
            f"HW block structure mismatch: block_size={block_size}, "
            f"expected={expected_total} (wire+hw_len+hw_json+hmac)"
        )
    hw_json = block[3 : 3 + hw_len]
    sig = block[3 + hw_len :]
    if not verify_hw_profile(fleet_key, wire_version, sent_challenge, peer_id, hw_json, sig):
        raise HandshakeHwValidationError("HW profile signature invalid")
    hw = HardwareExchange.from_json_bytes(hw_json)
    if wire_version != hw.wire_version:
        logger.warning(
            "HW handshake wire_version mismatch: header=%d, payload=%d — accepting",
            wire_version, hw.wire_version,
        )
    return base, hw


@dataclass
class PeerConnection:
    """A TCP connection to a single peer node.

    Provides send/recv with per-direction locks so that
    concurrent allreduce (send to right, recv from left)
    works safely on a single connection.
    """

    peer_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    link_type: LinkType = LinkType.UNKNOWN
    connected_at: float = field(default_factory=time.time)
    bytes_sent: int = 0
    bytes_received: int = 0
    # v2.2 PR 4: peer's HW profile from the signed handshake exchange, or
    # None if the peer is v2.1 (no HW in handshake) or the connection is open.
    peer_hw: Optional[HardwareExchange] = None
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _recv_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send_message(self, msg: WireMessage) -> None:
        """Send a WireMessage (header + payload with CRC32)."""
        data = msg.pack()
        async with self._send_lock:
            self.writer.write(data)
            await self.writer.drain()
            self.bytes_sent += len(data)

    async def recv_message(self, timeout: float = 120.0) -> WireMessage:
        """Receive a WireMessage with CRC32 verification."""
        async with self._recv_lock:
            msg = await asyncio.wait_for(
                WireMessage.read_from_stream(self.reader),
                timeout=timeout,
            )
            self.bytes_received += HEADER_SIZE + len(msg.payload)
            return msg

    async def send_bytes(
        self,
        payload: bytes,
        msg_type: MessageType = MessageType.TENSOR,
        stream_id: int = 1,
        sequence: int = 0,
    ) -> None:
        """Send raw bytes wrapped in a WireMessage."""
        msg = WireMessage(
            stream_id=stream_id,
            msg_type=msg_type,
            flags=MessageFlags.NONE,
            sequence=sequence,
            payload=payload,
        )
        await self.send_message(msg)

    async def recv_bytes(self, timeout: float = 120.0) -> bytes:
        """Receive raw bytes (unwrapped from WireMessage)."""
        msg = await self.recv_message(timeout=timeout)
        return msg.payload

    async def close(self) -> None:
        """Close this connection gracefully."""
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class PeerTransport:
    """Manages TCP connections to all peers in the cluster.

    Each peer gets one bidirectional TCP connection. The handshake protocol
    exchanges node IDs and, when the fleet is authenticated and both sides
    are v2.2+, also exchanges HMAC-signed hardware profiles (for coordinator
    election by real compute score instead of mDNS-broadcast zeros).

    Handshake when `security.fleet_key` is set and `local_hw` is provided:

        1. Client sends CONTROL{HANDSHAKE_V2} with `local_id || challenge_a`.
        2. Server verifies fleet key by checking the HANDSHAKE_V2 flag, builds
           an ACK payload: `local_id || response_a || challenge_b || hw_block_s`
           where `hw_block_s` is the structured + HMAC-signed HW exchange
           bound to `challenge_a` (the client's challenge).
        3. Client verifies `response_a` and parses+verifies `hw_block_s` against
           the challenge it originally sent. Then sends RESP:
           `response_b || hw_block_c` with `hw_block_c` bound to `challenge_b`.
        4. Server verifies `response_b` and parses+verifies `hw_block_c`.

    Fallback:
        - v2.1 client (no HANDSHAKE_V2 flag) → server uses legacy parsing and
          DOES NOT exchange HW. Logs a warning noting the mixed-version fleet
          will have degraded coordinator election.
        - Open fleet (no fleet_key) → handshake is just `local_id` ↔ `local_id`
          unchanged. No HW exchange (nothing to HMAC-sign with).
    """

    def __init__(
        self,
        local_id: str,
        config: Optional[TransportConfig] = None,
        security: Optional[SecurityConfig] = None,
        local_hw: Optional[HardwareExchange] = None,
    ):
        self.local_id = local_id
        self.config = config or TransportConfig()
        self._security = security or SecurityConfig()
        # Set to a real profile by callers that want HW exchange during the
        # authenticated handshake. When None, the server still advertises a
        # zero-filled profile — which matches the pre-v2.2 behavior where the
        # registry had no HW data for remote peers in secure mode.
        self._local_hw = local_hw or HardwareExchange()
        self._connections: dict[str, PeerConnection] = {}
        self._server: Optional[asyncio.Server] = None
        self._lock = asyncio.Lock()
        self._on_connect: Optional[Callable] = None
        self._rate_limiter = AuthRateLimiter()
        self._server_ssl_ctx = None
        if self._security.tls:
            self._server_ssl_ctx = create_server_ssl_context()

    @property
    def local_hw(self) -> HardwareExchange:
        return self._local_hw

    def set_local_hw(self, hw: HardwareExchange) -> None:
        """Update the local HW profile. Affects handshakes started after this call.

        PoolAgent calls this once it has profiled hardware at startup, before
        dispatching any client connects. Existing connections keep their
        already-exchanged HW.
        """
        self._local_hw = hw

    @property
    def peer_ids(self) -> list[str]:
        """IDs of all connected peers."""
        return list(self._connections.keys())

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    def get_connection(self, peer_id: str) -> Optional[PeerConnection]:
        """Get connection to a specific peer, or None."""
        return self._connections.get(peer_id)

    async def start_server(
        self,
        host: str = "0.0.0.0",
        port: int = 50052,
        on_connect: Optional[Callable] = None,
    ) -> None:
        """Start listening for incoming peer connections.

        Args:
            host: Bind address.
            port: Bind port.
            on_connect: Optional callback(peer_id, PeerConnection).
        """
        self._on_connect = on_connect

        async def handle_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            # Get peer IP for rate limiting
            peername = writer.get_extra_info("peername")
            peer_ip = peername[0] if peername else "unknown"

            try:
                # SECURITY: Rate limit — reject banned IPs immediately
                if self._rate_limiter.is_banned(peer_ip):
                    logger.warning("Rate limit: rejecting banned IP %s", peer_ip)
                    writer.close()
                    return

                # Apply backoff delay for IPs with recent failures
                delay = self._rate_limiter.get_delay(peer_ip)
                if delay > 0:
                    await asyncio.sleep(delay)

                # Read handshake: peer sends its ID (+ challenge if secure)
                msg = await asyncio.wait_for(
                    WireMessage.read_from_stream(reader),
                    timeout=self.config.connect_timeout_sec,
                )
                if msg.msg_type != MessageType.CONTROL:
                    writer.close()
                    return

                payload = msg.payload
                fleet_key = self._security.fleet_key

                if fleet_key:
                    # SECURITY: Downgrade protection — secure server rejects
                    # open handshakes (payload without challenge appended)
                    if len(payload) < CHALLENGE_SIZE + 1:
                        logger.warning(
                            "Auth handshake: payload too short from %s "
                            "(possible downgrade attack or misconfigured peer)",
                            peer_ip,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        writer.close()
                        return
                    peer_id = payload[:-CHALLENGE_SIZE].decode("utf-8")
                    challenge_a = payload[-CHALLENGE_SIZE:]

                    # v2.2 PR 4: distinguish v2.2 client from v2.1 client via
                    # the HANDSHAKE_V2 flag. v2.2 gets a signed-HW ACK; v2.1
                    # gets the legacy ACK (no HW) — this keeps mixed-fleet
                    # connections alive during upgrade rollout.
                    v2_handshake = bool(msg.flags & MessageFlags.HANDSHAKE_V2)
                    if not v2_handshake:
                        logger.info(
                            "Auth handshake with v2.1 client %s (%s): no HW "
                            "exchange, coordinator election will treat peer "
                            "as zero-compute until both sides upgrade to v2.2",
                            peer_id, peer_ip,
                        )

                    # Compute response to peer's challenge + send our own challenge
                    response_a = compute_response(fleet_key, challenge_a)
                    challenge_b = generate_challenge()
                    base_ack = (
                        self.local_id.encode("utf-8") + response_a + challenge_b
                    )
                    if v2_handshake:
                        # Append HW suffix bound to the client's challenge, so
                        # this payload can't be replayed against another session.
                        ack_payload = base_ack + _pack_hw_suffix(
                            fleet_key, self.local_id, self._local_hw, challenge_a,
                        )
                        ack_flags = MessageFlags.HANDSHAKE_V2
                    else:
                        ack_payload = base_ack
                        ack_flags = MessageFlags.NONE
                    ack = WireMessage(
                        stream_id=0,
                        msg_type=MessageType.CONTROL,
                        flags=ack_flags,
                        sequence=0,
                        payload=ack_payload,
                    )
                    conn = PeerConnection(peer_id=peer_id, reader=reader, writer=writer)
                    await conn.send_message(ack)

                    # Read peer's response to our challenge
                    msg2 = await asyncio.wait_for(
                        WireMessage.read_from_stream(reader),
                        timeout=self.config.connect_timeout_sec,
                    )
                    # v2.2: peel HW suffix off the RESP before verifying response_b.
                    # v2.1: RESP is just response_b.
                    if v2_handshake:
                        try:
                            base_resp, peer_hw = _peel_hw_suffix(
                                fleet_key, peer_id, msg2.payload, challenge_b,
                            )
                        except HandshakeHwValidationError as e:
                            logger.warning(
                                "Auth handshake: HW suffix from %s (%s) failed: %s",
                                peer_id, peer_ip, e,
                            )
                            self._rate_limiter.record_failure(peer_ip)
                            writer.close()
                            await writer.wait_closed()
                            return
                        response_b = base_resp
                    else:
                        response_b = msg2.payload
                        peer_hw = None
                    if len(response_b) != CHALLENGE_SIZE:
                        logger.warning(
                            "Auth handshake: RESP response_b wrong size from %s "
                            "(got %d, expected %d)",
                            peer_id, len(response_b), CHALLENGE_SIZE,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        writer.close()
                        await writer.wait_closed()
                        return
                    if not verify_response(fleet_key, challenge_b, response_b):
                        logger.warning(
                            "Auth handshake: peer %s (%s) failed challenge "
                            "(wrong token or attack)",
                            peer_id, peer_ip,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        writer.close()
                        await writer.wait_closed()
                        return

                    # HMAC on peer HW was already verified inside _peel_hw_suffix
                    # (bound to challenge_b — replay protection). Attach to conn.
                    if peer_hw is not None:
                        conn.peer_hw = peer_hw

                    # Auth succeeded
                    self._rate_limiter.record_success(peer_ip)
                    logger.debug("Auth handshake succeeded: peer=%s ip=%s", peer_id, peer_ip)
                else:
                    # SECURITY: Downgrade protection — open server rejects
                    # authenticated handshakes (prevents mixed-mode confusion)
                    if len(payload) > 256:
                        logger.warning(
                            "Open handshake: payload suspiciously large from %s "
                            "(possible auth handshake sent to open server)",
                            peer_ip,
                        )
                        writer.close()
                        return

                    peer_id = payload.decode("utf-8")
                    conn = PeerConnection(peer_id=peer_id, reader=reader, writer=writer)

                    ack = WireMessage(
                        stream_id=0,
                        msg_type=MessageType.CONTROL,
                        flags=MessageFlags.NONE,
                        sequence=0,
                        payload=self.local_id.encode("utf-8"),
                    )
                    await conn.send_message(ack)

            except Exception as e:
                logger.debug("Handshake error from %s: %s", peer_ip, e)
                try:
                    writer.close()
                except Exception:
                    pass
                return

            self._tune_socket(writer, conn.link_type)

            async with self._lock:
                self._connections[peer_id] = conn

            if self._on_connect:
                self._on_connect(peer_id, conn)

        self._server = await asyncio.start_server(
            handle_client, host, port, ssl=self._server_ssl_ctx,
        )

    async def stop_server(self) -> None:
        """Stop the server and close all connections."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def connect(
        self,
        peer_id: str,
        host: str,
        port: int,
        link_type: LinkType = LinkType.UNKNOWN,
    ) -> PeerConnection:
        """Connect to a peer and perform handshake.

        Args:
            peer_id: Expected peer ID.
            host: Peer's IP address.
            port: Peer's transport port.
            link_type: Network link type (for buffer tuning).

        Returns:
            The established PeerConnection.

        Raises:
            ConnectionError: If authentication fails.
        """
        ssl_ctx = create_client_ssl_context() if self._security.tls else None

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ssl_ctx),
            timeout=self.config.connect_timeout_sec,
        )

        conn = PeerConnection(
            peer_id=peer_id,
            reader=reader,
            writer=writer,
            link_type=link_type,
        )
        self._tune_socket(writer, link_type)

        fleet_key = self._security.fleet_key

        if fleet_key:
            # v2.2 PR 4 authenticated handshake: flag the initial CONTROL message
            # with HANDSHAKE_V2 so the server knows to include a signed HW block
            # in its ACK. Payload layout (step 1) is unchanged: node_id + challenge.
            challenge_a = generate_challenge()
            handshake_payload = self.local_id.encode("utf-8") + challenge_a

            handshake = WireMessage(
                stream_id=0,
                msg_type=MessageType.CONTROL,
                flags=MessageFlags.HANDSHAKE_V2,
                sequence=0,
                payload=handshake_payload,
            )
            await conn.send_message(handshake)

            # Read ack. In v2.2 the server ACK is:
            #   base: peer_id + response_a(32) + challenge_b(32)
            #   then: hw_block = wire_version(1) + hw_len(2) + hw_json + hmac(32)
            # In v2.1 the ACK is just the base.
            ack = await asyncio.wait_for(
                conn.recv_message(),
                timeout=self.config.connect_timeout_sec,
            )
            ack_payload = ack.payload
            if len(ack_payload) < CHALLENGE_SIZE * 2 + 1:
                await conn.close()
                raise ConnectionError("Auth handshake: server response too short")

            server_v2 = bool(ack.flags & MessageFlags.HANDSHAKE_V2)
            # Verify token (response_a) FIRST. If tokens mismatch, surface
            # that error cleanly — without it, a wrong-token client would
            # blame "HW suffix invalid" instead of "wrong token" because the
            # HW HMAC uses the same fleet_key and fails for the same reason.
            # Trailing challenge_b is the last 32B of the ack for v2.1; for
            # v2.2 it's the last 32B of the base section (before the HW suffix).
            # We don't know the base boundary yet, but we DO know response_a
            # ends immediately before challenge_b. In both versions, response_a
            # is located by finding the structural boundary — simpler: peel
            # HW first if v2, then verify response_a. On mismatch, raise the
            # legacy token-error message so existing callers/tests see the
            # familiar diagnostic regardless of whether the server is v2.1 or v2.2.
            if server_v2:
                try:
                    base_ack, peer_hw = _peel_hw_suffix(
                        fleet_key, peer_id, ack_payload, challenge_a,
                    )
                except HandshakeHwValidationError as e:
                    # A HW suffix HMAC failure is symptomatically identical
                    # to a wrong-token failure (same fleet_key, same challenge),
                    # so surface the same diagnostic. The underlying error is
                    # chained for debugging.
                    await conn.close()
                    raise ConnectionError(
                        f"Auth handshake failed: peer {peer_id} does not have the correct token"
                    ) from e
                if len(base_ack) < CHALLENGE_SIZE * 2 + 1:
                    await conn.close()
                    raise ConnectionError(
                        "Auth handshake v2: ACK base section too short after HW peel"
                    )
                response_a = base_ack[-(CHALLENGE_SIZE * 2):-CHALLENGE_SIZE]
                challenge_b = base_ack[-CHALLENGE_SIZE:]
                conn.peer_hw = peer_hw
            else:
                # Legacy / v2.1 ACK: peer_id + response_a(32) + challenge_b(32)
                response_a = ack_payload[-(CHALLENGE_SIZE * 2):-CHALLENGE_SIZE]
                challenge_b = ack_payload[-CHALLENGE_SIZE:]

            # Verify server proved it knows the token
            if not verify_response(fleet_key, challenge_a, response_a):
                await conn.close()
                raise ConnectionError(
                    f"Auth handshake failed: peer {peer_id} does not have the correct token"
                )

            # Respond to server's challenge (prove we know the token too).
            # v2.2: append a HW suffix bound to server's challenge_b. v2.1: just the response.
            response_b = compute_response(fleet_key, challenge_b)
            if server_v2:
                resp_payload = response_b + _pack_hw_suffix(
                    fleet_key, self.local_id, self._local_hw, challenge_b,
                )
                resp_flags = MessageFlags.HANDSHAKE_V2
            else:
                resp_payload = response_b
                resp_flags = MessageFlags.NONE
            resp_msg = WireMessage(
                stream_id=0,
                msg_type=MessageType.CONTROL,
                flags=resp_flags,
                sequence=0,
                payload=resp_payload,
            )
            await conn.send_message(resp_msg)
        else:
            # Open handshake (backward compatible)
            handshake = WireMessage(
                stream_id=0,
                msg_type=MessageType.CONTROL,
                flags=MessageFlags.NONE,
                sequence=0,
                payload=self.local_id.encode("utf-8"),
            )
            await conn.send_message(handshake)

            await asyncio.wait_for(
                conn.recv_message(),
                timeout=self.config.connect_timeout_sec,
            )

        async with self._lock:
            self._connections[peer_id] = conn

        return conn

    async def disconnect(self, peer_id: str) -> None:
        """Disconnect from a specific peer."""
        async with self._lock:
            conn = self._connections.pop(peer_id, None)
        if conn:
            await conn.close()

    async def disconnect_all(self) -> None:
        """Disconnect from all peers and stop server."""
        async with self._lock:
            conns = list(self._connections.values())
            self._connections.clear()
        for conn in conns:
            await conn.close()
        await self.stop_server()

    async def send(
        self,
        peer_id: str,
        payload: bytes,
        msg_type: MessageType = MessageType.GRADIENT,
    ) -> None:
        """Send raw bytes to a peer (wrapped in WireMessage)."""
        conn = self._connections.get(peer_id)
        if not conn:
            raise ConnectionError(f"Not connected to peer {peer_id}")
        await conn.send_bytes(payload, msg_type=msg_type)

    async def recv(self, peer_id: str) -> bytes:
        """Receive raw bytes from a peer (unwrapped from WireMessage)."""
        conn = self._connections.get(peer_id)
        if not conn:
            raise ConnectionError(f"Not connected to peer {peer_id}")
        return await conn.recv_bytes(timeout=self.config.recv_timeout_sec)

    def _tune_socket(self, writer: asyncio.StreamWriter, link_type: LinkType) -> None:
        """Tune TCP socket options for the given link type."""
        sock = writer.get_extra_info("socket")
        if not sock:
            return
        buf_size = self.config.buffer_size_for_link(link_type)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buf_size)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buf_size)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
