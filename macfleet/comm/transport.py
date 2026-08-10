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
import math
import socket
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from macfleet.comm.protocol import (
    HEADER_SIZE,
    MessageFlags,
    MessageType,
    WireMessage,
)
from macfleet.pool.network import LinkType
from macfleet.security.audit import audit_event
from macfleet.security.auth import (
    CHALLENGE_SIZE,
    HS_LABEL_CLIENT_RESP,
    HS_LABEL_SERVER_RESP,
    HW_HANDSHAKE_MAX_JSON_BYTES,
    HW_HANDSHAKE_WIRE_VERSION,
    MAX_NODE_ID_BYTES,
    AuthRateLimiter,
    HandshakeHwValidationError,
    SecurityConfig,
    compute_client_hello_proof,
    compute_response,
    create_client_ssl_context,
    create_server_tls_context,
    generate_challenge,
    sign_hw_profile,
    tls_channel_binding_from_writer,
    verify_client_hello_proof,
    verify_hw_profile,
    verify_response,
)

logger = logging.getLogger(__name__)

MAX_HW_GPU_CORES = 1024
MAX_HW_RAM_GB = 65_536
MAX_HW_MEMORY_BANDWIDTH_GBPS = 100_000
MAX_HW_CHIP_NAME_BYTES = 128


def _audit_transport_auth(event: str, **fields: Any) -> None:
    """Record transport auth events without placing secrets in the log."""
    audit_event(event, component="transport", **fields)


class PeerAuthError(ConnectionError):
    """Raised when a handshake fails because the peer does not hold the
    correct fleet token (or its HW payload fails HMAC verification, which
    is symptomatically identical).

    Subclasses ConnectionError so existing callers that catch
    ConnectionError keep working. Mesh formation catches this specifically
    to fail fast instead of retrying a hopeless connection until timeout.
    """


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


def _validate_hw_int(
    value: object,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Require a strict JSON integer within the HW protocol bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise HandshakeHwValidationError(f"{field_name} must be an integer")
    if not minimum <= value <= maximum:
        raise HandshakeHwValidationError(
            f"{field_name} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def _validate_hw_number(
    value: object,
    *,
    field_name: str,
    maximum: float,
) -> float:
    """Require a finite, non-negative JSON number within HW protocol bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HandshakeHwValidationError(f"{field_name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= maximum:
        raise HandshakeHwValidationError(
            f"{field_name} must be finite and between 0 and {maximum:g}, got {value}"
        )
    return parsed


def _validate_hw_bool(value: object, *, field_name: str) -> bool:
    """Require an actual JSON boolean, not Python's integer-compatible bool."""
    if not isinstance(value, bool):
        raise HandshakeHwValidationError(f"{field_name} must be a boolean")
    return value


def _validate_hw_chip_name(value: object) -> str:
    """Require bounded printable text before displaying a peer chip name."""
    if not isinstance(value, str):
        raise HandshakeHwValidationError("chip_name must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as e:
        raise HandshakeHwValidationError("chip_name is not valid UTF-8 text") from e
    if len(encoded) > MAX_HW_CHIP_NAME_BYTES:
        raise HandshakeHwValidationError(
            f"chip_name exceeds {MAX_HW_CHIP_NAME_BYTES} UTF-8 bytes"
        )
    if any(not character.isprintable() for character in value):
        raise HandshakeHwValidationError("chip_name contains control characters")
    return value


def _validate_hw_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate known HW fields while preserving missing-field defaults."""
    validators: dict[str, Callable[[object], object]] = {
        "wire_version": lambda value: _validate_hw_int(
            value, field_name="wire_version", minimum=1, maximum=255
        ),
        "gpu_cores": lambda value: _validate_hw_int(
            value, field_name="gpu_cores", minimum=0, maximum=MAX_HW_GPU_CORES
        ),
        "ram_gb": lambda value: _validate_hw_number(
            value, field_name="ram_gb", maximum=MAX_HW_RAM_GB
        ),
        "memory_bandwidth_gbps": lambda value: _validate_hw_number(
            value,
            field_name="memory_bandwidth_gbps",
            maximum=MAX_HW_MEMORY_BANDWIDTH_GBPS,
        ),
        "chip_name": _validate_hw_chip_name,
        "has_ane": lambda value: _validate_hw_bool(value, field_name="has_ane"),
        "mps_available": lambda value: _validate_hw_bool(
            value, field_name="mps_available"
        ),
        "mlx_available": lambda value: _validate_hw_bool(
            value, field_name="mlx_available"
        ),
        "data_port": lambda value: _validate_hw_int(
            value, field_name="data_port", minimum=0, maximum=65_535
        ),
    }
    return {
        key: validators[key](value)
        for key, value in payload.items()
        if key in validators
    }


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
        """Validate and encode stable standards-compliant JSON for signing."""
        payload = _validate_hw_payload(asdict(self))
        return json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "HardwareExchange":
        """Parse JSON bytes back into a HardwareExchange.

        Raises HandshakeHwValidationError on malformed input so the handshake
        path can reject without leaking JSON errors upstream.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            raise HandshakeHwValidationError(f"HW payload deserialization failed: {e}") from e
        if not isinstance(payload, dict):
            raise HandshakeHwValidationError("HW payload not a JSON object")
        return cls(**_validate_hw_payload(payload))


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
            try:
                msg = await asyncio.wait_for(
                    WireMessage.read_from_stream(self.reader),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # A cancelled readexactly may have consumed a partial frame,
                # leaving the StreamReader mid-message. Invalidate the
                # connection rather than reusing a corrupt stream.
                self.writer.close()
                try:
                    await self.writer.wait_closed()
                except (OSError, asyncio.TimeoutError):
                    pass
                raise
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

    Handshake (v3, since v2.3) when `security.fleet_key` is set:

        1. Client sends CONTROL{HANDSHAKE_V2} with
           `local_id || challenge_a || hello_proof` where `hello_proof =
           HMAC(key, label_C1 || local_id || ':' || challenge_a || binding)`
           and `binding` is SHA-256 of the server's TLS cert as seen by the
           client. The client proves token knowledge FIRST.
        2. Server verifies `hello_proof` — on failure it closes without
           sending a byte (no HMAC oracle, no HW disclosure to
           unauthenticated peers). On success it sends ACK:
           `local_id || response_a || challenge_b || hw_block_s` with
           `response_a` domain-labeled and channel-bound, and `hw_block_s`
           the HMAC-signed HW exchange bound to `challenge_a`.
        3. Client verifies `response_a` (channel binding defeats TLS-relay
           MITM) and `hw_block_s`, then sends RESP:
           `response_b || hw_block_c` bound to `challenge_b`.
        4. Server verifies `response_b` and `hw_block_c`.

    Compatibility:
        - Secure fleets require every node on the same MacFleet version
          (>= 2.3). Pre-v2.3 secure clients are rejected (their hello has
          no proof — answering it would reopen the brute-force oracle).
        - Open fleet (no fleet_key) → handshake is just `local_id` ↔
          `local_id` unchanged. No HW exchange (nothing to HMAC-sign with).
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
        # SHA-256 of this server's TLS cert (DER) — the channel binding mixed
        # into every handshake HMAC. Empty when TLS is off (open fleets).
        self._server_cert_binding = b""
        if self._security.tls:
            self._server_ssl_ctx, self._server_cert_binding = create_server_tls_context()

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
                    _audit_transport_auth(
                        "transport.auth_rate_limited",
                        role="server",
                        local_id=self.local_id,
                        peer_ip=peer_ip,
                    )
                    writer.close()
                    await writer.wait_closed()
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
                    _audit_transport_auth(
                        "transport.auth_failed",
                        role="server",
                        local_id=self.local_id,
                        peer_ip=peer_ip,
                        reason="non_control_handshake",
                    )
                    writer.close()
                    await writer.wait_closed()
                    return

                payload = msg.payload
                fleet_key = self._security.fleet_key

                if fleet_key:
                    # v2.3 handshake (v3): the client proves token knowledge in
                    # its FIRST message. The server sends NOTHING — no HMAC
                    # response, no HW profile — until that proof verifies.
                    # Payload: node_id || challenge_a(32) || hello_proof(32).
                    #
                    # SECURITY: Downgrade protection — reject open handshakes
                    # and pre-v2.3 secure handshakes (no proof appended). A
                    # server that answered proof-less hellos would be an
                    # offline brute-force oracle for anyone on the LAN.
                    if len(payload) < 2 * CHALLENGE_SIZE + 1:
                        logger.warning(
                            "Auth handshake: payload too short from %s "
                            "(open/pre-v2.3 client, downgrade attack, or "
                            "misconfigured peer)",
                            peer_ip,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        _audit_transport_auth(
                            "transport.auth_failed",
                            role="server",
                            local_id=self.local_id,
                            peer_ip=peer_ip,
                            reason="secure_payload_too_short",
                        )
                        writer.close()
                        await writer.wait_closed()
                        return
                    if not (msg.flags & MessageFlags.HANDSHAKE_V2):
                        logger.warning(
                            "Auth handshake: client %s did not send the v2+ "
                            "handshake flag. Secure fleets require all nodes "
                            "on the same MacFleet version (>= 2.3).",
                            peer_ip,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        _audit_transport_auth(
                            "transport.auth_failed",
                            role="server",
                            local_id=self.local_id,
                            peer_ip=peer_ip,
                            reason="missing_v2_flag",
                        )
                        writer.close()
                        await writer.wait_closed()
                        return
                    hello_proof = payload[-CHALLENGE_SIZE:]
                    challenge_a = payload[-2 * CHALLENGE_SIZE:-CHALLENGE_SIZE]
                    peer_id_bytes = payload[:-2 * CHALLENGE_SIZE]
                    if len(peer_id_bytes) > MAX_NODE_ID_BYTES:
                        logger.warning(
                            "Auth handshake: peer_id too long from %s "
                            "(%d bytes, max %d)",
                            peer_ip, len(peer_id_bytes), MAX_NODE_ID_BYTES,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        _audit_transport_auth(
                            "transport.auth_failed",
                            role="server",
                            local_id=self.local_id,
                            peer_ip=peer_ip,
                            reason="peer_id_too_long",
                        )
                        writer.close()
                        await writer.wait_closed()
                        return
                    peer_id = peer_id_bytes.decode("utf-8")

                    # Verify the client knows the token BEFORE revealing
                    # anything. The proof is bound to this server's TLS cert,
                    # so a MITM relaying a hello it obtained on its own TLS
                    # leg fails here.
                    if not verify_client_hello_proof(
                        fleet_key, peer_id, challenge_a, hello_proof,
                        self._server_cert_binding,
                    ):
                        logger.warning(
                            "Auth handshake: hello proof from %s (%s) invalid "
                            "(wrong token, MITM, or version mismatch)",
                            peer_id, peer_ip,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        _audit_transport_auth(
                            "transport.auth_failed",
                            role="server",
                            local_id=self.local_id,
                            peer_id=peer_id,
                            peer_ip=peer_ip,
                            reason="invalid_client_hello_proof",
                        )
                        writer.close()
                        await writer.wait_closed()
                        return

                    # Client is authentic — now prove ourselves and challenge
                    # them. Both digests are channel-bound and domain-labeled.
                    response_a = compute_response(
                        fleet_key, challenge_a,
                        label=HS_LABEL_SERVER_RESP,
                        channel_binding=self._server_cert_binding,
                    )
                    challenge_b = generate_challenge()
                    ack_payload = (
                        self.local_id.encode("utf-8") + response_a + challenge_b
                        + _pack_hw_suffix(
                            fleet_key, self.local_id, self._local_hw, challenge_a,
                        )
                    )
                    ack = WireMessage(
                        stream_id=0,
                        msg_type=MessageType.CONTROL,
                        flags=MessageFlags.HANDSHAKE_V2,
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
                        _audit_transport_auth(
                            "transport.auth_failed",
                            role="server",
                            local_id=self.local_id,
                            peer_id=peer_id,
                            peer_ip=peer_ip,
                            reason="invalid_hw_suffix",
                        )
                        writer.close()
                        await writer.wait_closed()
                        return
                    response_b = base_resp
                    if len(response_b) != CHALLENGE_SIZE:
                        logger.warning(
                            "Auth handshake: RESP response_b wrong size from %s "
                            "(got %d, expected %d)",
                            peer_id, len(response_b), CHALLENGE_SIZE,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        _audit_transport_auth(
                            "transport.auth_failed",
                            role="server",
                            local_id=self.local_id,
                            peer_id=peer_id,
                            peer_ip=peer_ip,
                            reason="invalid_client_response_size",
                        )
                        writer.close()
                        await writer.wait_closed()
                        return
                    if not verify_response(
                        fleet_key, challenge_b, response_b,
                        label=HS_LABEL_CLIENT_RESP,
                        channel_binding=self._server_cert_binding,
                    ):
                        logger.warning(
                            "Auth handshake: peer %s (%s) failed challenge "
                            "(wrong token or attack)",
                            peer_id, peer_ip,
                        )
                        self._rate_limiter.record_failure(peer_ip)
                        _audit_transport_auth(
                            "transport.auth_failed",
                            role="server",
                            local_id=self.local_id,
                            peer_id=peer_id,
                            peer_ip=peer_ip,
                            reason="invalid_client_challenge_response",
                        )
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
                    if len(payload) > MAX_NODE_ID_BYTES:
                        logger.warning(
                            "Open handshake: payload suspiciously large from %s "
                            "(possible auth handshake sent to open server)",
                            peer_ip,
                        )
                        _audit_transport_auth(
                            "transport.auth_failed",
                            role="server",
                            local_id=self.local_id,
                            peer_ip=peer_ip,
                            reason="auth_payload_to_open_server",
                        )
                        writer.close()
                        await writer.wait_closed()
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
                _audit_transport_auth(
                    "transport.handshake_error",
                    role="server",
                    local_id=self.local_id,
                    peer_ip=peer_ip,
                    error_type=type(e).__name__,
                )
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
                return

            self._tune_socket(writer, conn.link_type)

            async with self._lock:
                old_conn = self._connections.get(peer_id)
                self._connections[peer_id] = conn

            # Close any stale connection for this peer_id outside the lock to
            # avoid leaking its socket on reconnect/duplicate handshake.
            if old_conn is not None and old_conn is not conn:
                await old_conn.close()

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
            # v2.3 handshake (v3): prove token knowledge in the FIRST message,
            # bound to the server's TLS certificate. Payload:
            #   node_id || challenge_a(32) || hello_proof(32)
            # The server reveals nothing until the proof verifies.
            channel_binding = tls_channel_binding_from_writer(writer)
            if self._security.tls and not channel_binding:
                # Fail closed: secure mode mandates TLS; a missing ssl_object
                # means the channel is not what we think it is.
                await conn.close()
                _audit_transport_auth(
                    "transport.auth_failed",
                    role="client",
                    local_id=self.local_id,
                    peer_id=peer_id,
                    peer_host=host,
                    reason="missing_tls_channel",
                )
                raise ConnectionError(
                    f"Auth handshake: expected a TLS channel to {peer_id} but "
                    f"none was negotiated"
                )

            challenge_a = generate_challenge()
            hello_proof = compute_client_hello_proof(
                fleet_key, self.local_id, challenge_a, channel_binding,
            )
            handshake_payload = (
                self.local_id.encode("utf-8") + challenge_a + hello_proof
            )

            handshake = WireMessage(
                stream_id=0,
                msg_type=MessageType.CONTROL,
                flags=MessageFlags.HANDSHAKE_V2,
                sequence=0,
                payload=handshake_payload,
            )
            await conn.send_message(handshake)

            # Read ack. The server ACK is:
            #   base: peer_id + response_a(32) + challenge_b(32)
            #   then: hw_block = wire_version(1) + hw_len(2) + hw_json + hmac(32)
            # A server that finds our hello proof invalid closes WITHOUT
            # replying (it must not act as an HMAC oracle), so EOF here
            # almost always means a token mismatch.
            try:
                ack = await asyncio.wait_for(
                    conn.recv_message(),
                    timeout=self.config.connect_timeout_sec,
                )
            except (asyncio.IncompleteReadError, ConnectionResetError, EOFError) as e:
                await conn.close()
                _audit_transport_auth(
                    "transport.auth_failed",
                    role="client",
                    local_id=self.local_id,
                    peer_id=peer_id,
                    peer_host=host,
                    reason="server_rejected_hello",
                    error_type=type(e).__name__,
                )
                raise PeerAuthError(
                    f"Auth handshake failed: peer {peer_id} rejected our hello "
                    f"— this node likely does not have the correct token "
                    f"(secure servers stay silent to unauthenticated peers)"
                ) from e
            ack_payload = ack.payload
            if len(ack_payload) < CHALLENGE_SIZE * 2 + 1:
                await conn.close()
                _audit_transport_auth(
                    "transport.auth_failed",
                    role="client",
                    local_id=self.local_id,
                    peer_id=peer_id,
                    peer_host=host,
                    reason="server_response_too_short",
                )
                raise ConnectionError("Auth handshake: server response too short")

            if not (ack.flags & MessageFlags.HANDSHAKE_V2):
                await conn.close()
                _audit_transport_auth(
                    "transport.auth_failed",
                    role="client",
                    local_id=self.local_id,
                    peer_id=peer_id,
                    peer_host=host,
                    reason="server_missing_v2_flag",
                )
                raise PeerAuthError(
                    f"Auth handshake failed: peer {peer_id} did not complete "
                    f"the authenticated handshake (pre-2.3 version or open "
                    f"fleet — secure fleets need every node on the same "
                    f"MacFleet version)"
                )

            # Peel the HW suffix first, then verify response_a. A HW-suffix
            # HMAC failure is symptomatically identical to a wrong-token
            # failure (same fleet_key, same challenge), so surface the same
            # diagnostic; the underlying error is chained for debugging.
            try:
                base_ack, peer_hw = _peel_hw_suffix(
                    fleet_key, peer_id, ack_payload, challenge_a,
                )
            except HandshakeHwValidationError as e:
                await conn.close()
                _audit_transport_auth(
                    "transport.auth_failed",
                    role="client",
                    local_id=self.local_id,
                    peer_id=peer_id,
                    peer_host=host,
                    reason="server_hw_invalid",
                )
                raise PeerAuthError(
                    f"Auth handshake failed: peer {peer_id} does not have the correct token"
                ) from e
            if len(base_ack) < CHALLENGE_SIZE * 2 + 1:
                await conn.close()
                _audit_transport_auth(
                    "transport.auth_failed",
                    role="client",
                    local_id=self.local_id,
                    peer_id=peer_id,
                    peer_host=host,
                    reason="ack_base_too_short",
                )
                raise ConnectionError(
                    "Auth handshake v2: ACK base section too short after HW peel"
                )
            response_a = base_ack[-(CHALLENGE_SIZE * 2):-CHALLENGE_SIZE]
            challenge_b = base_ack[-CHALLENGE_SIZE:]
            conn.peer_hw = peer_hw

            # Verify server proved it knows the token over THIS TLS channel.
            # A MITM that terminated TLS on both legs shows us a different
            # cert than the real server signed against → mismatch → reject.
            if not verify_response(
                fleet_key, challenge_a, response_a,
                label=HS_LABEL_SERVER_RESP,
                channel_binding=channel_binding,
            ):
                await conn.close()
                _audit_transport_auth(
                    "transport.auth_failed",
                    role="client",
                    local_id=self.local_id,
                    peer_id=peer_id,
                    peer_host=host,
                    reason="invalid_server_challenge_response",
                )
                raise PeerAuthError(
                    f"Auth handshake failed: peer {peer_id} does not have the correct token"
                )

            # Respond to server's challenge (prove we know the token too),
            # with a HW suffix bound to the server's challenge_b.
            response_b = compute_response(
                fleet_key, challenge_b,
                label=HS_LABEL_CLIENT_RESP,
                channel_binding=channel_binding,
            )
            resp_payload = response_b + _pack_hw_suffix(
                fleet_key, self.local_id, self._local_hw, challenge_b,
            )
            resp_msg = WireMessage(
                stream_id=0,
                msg_type=MessageType.CONTROL,
                flags=MessageFlags.HANDSHAKE_V2,
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
