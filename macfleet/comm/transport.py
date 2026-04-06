"""Adaptive TCP transport for MacFleet v2.

Manages per-peer TCP connections with:
- Handshake protocol for peer identification
- Adaptive buffer sizes based on link type (WiFi/Ethernet/TB4)
- Per-connection send/recv locks for safe concurrent operations
- WireMessage protocol with CRC32 verification
"""

from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from macfleet.comm.protocol import (
    HEADER_SIZE,
    MessageFlags,
    MessageType,
    WireMessage,
)
from macfleet.pool.network import LinkType


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

    Each peer gets one bidirectional TCP connection. The handshake
    protocol exchanges node IDs so both sides can identify who
    they're talking to.

    Handshake (initiator):
        1. Open TCP connection
        2. Send CONTROL WireMessage with local_id as payload
        3. Receive CONTROL ack with remote's id

    Handshake (acceptor):
        1. Receive CONTROL WireMessage with remote's id
        2. Send CONTROL ack with local_id as payload
        3. Register connection under remote's id
    """

    def __init__(
        self,
        local_id: str,
        config: Optional[TransportConfig] = None,
    ):
        self.local_id = local_id
        self.config = config or TransportConfig()
        self._connections: dict[str, PeerConnection] = {}
        self._server: Optional[asyncio.Server] = None
        self._lock = asyncio.Lock()
        self._on_connect: Optional[Callable] = None

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
            try:
                # Read handshake: peer sends its ID
                msg = await asyncio.wait_for(
                    WireMessage.read_from_stream(reader),
                    timeout=self.config.connect_timeout_sec,
                )
                if msg.msg_type != MessageType.CONTROL:
                    writer.close()
                    return
                peer_id = msg.payload.decode("utf-8")
            except Exception:
                try:
                    writer.close()
                except Exception:
                    pass
                return

            conn = PeerConnection(peer_id=peer_id, reader=reader, writer=writer)
            self._tune_socket(writer, conn.link_type)

            # Send ack with our local_id
            ack = WireMessage(
                stream_id=0,
                msg_type=MessageType.CONTROL,
                flags=MessageFlags.NONE,
                sequence=0,
                payload=self.local_id.encode("utf-8"),
            )
            await conn.send_message(ack)

            async with self._lock:
                self._connections[peer_id] = conn

            if self._on_connect:
                self._on_connect(peer_id, conn)

        self._server = await asyncio.start_server(handle_client, host, port)

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
        """
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=self.config.connect_timeout_sec,
        )

        conn = PeerConnection(
            peer_id=peer_id,
            reader=reader,
            writer=writer,
            link_type=link_type,
        )
        self._tune_socket(writer, link_type)

        # Send handshake with our local_id
        handshake = WireMessage(
            stream_id=0,
            msg_type=MessageType.CONTROL,
            flags=MessageFlags.NONE,
            sequence=0,
            payload=self.local_id.encode("utf-8"),
        )
        await conn.send_message(handshake)

        # Wait for ack
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
