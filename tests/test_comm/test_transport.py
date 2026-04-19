"""Tests for the adaptive TCP transport layer."""

import asyncio

import pytest

from macfleet.comm.transport import (
    PeerTransport,
    TransportConfig,
)
from macfleet.pool.network import LinkType


class TestTransportConfig:
    def test_default_values(self):
        cfg = TransportConfig()
        assert cfg.recv_timeout_sec == 120.0
        assert cfg.connect_timeout_sec == 10.0

    def test_buffer_size_for_link(self):
        cfg = TransportConfig()
        assert cfg.buffer_size_for_link(LinkType.THUNDERBOLT) == 4_194_304
        assert cfg.buffer_size_for_link(LinkType.ETHERNET) == 2_097_152
        assert cfg.buffer_size_for_link(LinkType.WIFI) == 1_048_576
        assert cfg.buffer_size_for_link(LinkType.UNKNOWN) == 1_048_576


class TestPeerTransport:
    """Integration tests using loopback TCP connections."""

    @pytest.fixture
    def config(self):
        return TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=5.0)

    @pytest.mark.asyncio
    async def test_connect_and_handshake(self, config):
        """Two transports connect via loopback and complete handshake."""
        server_transport = PeerTransport(local_id="server", config=config)
        client_transport = PeerTransport(local_id="client", config=config)

        await server_transport.start_server("127.0.0.1", 0)
        port = server_transport._server.sockets[0].getsockname()[1]

        try:
            await client_transport.connect("server", "127.0.0.1", port)
            # Give server time to process the handshake
            await asyncio.sleep(0.1)

            assert "server" in client_transport.peer_ids
            assert "client" in server_transport.peer_ids
        finally:
            await client_transport.disconnect_all()
            await server_transport.disconnect_all()

    @pytest.mark.asyncio
    async def test_send_recv_bytes(self, config):
        """Send and receive raw bytes between two peers."""
        server = PeerTransport(local_id="server", config=config)
        client = PeerTransport(local_id="client", config=config)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        try:
            await client.connect("server", "127.0.0.1", port)
            await asyncio.sleep(0.1)

            # Client sends to server
            payload = b"hello from client"
            await client.send("server", payload)
            received = await server.recv("client")
            assert received == payload

            # Server sends to client
            payload2 = b"hello from server"
            await server.send("client", payload2)
            received2 = await client.recv("server")
            assert received2 == payload2
        finally:
            await client.disconnect_all()
            await server.disconnect_all()

    @pytest.mark.asyncio
    async def test_large_payload(self, config):
        """Send a large payload (1 MB) to verify chunked transfer works."""
        server = PeerTransport(local_id="server", config=config)
        client = PeerTransport(local_id="client", config=config)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        try:
            await client.connect("server", "127.0.0.1", port)
            await asyncio.sleep(0.1)

            payload = bytes(range(256)) * 4096  # 1 MB
            await client.send("server", payload)
            received = await server.recv("client")
            assert received == payload
        finally:
            await client.disconnect_all()
            await server.disconnect_all()

    @pytest.mark.asyncio
    async def test_concurrent_send_recv(self, config):
        """Both peers send simultaneously (like allreduce direct exchange)."""
        a = PeerTransport(local_id="node-a", config=config)
        b = PeerTransport(local_id="node-b", config=config)

        await b.start_server("127.0.0.1", 0)
        port = b._server.sockets[0].getsockname()[1]

        try:
            await a.connect("node-b", "127.0.0.1", port)
            await asyncio.sleep(0.1)

            payload_a = b"data from a"
            payload_b = b"data from b"

            async def a_sends():
                await a.send("node-b", payload_a)

            async def b_sends():
                await b.send("node-a", payload_b)

            async def a_recvs():
                return await a.recv("node-b")

            async def b_recvs():
                return await b.recv("node-a")

            # Both send and receive concurrently
            _, _, recv_a, recv_b = await asyncio.gather(
                a_sends(), b_sends(), a_recvs(), b_recvs()
            )

            assert recv_a == payload_b
            assert recv_b == payload_a
        finally:
            await a.disconnect_all()
            await b.disconnect_all()

    @pytest.mark.asyncio
    async def test_disconnect(self, config):
        """Disconnecting removes the peer from the connection list."""
        server = PeerTransport(local_id="server", config=config)
        client = PeerTransport(local_id="client", config=config)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        try:
            await client.connect("server", "127.0.0.1", port)
            await asyncio.sleep(0.1)
            assert client.connection_count == 1

            await client.disconnect("server")
            assert client.connection_count == 0
        finally:
            await server.disconnect_all()

    @pytest.mark.asyncio
    async def test_send_to_unknown_peer_raises(self, config):
        """Sending to an unconnected peer raises ConnectionError."""
        transport = PeerTransport(local_id="solo", config=config)
        with pytest.raises(ConnectionError):
            await transport.send("nonexistent", b"data")

    @pytest.mark.asyncio
    async def test_recv_from_unknown_peer_raises(self, config):
        """Receiving from an unconnected peer raises ConnectionError."""
        transport = PeerTransport(local_id="solo", config=config)
        with pytest.raises(ConnectionError):
            await transport.recv("nonexistent")

    @pytest.mark.asyncio
    async def test_on_connect_callback(self, config):
        """The on_connect callback fires when a peer connects."""
        connected_peers = []

        def on_connect(peer_id, conn):
            connected_peers.append(peer_id)

        server = PeerTransport(local_id="server", config=config)
        client = PeerTransport(local_id="client", config=config)

        await server.start_server("127.0.0.1", 0, on_connect=on_connect)
        port = server._server.sockets[0].getsockname()[1]

        try:
            await client.connect("server", "127.0.0.1", port)
            await asyncio.sleep(0.1)

            assert "client" in connected_peers
        finally:
            await client.disconnect_all()
            await server.disconnect_all()

    @pytest.mark.asyncio
    async def test_multiple_peers(self, config):
        """A server can handle connections from multiple clients."""
        server = PeerTransport(local_id="server", config=config)
        client1 = PeerTransport(local_id="client1", config=config)
        client2 = PeerTransport(local_id="client2", config=config)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        try:
            await client1.connect("server", "127.0.0.1", port)
            await client2.connect("server", "127.0.0.1", port)
            await asyncio.sleep(0.1)

            assert server.connection_count == 2
            assert "client1" in server.peer_ids
            assert "client2" in server.peer_ids

            # Each client can communicate independently
            await client1.send("server", b"from-1")
            await client2.send("server", b"from-2")

            r1 = await server.recv("client1")
            r2 = await server.recv("client2")
            assert r1 == b"from-1"
            assert r2 == b"from-2"
        finally:
            await client1.disconnect_all()
            await client2.disconnect_all()
            await server.disconnect_all()

    @pytest.mark.asyncio
    async def test_bytes_tracking(self, config):
        """PeerConnection tracks bytes sent and received."""
        server = PeerTransport(local_id="server", config=config)
        client = PeerTransport(local_id="client", config=config)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        try:
            await client.connect("server", "127.0.0.1", port)
            await asyncio.sleep(0.1)

            payload = b"x" * 1000
            await client.send("server", payload)
            await server.recv("client")

            client_conn = client.get_connection("server")
            assert client_conn.bytes_sent > 1000  # payload + headers
        finally:
            await client.disconnect_all()
            await server.disconnect_all()
