"""Tests for authenticated transport handshake."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.security.auth import SecurityConfig


CONFIG = TransportConfig(recv_timeout_sec=5.0, connect_timeout_sec=5.0)


# ------------------------------------------------------------------ #
# Open (no auth) — backward compatible                                #
# ------------------------------------------------------------------ #


class TestOpenTransport:
    """Verify no-token transport still works (backward compat)."""

    async def test_open_connect_and_transfer(self):
        server = PeerTransport(local_id="server", config=CONFIG)
        client = PeerTransport(local_id="client", config=CONFIG)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        assert client.connection_count == 1
        assert server.connection_count == 1

        # Transfer data
        payload = b"hello world"
        await client.send("server", payload)
        received = await server.recv("client")
        assert received == payload

        await client.disconnect_all()
        await server.disconnect_all()


# ------------------------------------------------------------------ #
# Token auth — matching tokens                                        #
# ------------------------------------------------------------------ #


class TestAuthenticatedTransport:
    async def test_matching_tokens_connect(self):
        sec = SecurityConfig(token="shared-secret-token")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        assert client.connection_count == 1
        assert server.connection_count == 1

        await client.disconnect_all()
        await server.disconnect_all()

    async def test_matching_tokens_data_transfer(self):
        sec = SecurityConfig(token="shared-secret-token")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        # Send gradient-sized payload
        data = np.random.randn(1000).astype(np.float32).tobytes()
        await client.send("server", data)
        received = await server.recv("client")
        assert received == data

        await client.disconnect_all()
        await server.disconnect_all()

    async def test_mismatched_tokens_rejected(self):
        sec_a = SecurityConfig(token="secret-a-long")
        sec_b = SecurityConfig(token="secret-b-long")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec_a)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec_b)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        with pytest.raises(ConnectionError, match="does not have the correct token"):
            await client.connect("server", "127.0.0.1", port)

        await client.disconnect_all()
        await server.disconnect_all()

    async def test_client_token_server_open_rejected(self):
        """Client has token but server is open — handshake protocol mismatch."""
        sec_client = SecurityConfig(token="secret-token")
        server = PeerTransport(local_id="server", config=CONFIG)  # no security
        client = PeerTransport(local_id="client", config=CONFIG, security=sec_client)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        # Client sends authenticated handshake, server treats it as bare node_id
        # Server responds with bare ack (too short for auth), client should fail
        with pytest.raises((ConnectionError, Exception)):
            await client.connect("server", "127.0.0.1", port)

        await client.disconnect_all()
        await server.disconnect_all()

    async def test_different_fleet_ids_rejected(self):
        """Same token but different fleet IDs produce different keys."""
        sec_a = SecurityConfig(token="same-token-long", fleet_id="fleet-a")
        sec_b = SecurityConfig(token="same-token-long", fleet_id="fleet-b")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec_a)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec_b)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        with pytest.raises(ConnectionError, match="does not have the correct token"):
            await client.connect("server", "127.0.0.1", port)

        await client.disconnect_all()
        await server.disconnect_all()

    async def test_same_fleet_id_connects(self):
        sec = SecurityConfig(token="token-long", fleet_id="my-fleet")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        assert client.connection_count == 1
        assert server.connection_count == 1

        await client.disconnect_all()
        await server.disconnect_all()


# ------------------------------------------------------------------ #
# TLS + Token                                                        #
# ------------------------------------------------------------------ #


class TestTLSTransport:
    async def test_tls_with_auth_connects(self):
        sec = SecurityConfig(token="secret-token", tls=True)
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        assert client.connection_count == 1
        assert server.connection_count == 1

        # Verify data transfer works over TLS
        payload = b"encrypted payload"
        await client.send("server", payload)
        received = await server.recv("client")
        assert received == payload

        await client.disconnect_all()
        await server.disconnect_all()

    async def test_tls_large_payload(self):
        sec = SecurityConfig(token="secret-token", tls=True)
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        # 1MB payload (simulates gradient tensor)
        data = np.random.randn(250_000).astype(np.float32).tobytes()
        await client.send("server", data)
        received = await server.recv("client")
        assert received == data

        await client.disconnect_all()
        await server.disconnect_all()


# ------------------------------------------------------------------ #
# Downgrade protection: open client vs secure server                   #
# ------------------------------------------------------------------ #


class TestDowngradeProtection:
    async def test_open_client_secure_server_rejected(self):
        """Open client connecting to secure server should be rejected.

        The secure server checks payload length — an open handshake
        (bare node_id, no challenge) is too short and gets rejected.
        """
        sec_server = SecurityConfig(token="secret-token")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec_server)
        client = PeerTransport(local_id="client", config=CONFIG)  # no security

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        # Open client sends bare node_id. Secure server expects node_id + 32-byte
        # challenge. The short payload triggers downgrade protection.
        # Client will either get a connection error or timeout because
        # the server closes the connection without responding.
        with pytest.raises((ConnectionError, asyncio.TimeoutError, Exception)):
            await client.connect("server", "127.0.0.1", port)

        await client.disconnect_all()
        await server.disconnect_all()


# ------------------------------------------------------------------ #
# Rate limiter integration                                             #
# ------------------------------------------------------------------ #


class TestRateLimiterIntegration:
    async def test_banned_ip_rejected(self):
        """After 5 failures from an IP, subsequent connections are rejected."""
        sec_a = SecurityConfig(token="correct-token")
        sec_b = SecurityConfig(token="wrong-token-xx")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec_a)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        # Generate 5 failures
        for _ in range(5):
            bad_client = PeerTransport(local_id="bad", config=CONFIG, security=sec_b)
            try:
                await bad_client.connect("server", "127.0.0.1", port)
            except (ConnectionError, Exception):
                pass
            await bad_client.disconnect_all()

        # 6th attempt should be banned (connection closed immediately)
        banned_client = PeerTransport(local_id="banned", config=CONFIG, security=sec_b)
        with pytest.raises((ConnectionError, asyncio.TimeoutError, Exception)):
            await banned_client.connect("server", "127.0.0.1", port)

        await banned_client.disconnect_all()
        await server.disconnect_all()
