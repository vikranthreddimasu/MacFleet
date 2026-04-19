"""Tests for authenticated heartbeat protocol."""

from __future__ import annotations

import asyncio
import secrets

from macfleet.pool.heartbeat import GossipHeartbeat
from macfleet.security.auth import SecurityConfig, sign_heartbeat, verify_heartbeat

# ------------------------------------------------------------------ #
# Heartbeat signing at the protocol level                             #
# ------------------------------------------------------------------ #


class TestHeartbeatProtocol:
    def test_sign_verify_roundtrip(self):
        fleet_key = SecurityConfig(token="secret-token").fleet_key
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(fleet_key, "node-0", nonce)
        assert verify_heartbeat(fleet_key, "node-0", nonce, sig)

    def test_different_fleet_keys_fail(self):
        key_a = SecurityConfig(token="token-aaa").fleet_key
        key_b = SecurityConfig(token="token-bbb").fleet_key
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(key_a, "node-0", nonce)
        assert not verify_heartbeat(key_b, "node-0", nonce, sig)

    def test_replay_different_nonce_fails(self):
        fleet_key = SecurityConfig(token="secret-token").fleet_key
        nonce_a = secrets.token_bytes(16)
        nonce_b = secrets.token_bytes(16)
        sig = sign_heartbeat(fleet_key, "node-0", nonce_a)
        assert not verify_heartbeat(fleet_key, "node-0", nonce_b, sig)

    def test_spoofed_node_id_fails(self):
        fleet_key = SecurityConfig(token="secret-token").fleet_key
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(fleet_key, "real-node", nonce)
        assert not verify_heartbeat(fleet_key, "fake-node", nonce, sig)


# ------------------------------------------------------------------ #
# Heartbeat integration with GossipHeartbeat                          #
# ------------------------------------------------------------------ #


class TestAuthenticatedHeartbeatServer:
    """Test the heartbeat responder handles authenticated messages."""

    async def test_authenticated_ping_pong(self):
        """Simulate authenticated APING/APONG exchange."""
        sec = SecurityConfig(token="fleet-secret-token")
        fleet_key = sec.fleet_key

        # Start a heartbeat responder (simulated inline)
        async def handle_ping(reader, writer):
            data = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if data.startswith(b"APING"):
                parts = data.decode().strip().split(" ")
                assert len(parts) == 4
                _, peer_id, nonce_hex, sig_hex = parts
                nonce = bytes.fromhex(nonce_hex)
                sig = bytes.fromhex(sig_hex)
                assert verify_heartbeat(fleet_key, peer_id, nonce, sig)

                # Respond with authenticated PONG
                resp_nonce = secrets.token_bytes(16)
                resp_sig = sign_heartbeat(fleet_key, "responder", resp_nonce)
                writer.write(f"APONG responder {resp_nonce.hex()} {resp_sig.hex()}\n".encode())
                await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle_ping, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        # Send authenticated ping
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(fleet_key, "pinger", nonce)
        writer.write(f"APING pinger {nonce.hex()} {sig.hex()}\n".encode())
        await writer.drain()

        response = await asyncio.wait_for(reader.readline(), timeout=2.0)
        assert response.startswith(b"APONG")

        # Verify APONG signature
        parts = response.decode().strip().split(" ")
        assert len(parts) == 4
        _, resp_node_id, resp_nonce_hex, resp_sig_hex = parts
        resp_nonce = bytes.fromhex(resp_nonce_hex)
        resp_sig = bytes.fromhex(resp_sig_hex)
        assert verify_heartbeat(fleet_key, resp_node_id, resp_nonce, resp_sig)

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    async def test_unauthenticated_ping_rejected_by_secure_server(self):
        """A plain PING is ignored when the server requires auth."""
        sec = SecurityConfig(token="fleet-secret-token")
        fleet_key = sec.fleet_key

        async def handle_ping(reader, writer):
            data = await asyncio.wait_for(reader.readline(), timeout=2.0)
            # Secure server only responds to APING
            if fleet_key and data.startswith(b"APING"):
                parts = data.decode().strip().split(" ")
                if len(parts) == 4:
                    _, peer_id, nonce_hex, sig_hex = parts
                    nonce = bytes.fromhex(nonce_hex)
                    sig = bytes.fromhex(sig_hex)
                    if verify_heartbeat(fleet_key, peer_id, nonce, sig):
                        resp_nonce = secrets.token_bytes(16)
                        resp_sig = sign_heartbeat(fleet_key, "responder", resp_nonce)
                        writer.write(
                            f"APONG responder {resp_nonce.hex()} {resp_sig.hex()}\n".encode()
                        )
                        await writer.drain()
            # Plain PING → no response (silent reject)
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle_ping, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"PING attacker\n")
        await writer.drain()

        # Should get no response (connection closed)
        response = await asyncio.wait_for(reader.read(1024), timeout=1.0)
        assert not response.startswith(b"PONG")

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    async def test_wrong_key_ping_rejected(self):
        """APING with wrong fleet key gets no valid response."""
        sec = SecurityConfig(token="correct-token")
        fleet_key = sec.fleet_key
        wrong_key = SecurityConfig(token="wrong-token").fleet_key

        async def handle_ping(reader, writer):
            data = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if data.startswith(b"APING"):
                parts = data.decode().strip().split(" ")
                if len(parts) == 4:
                    _, peer_id, nonce_hex, sig_hex = parts
                    nonce = bytes.fromhex(nonce_hex)
                    sig = bytes.fromhex(sig_hex)
                    if verify_heartbeat(fleet_key, peer_id, nonce, sig):
                        resp_nonce = secrets.token_bytes(16)
                        resp_sig = sign_heartbeat(fleet_key, "responder", resp_nonce)
                        writer.write(
                            f"APONG responder {resp_nonce.hex()} {resp_sig.hex()}\n".encode()
                        )
                        await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle_ping, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        # Send APING signed with wrong key
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(wrong_key, "attacker", nonce)
        writer.write(f"APING attacker {nonce.hex()} {sig.hex()}\n".encode())
        await writer.drain()

        # Server should not respond with APONG
        response = await asyncio.wait_for(reader.read(1024), timeout=1.0)
        assert not response.startswith(b"APONG")

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()


class TestGossipHeartbeatSecurity:
    def test_heartbeat_accepts_security_config(self):
        sec = SecurityConfig(token="test-token")
        hb = GossipHeartbeat(node_id="node-0", security=sec)
        assert hb._security.is_secure

    def test_heartbeat_default_no_security(self):
        hb = GossipHeartbeat(node_id="node-0")
        assert not hb._security.is_secure
