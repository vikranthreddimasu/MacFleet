"""Tests for the v2.3 handshake hardening (handshake v3).

Threats closed and verified here:

1. Unauthenticated HMAC oracle — a secure server used to answer ANY
   connector with HMAC(fleet_key, attacker_chosen_challenge) plus its
   signed hardware profile. Now the client proves token knowledge in its
   FIRST message and the server stays byte-silent otherwise.
2. TLS MITM relay — handshake HMACs are bound to the server's TLS cert
   fingerprint, so an attacker terminating TLS on both legs produces
   mismatched digests.
3. Offline dictionary attacks — fleet keys derive via scrypt
   (memory-hard) instead of one HMAC-SHA256.
4. Heartbeat APONG replay — response signatures are bound to the
   request nonce.
"""

from __future__ import annotations

import asyncio
import secrets

import pytest

from macfleet.comm.protocol import MessageFlags, MessageType, WireMessage
from macfleet.comm.transport import PeerTransport, TransportConfig
from macfleet.security.auth import (
    HS_LABEL_CLIENT_RESP,
    HS_LABEL_SERVER_RESP,
    SecurityConfig,
    compute_client_hello_proof,
    compute_response,
    create_client_ssl_context,
    create_server_tls_context,
    generate_challenge,
    sign_heartbeat_response,
    verify_client_hello_proof,
    verify_heartbeat_response,
    verify_response,
)

CONFIG = TransportConfig(recv_timeout_sec=5.0, connect_timeout_sec=5.0)


# ------------------------------------------------------------------ #
# 1. The oracle is closed                                             #
# ------------------------------------------------------------------ #


class TestOracleClosed:
    async def test_server_silent_to_bad_hello_proof(self):
        """A v3-shaped hello with a garbage proof gets ZERO bytes back."""
        sec = SecurityConfig(token="fleet-token-long-enough")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        try:
            ssl_ctx = create_client_ssl_context()
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", port, ssl=ssl_ctx,
            )
            # Correct shape, wrong proof: attacker without the token.
            payload = (
                b"attacker" + generate_challenge() + secrets.token_bytes(32)
            )
            msg = WireMessage(
                stream_id=0, msg_type=MessageType.CONTROL,
                flags=MessageFlags.HANDSHAKE_V2, sequence=0, payload=payload,
            )
            writer.write(msg.pack())
            await writer.drain()

            data = await asyncio.wait_for(reader.read(4096), timeout=3.0)
            assert data == b"", (
                "server must not reveal an HMAC response or HW profile to an "
                "unauthenticated connector"
            )
            assert "attacker" not in server.peer_ids
            # The failed proof counts against the rate limiter.
            assert server._rate_limiter.get_delay("127.0.0.1") > 0

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        finally:
            await server.disconnect_all()

    async def test_matching_tokens_still_connect_and_transfer(self):
        """Sanity: the hardened handshake still works for honest peers."""
        sec = SecurityConfig(token="fleet-token-long-enough")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        try:
            await client.send("server", b"gradient bytes")
            assert await server.recv("client") == b"gradient bytes"
            # HW exchange survived the hardening (always-on in v3)
            assert server.get_connection("client").peer_hw is not None
            assert client.get_connection("server").peer_hw is not None
        finally:
            await client.disconnect_all()
            await server.disconnect_all()


# ------------------------------------------------------------------ #
# 2. Channel binding (MITM relay defeat)                              #
# ------------------------------------------------------------------ #


class TestChannelBinding:
    def test_hello_proof_bound_to_channel(self):
        """A proof computed against cert A does not verify against cert B —
        exactly what a TLS-terminating relay would produce."""
        key = SecurityConfig(token="fleet-token-long-enough").fleet_key
        challenge = generate_challenge()
        _, cert_a = create_server_tls_context()
        _, cert_b = create_server_tls_context()
        assert cert_a != cert_b  # fresh cert per server process

        proof = compute_client_hello_proof(key, "node-1", challenge, cert_a)
        assert verify_client_hello_proof(key, "node-1", challenge, proof, cert_a)
        assert not verify_client_hello_proof(key, "node-1", challenge, proof, cert_b)

    def test_response_bound_to_channel(self):
        key = SecurityConfig(token="fleet-token-long-enough").fleet_key
        challenge = generate_challenge()
        _, cert_a = create_server_tls_context()
        _, cert_b = create_server_tls_context()

        resp = compute_response(
            key, challenge, label=HS_LABEL_SERVER_RESP, channel_binding=cert_a,
        )
        assert verify_response(
            key, challenge, resp,
            label=HS_LABEL_SERVER_RESP, channel_binding=cert_a,
        )
        assert not verify_response(
            key, challenge, resp,
            label=HS_LABEL_SERVER_RESP, channel_binding=cert_b,
        )

    def test_domain_separation_between_steps(self):
        """A digest from one handshake step can't stand in for another's."""
        key = SecurityConfig(token="fleet-token-long-enough").fleet_key
        challenge = generate_challenge()
        binding = secrets.token_bytes(32)

        server_resp = compute_response(
            key, challenge, label=HS_LABEL_SERVER_RESP, channel_binding=binding,
        )
        assert not verify_response(
            key, challenge, server_resp,
            label=HS_LABEL_CLIENT_RESP, channel_binding=binding,
        )

    def test_proof_bound_to_node_id(self):
        """The hello proof covers the claimed node_id (no identity swap)."""
        key = SecurityConfig(token="fleet-token-long-enough").fleet_key
        challenge = generate_challenge()
        proof = compute_client_hello_proof(key, "honest-node", challenge)
        assert not verify_client_hello_proof(key, "evil-node", challenge, proof)


# ------------------------------------------------------------------ #
# 3. scrypt key derivation                                            #
# ------------------------------------------------------------------ #


class TestScryptDerivation:
    def test_key_is_32_bytes_and_deterministic(self):
        a = SecurityConfig(token="fleet-token-long-enough")
        b = SecurityConfig(token="fleet-token-long-enough")
        assert a.fleet_key == b.fleet_key
        assert len(a.fleet_key) == 32

    def test_fleet_id_scopes_the_key(self):
        a = SecurityConfig(token="fleet-token-long-enough", fleet_id="alpha")
        b = SecurityConfig(token="fleet-token-long-enough", fleet_id="beta")
        assert a.fleet_key != b.fleet_key

    def test_short_token_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="macfleet.security.auth"):
            SecurityConfig(token="shorttok")  # 8 chars: allowed but warned
        assert any("dictionary-attackable" in r.message for r in caplog.records)

    def test_long_token_no_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="macfleet.security.auth"):
            SecurityConfig(token="this-token-is-long-enough-to-be-fine")
        assert not any(
            "dictionary-attackable" in r.message for r in caplog.records
        )


# ------------------------------------------------------------------ #
# 4. Heartbeat response binding                                       #
# ------------------------------------------------------------------ #


class TestHeartbeatResponseBinding:
    def test_response_bound_to_request_nonce(self):
        key = SecurityConfig(token="fleet-token-long-enough").fleet_key
        req_a = secrets.token_bytes(16)
        req_b = secrets.token_bytes(16)
        resp_nonce = secrets.token_bytes(16)

        sig = sign_heartbeat_response(key, "node-1", resp_nonce, req_a)
        assert verify_heartbeat_response(key, "node-1", resp_nonce, req_a, sig)
        # Replay against a different request fails.
        assert not verify_heartbeat_response(key, "node-1", resp_nonce, req_b, sig)

    def test_response_with_hw_bound_to_request_nonce(self):
        key = SecurityConfig(token="fleet-token-long-enough").fleet_key
        req_a = secrets.token_bytes(16)
        req_b = secrets.token_bytes(16)
        resp_nonce = secrets.token_bytes(16)
        hw = b'{"gpu_cores":40}'

        sig = sign_heartbeat_response(key, "node-1", resp_nonce, req_a, hw_json=hw)
        assert verify_heartbeat_response(
            key, "node-1", resp_nonce, req_a, sig, hw_json=hw,
        )
        assert not verify_heartbeat_response(
            key, "node-1", resp_nonce, req_b, sig, hw_json=hw,
        )
        # Tampered HW also fails.
        assert not verify_heartbeat_response(
            key, "node-1", resp_nonce, req_a, sig, hw_json=b'{"gpu_cores":999}',
        )


# ------------------------------------------------------------------ #
# Doctor surfaces security posture                                    #
# ------------------------------------------------------------------ #


class TestDoctorSecuritySection:
    def test_doctor_includes_security_checks(self):
        from click.testing import CliRunner

        from macfleet.cli.main import cli

        result = CliRunner().invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "Security" in result.output
        assert "Fleet token" in result.output


_ = pytest  # asyncio_mode=auto handles async tests; keep import referenced


class TestPeerIdLengthCap:
    async def test_oversized_peer_id_is_rejected_silently(self):
        """Secure servers must reject oversized node_id without oracle leak."""
        from macfleet.security.auth import MAX_NODE_ID_BYTES

        sec = SecurityConfig(token="fleet-token-long-enough")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        try:
            ssl_ctx = create_client_ssl_context()
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", port, ssl=ssl_ctx,
            )
            huge_id = b"x" * (MAX_NODE_ID_BYTES + 1)
            payload = huge_id + generate_challenge() + secrets.token_bytes(32)
            msg = WireMessage(
                stream_id=0, msg_type=MessageType.CONTROL,
                flags=MessageFlags.HANDSHAKE_V2, sequence=0, payload=payload,
            )
            writer.write(msg.pack())
            await writer.drain()

            data = await asyncio.wait_for(reader.read(4096), timeout=3.0)
            assert data == b""
            assert server.peer_ids == []
            assert server._rate_limiter.get_delay("127.0.0.1") > 0

            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        finally:
            await server.disconnect_all()
