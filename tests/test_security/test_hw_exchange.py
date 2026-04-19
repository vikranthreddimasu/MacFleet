"""Tests for v2.2 PR 4 — signed HW-profile handshake exchange.

Covers Issue 2 (election fix), A5 (replay protection via challenge-bound HMAC),
and A7 (wire_version byte in the payload).
"""

from __future__ import annotations

import asyncio

import pytest

from macfleet.comm.protocol import MessageFlags, MessageType, WireMessage
from macfleet.comm.transport import (
    HardwareExchange,
    PeerTransport,
    TransportConfig,
    _pack_hw_suffix,
    _peel_hw_suffix,
)
from macfleet.security.auth import (
    CHALLENGE_SIZE,
    HW_HANDSHAKE_MAX_JSON_BYTES,
    HW_HANDSHAKE_WIRE_VERSION,
    HandshakeHwValidationError,
    SecurityConfig,
    generate_challenge,
    sign_hw_profile,
    verify_hw_profile,
)

CONFIG = TransportConfig(recv_timeout_sec=5.0, connect_timeout_sec=5.0)


class TestHardwareExchangeDataclass:
    def test_default_wire_version(self):
        hw = HardwareExchange()
        assert hw.wire_version == HW_HANDSHAKE_WIRE_VERSION

    def test_roundtrip_json(self):
        hw = HardwareExchange(
            gpu_cores=40, ram_gb=192.0, memory_bandwidth_gbps=800.0,
            chip_name="Apple M4 Ultra", has_ane=True,
            mps_available=True, mlx_available=True, data_port=50052,
        )
        data = hw.to_json_bytes()
        hw2 = HardwareExchange.from_json_bytes(data)
        assert hw2 == hw

    def test_json_is_stable_sorted(self):
        """sort_keys=True → identical HW → identical bytes → deterministic HMAC."""
        hw = HardwareExchange(gpu_cores=10, ram_gb=16.0)
        assert hw.to_json_bytes() == hw.to_json_bytes()

    def test_from_json_bytes_ignores_unknown_fields(self):
        """Forward compat: future versions can add fields without breaking today."""
        hw = HardwareExchange.from_json_bytes(
            b'{"wire_version":1,"gpu_cores":8,"future_field":"ignored"}'
        )
        assert hw.gpu_cores == 8

    def test_from_json_bytes_rejects_non_object(self):
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'[1,2,3]')

    def test_from_json_bytes_rejects_invalid_utf8(self):
        with pytest.raises(HandshakeHwValidationError):
            HardwareExchange.from_json_bytes(b'\xff\xfe\xfd')


class TestSignVerifyHwProfile:
    def _key(self) -> bytes:
        return SecurityConfig(token="shared-secret-token-long").fleet_key

    def test_sign_verify_roundtrip(self):
        key = self._key()
        challenge = generate_challenge()
        hw_json = b'{"gpu_cores":16}'
        sig = sign_hw_profile(key, 1, challenge, "node-a", hw_json)
        assert verify_hw_profile(key, 1, challenge, "node-a", hw_json, sig)

    def test_wrong_key_rejected(self):
        key_a = self._key()
        key_b = SecurityConfig(token="other-token-also-long").fleet_key
        challenge = generate_challenge()
        sig = sign_hw_profile(key_a, 1, challenge, "node-a", b'{}')
        assert not verify_hw_profile(key_b, 1, challenge, "node-a", b'{}', sig)

    def test_replay_with_different_challenge_rejected(self):
        """A5: signatures are bound to the challenge; replay from another session fails."""
        key = self._key()
        ch_a = generate_challenge()
        ch_b = generate_challenge()
        sig = sign_hw_profile(key, 1, ch_a, "node-a", b'{}')
        assert not verify_hw_profile(key, 1, ch_b, "node-a", b'{}', sig)

    def test_spoofed_node_id_rejected(self):
        key = self._key()
        challenge = generate_challenge()
        sig = sign_hw_profile(key, 1, challenge, "real-node", b'{}')
        assert not verify_hw_profile(key, 1, challenge, "fake-node", b'{}', sig)

    def test_tampered_hw_json_rejected(self):
        key = self._key()
        challenge = generate_challenge()
        sig = sign_hw_profile(key, 1, challenge, "node-a", b'{"gpu_cores":8}')
        # Swap the HW — attacker claims higher score
        assert not verify_hw_profile(
            key, 1, challenge, "node-a", b'{"gpu_cores":100}', sig,
        )

    def test_wire_version_included_in_hmac(self):
        """A7: wire_version participates in the HMAC → can't strip/downgrade."""
        key = self._key()
        challenge = generate_challenge()
        sig_v1 = sign_hw_profile(key, 1, challenge, "node-a", b'{}')
        # Verify with a different wire_version fails
        assert not verify_hw_profile(key, 2, challenge, "node-a", b'{}', sig_v1)


class TestHwSuffixPackPeel:
    def _key(self) -> bytes:
        return SecurityConfig(token="fleet-token-long-enough").fleet_key

    def test_pack_peel_roundtrip(self):
        key = self._key()
        challenge = generate_challenge()
        hw = HardwareExchange(
            gpu_cores=16, ram_gb=48.0, memory_bandwidth_gbps=400.0,
            chip_name="Apple M4 Max", has_ane=True, mps_available=True,
            data_port=50052,
        )
        suffix = _pack_hw_suffix(key, "studio.local", hw, challenge)

        # Simulate the full ACK: base + suffix. base = peer_id + response + challenge
        base = b"server-node\x00" + b"A" * 32 + b"B" * 32
        payload = base + suffix

        peeled_base, peeled_hw = _peel_hw_suffix(key, "studio.local", payload, challenge)
        assert peeled_base == base
        assert peeled_hw == hw

    def test_peel_rejects_wrong_challenge(self):
        """A5 via the wire: suffix signed for challenge X fails to peel with challenge Y."""
        key = self._key()
        ch_a = generate_challenge()
        ch_b = generate_challenge()
        hw = HardwareExchange(gpu_cores=8)
        suffix = _pack_hw_suffix(key, "peer-a", hw, ch_a)
        payload = b"prefix" + suffix
        with pytest.raises(HandshakeHwValidationError, match="signature invalid"):
            _peel_hw_suffix(key, "peer-a", payload, ch_b)

    def test_peel_rejects_tampered_hw_json(self):
        key = self._key()
        challenge = generate_challenge()
        hw = HardwareExchange(gpu_cores=8)
        suffix = bytearray(_pack_hw_suffix(key, "peer-a", hw, challenge))
        # Flip a byte in the hw_json region (byte 3 is start of hw_json)
        suffix[5] ^= 0xFF
        payload = b"prefix" + bytes(suffix)
        with pytest.raises(HandshakeHwValidationError):
            _peel_hw_suffix(key, "peer-a", payload, challenge)

    def test_peel_rejects_oversize_block_size(self):
        """A malicious trailer claiming huge block_size is rejected."""
        key = self._key()
        challenge = generate_challenge()
        # Forge trailing block_size > max allowed
        import struct
        bogus = b"xxx" + struct.pack("!H", HW_HANDSHAKE_MAX_JSON_BYTES * 4)
        with pytest.raises(HandshakeHwValidationError, match="block_size"):
            _peel_hw_suffix(key, "peer-a", b"prefix" + bogus, challenge)

    def test_peel_rejects_suffix_larger_than_payload(self):
        """block_size trailer that exceeds payload length is rejected."""
        key = self._key()
        challenge = generate_challenge()
        import struct
        # block_size=100 but payload is only 20 bytes before the trailer
        bogus = b"x" * 20 + struct.pack("!H", 100)
        with pytest.raises(HandshakeHwValidationError, match="exceeds payload"):
            _peel_hw_suffix(key, "peer-a", bogus, challenge)

    def test_peel_rejects_oversize_hw_json(self):
        """Reject HW-json claiming length above HW_HANDSHAKE_MAX_JSON_BYTES.

        The trailing block_size check catches this first — it guards against
        any claim that the HW block is too large, regardless of whether
        the oversize is in block_size or the internal hw_len field.
        """
        key = self._key()
        challenge = generate_challenge()
        import struct
        wire = 1
        hw_len = HW_HANDSHAKE_MAX_JSON_BYTES + 100
        fake_hw = b"x" * hw_len
        fake_sig = b"s" * 32
        body = struct.pack("!BH", wire, hw_len) + fake_hw + fake_sig
        payload = b"prefix" + body + struct.pack("!H", len(body))
        # block_size check fires first and catches the oversize
        with pytest.raises(HandshakeHwValidationError, match="outside valid range"):
            _peel_hw_suffix(key, "peer-a", payload, challenge)


class TestTransportV2Handshake:
    """End-to-end handshake with HW exchange over loopback TCP."""

    async def test_v2_handshake_exchanges_hw_both_directions(self):
        sec = SecurityConfig(token="shared-secret-token-long")
        server_hw = HardwareExchange(
            gpu_cores=60, ram_gb=192.0, memory_bandwidth_gbps=800.0,
            chip_name="Apple M4 Ultra", has_ane=True, mps_available=True,
            data_port=50052,
        )
        client_hw = HardwareExchange(
            gpu_cores=10, ram_gb=16.0, memory_bandwidth_gbps=100.0,
            chip_name="Apple M4", has_ane=True, mps_available=True,
            data_port=50053,
        )

        server = PeerTransport(local_id="server", config=CONFIG, security=sec, local_hw=server_hw)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec, local_hw=client_hw)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        try:
            # Client sees server's HW
            client_conn = client.get_connection("server")
            assert client_conn is not None
            assert client_conn.peer_hw is not None
            assert client_conn.peer_hw.gpu_cores == 60
            assert client_conn.peer_hw.chip_name == "Apple M4 Ultra"
            assert client_conn.peer_hw.data_port == 50052

            # Server sees client's HW
            server_conn = server.get_connection("client")
            assert server_conn is not None
            assert server_conn.peer_hw is not None
            assert server_conn.peer_hw.gpu_cores == 10
            assert server_conn.peer_hw.chip_name == "Apple M4"
            assert server_conn.peer_hw.data_port == 50053
        finally:
            await client.disconnect_all()
            await server.disconnect_all()

    async def test_v2_handshake_with_default_hw_exchanges_zero_profile(self):
        """When no local_hw is passed, peers exchange default zero-filled HW."""
        sec = SecurityConfig(token="fleet-token-long-enough")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        try:
            server_conn = server.get_connection("client")
            assert server_conn.peer_hw is not None
            assert server_conn.peer_hw.gpu_cores == 0
            assert server_conn.peer_hw.chip_name == "unknown"
        finally:
            await client.disconnect_all()
            await server.disconnect_all()

    async def test_open_fleet_no_hw_exchange(self):
        """Open fleet (no token) doesn't exchange HW — nothing to HMAC-sign with."""
        server = PeerTransport(local_id="server", config=CONFIG)  # no security
        client = PeerTransport(local_id="client", config=CONFIG)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        try:
            assert server.get_connection("client").peer_hw is None
            assert client.get_connection("server").peer_hw is None
        finally:
            await client.disconnect_all()
            await server.disconnect_all()

    async def test_v1_client_v2_server_falls_back_gracefully(self):
        """A v2.1-style client (no HANDSHAKE_V2 flag) can still connect to a v2.2 server.

        The server logs a warning and skips HW exchange, but the connection
        succeeds so mixed-version fleets stay connected during upgrade rollout.

        Secure mode forces TLS, so this test has to speak TLS to the server
        while hand-rolling the v2.1-style plaintext handshake payload.
        """
        from macfleet.security.auth import compute_response, create_client_ssl_context

        sec = SecurityConfig(token="fleet-token-long-enough")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        try:
            # Hand-roll a v2.1-style client handshake: no HANDSHAKE_V2 flag.
            # Use TLS since SecurityConfig forced tls=True on the server.
            ssl_ctx = create_client_ssl_context()
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", port, ssl=ssl_ctx,
            )
            challenge_a = generate_challenge()
            hs_payload = b"legacy-client" + challenge_a
            msg = WireMessage(
                stream_id=0, msg_type=MessageType.CONTROL, flags=MessageFlags.NONE,
                sequence=0, payload=hs_payload,
            )
            writer.write(msg.pack())
            await writer.drain()

            # Server ACK (legacy format — no HW suffix)
            ack = await asyncio.wait_for(WireMessage.read_from_stream(reader), timeout=2.0)
            assert ack.flags == MessageFlags.NONE, "server should reply with v1 flags"
            # ACK base: server_id + response_a(32) + challenge_b(32)
            assert len(ack.payload) >= 64 + 1
            challenge_b = ack.payload[-CHALLENGE_SIZE:]

            # Send v2.1-style RESP: just response_b
            response_b = compute_response(sec.fleet_key, challenge_b)
            resp = WireMessage(
                stream_id=0, msg_type=MessageType.CONTROL, flags=MessageFlags.NONE,
                sequence=0, payload=response_b,
            )
            writer.write(resp.pack())
            await writer.drain()
            await asyncio.sleep(0.1)

            # Server accepts the connection without HW
            assert "legacy-client" in server.peer_ids
            assert server.get_connection("legacy-client").peer_hw is None

            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        finally:
            await server.disconnect_all()

    async def test_mismatched_tokens_still_rejected_in_v2(self):
        """v2.2 handshake still verifies the underlying HMAC challenge-response."""
        sec_a = SecurityConfig(token="token-a-long-enough")
        sec_b = SecurityConfig(token="token-b-long-enough")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec_a)
        client = PeerTransport(local_id="client", config=CONFIG, security=sec_b)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]

        with pytest.raises((ConnectionError, Exception)):
            await client.connect("server", "127.0.0.1", port)

        await server.disconnect_all()

    async def test_set_local_hw_updates_profile_for_future_connects(self):
        """PoolAgent uses set_local_hw() after profiling hardware at startup."""
        sec = SecurityConfig(token="fleet-token-long-enough")
        server = PeerTransport(local_id="server", config=CONFIG, security=sec)
        server.set_local_hw(HardwareExchange(gpu_cores=99, chip_name="Patched"))

        client = PeerTransport(local_id="client", config=CONFIG, security=sec)
        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("server", "127.0.0.1", port)
        await asyncio.sleep(0.05)

        try:
            peer_hw = client.get_connection("server").peer_hw
            assert peer_hw.gpu_cores == 99
            assert peer_hw.chip_name == "Patched"
        finally:
            await client.disconnect_all()
            await server.disconnect_all()
