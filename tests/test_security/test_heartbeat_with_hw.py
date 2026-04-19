"""Tests for v2.2 heartbeat-with-HW protocol (Issue 6 / PR 5).

Covers:
    - sign/verify primitives over `node_id || nonce || hw_json`
    - APING v2 wire format (5 fields) accepted by agent server handler
    - APING v1 wire format (4 fields) still accepted for v2.1 compat
    - Replay-protection: re-using a nonce with different HW fails verification
    - DoS bounds: oversize HW payload rejected
    - `_add_manual_peer` registers peer with real HW pulled from APONG v2
"""

from __future__ import annotations

import asyncio
import secrets

import pytest

from macfleet.comm.transport import HardwareExchange
from macfleet.pool.agent import PoolAgent
from macfleet.security.auth import (
    HW_HANDSHAKE_MAX_JSON_BYTES,
    SecurityConfig,
    sign_heartbeat_with_hw,
    verify_heartbeat_with_hw,
)

# ------------------------------------------------------------------ #
# Primitives: sign/verify HMAC with HW payload                        #
# ------------------------------------------------------------------ #


class TestSignVerifyHeartbeatWithHw:
    def test_roundtrip(self):
        fleet_key = SecurityConfig(token="fleet-secret").fleet_key
        nonce = secrets.token_bytes(16)
        hw_json = HardwareExchange(gpu_cores=10, chip_name="M4").to_json_bytes()
        sig = sign_heartbeat_with_hw(fleet_key, "node-0", nonce, hw_json)
        assert verify_heartbeat_with_hw(fleet_key, "node-0", nonce, hw_json, sig)

    def test_wrong_key_fails(self):
        key_a = SecurityConfig(token="aaa-token").fleet_key
        key_b = SecurityConfig(token="bbb-token").fleet_key
        nonce = secrets.token_bytes(16)
        hw_json = HardwareExchange(gpu_cores=10).to_json_bytes()
        sig = sign_heartbeat_with_hw(key_a, "node-0", nonce, hw_json)
        assert not verify_heartbeat_with_hw(key_b, "node-0", nonce, hw_json, sig)

    def test_tampered_hw_fails(self):
        """Swapping HW after signing must invalidate the signature."""
        fleet_key = SecurityConfig(token="fleet-secret").fleet_key
        nonce = secrets.token_bytes(16)
        real_hw = HardwareExchange(gpu_cores=8).to_json_bytes()
        lying_hw = HardwareExchange(gpu_cores=128).to_json_bytes()  # inflate score
        sig = sign_heartbeat_with_hw(fleet_key, "node-0", nonce, real_hw)
        assert not verify_heartbeat_with_hw(fleet_key, "node-0", nonce, lying_hw, sig)

    def test_tampered_nonce_fails(self):
        fleet_key = SecurityConfig(token="fleet-secret").fleet_key
        nonce_a = secrets.token_bytes(16)
        nonce_b = secrets.token_bytes(16)
        hw_json = HardwareExchange(gpu_cores=8).to_json_bytes()
        sig = sign_heartbeat_with_hw(fleet_key, "node-0", nonce_a, hw_json)
        assert not verify_heartbeat_with_hw(fleet_key, "node-0", nonce_b, hw_json, sig)

    def test_spoofed_node_id_fails(self):
        fleet_key = SecurityConfig(token="fleet-secret").fleet_key
        nonce = secrets.token_bytes(16)
        hw_json = HardwareExchange(gpu_cores=8).to_json_bytes()
        sig = sign_heartbeat_with_hw(fleet_key, "real-node", nonce, hw_json)
        assert not verify_heartbeat_with_hw(fleet_key, "fake-node", nonce, hw_json, sig)


# ------------------------------------------------------------------ #
# Wire format: server handler accepts both 4-field and 5-field APING  #
# ------------------------------------------------------------------ #


def _start_agent(token: str, fleet_id=None) -> PoolAgent:
    """Build an agent with a synthetic HardwareProfile and skip network bring-up.

    The handler under test only depends on `self.hardware`, `self.node_id`,
    `self.data_port` and `self._security` — none of the background tasks.
    """
    from macfleet.engines.base import HardwareProfile

    agent = PoolAgent(
        token=token, fleet_id=fleet_id, tls=False,
        port=50051, data_port=50052,
    )
    agent.hardware = HardwareProfile(
        hostname="test-host",
        node_id="test-host-abcd1234",
        gpu_cores=10,
        ram_gb=24.0,
        memory_bandwidth_gbps=300.0,
        has_ane=True,
        chip_name="Apple M4 Pro (test)",
        mps_available=True,
        mlx_available=True,
    )
    return agent


class TestAPingV2ServerHandler:
    async def test_v2_ping_returns_v2_pong(self):
        sec = SecurityConfig(token="fleet-token")
        fleet_key = sec.fleet_key
        agent = _start_agent("fleet-token")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0,
        )
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        nonce = secrets.token_bytes(16)
        client_hw = HardwareExchange(gpu_cores=16, chip_name="M4 Ultra").to_json_bytes()
        sig = sign_heartbeat_with_hw(fleet_key, "pinger", nonce, client_hw)
        writer.write(
            f"APING pinger {nonce.hex()} {sig.hex()} {client_hw.hex()}\n".encode()
        )
        await writer.drain()

        response = await asyncio.wait_for(reader.readline(), timeout=2.0)
        assert response.startswith(b"APONG")

        parts = response.decode().strip().split(" ")
        assert len(parts) == 5, f"expected 5-field APONG v2, got: {parts}"
        _, peer_id, resp_nonce_hex, resp_sig_hex, peer_hw_hex = parts
        resp_nonce = bytes.fromhex(resp_nonce_hex)
        resp_sig = bytes.fromhex(resp_sig_hex)
        peer_hw_json = bytes.fromhex(peer_hw_hex)
        assert verify_heartbeat_with_hw(
            fleet_key, peer_id, resp_nonce, peer_hw_json, resp_sig,
        )
        peer_hw = HardwareExchange.from_json_bytes(peer_hw_json)
        # Agent's local HW should round-trip through APONG v2
        assert peer_hw.gpu_cores == 10
        assert peer_hw.ram_gb == 24.0
        assert peer_hw.chip_name == "Apple M4 Pro (test)"
        assert peer_hw.data_port == 50052

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    async def test_v1_ping_still_returns_v1_pong(self):
        """A v2.1 (4-field) APING from a legacy peer still gets a v1 APONG."""
        from macfleet.security.auth import sign_heartbeat

        sec = SecurityConfig(token="fleet-token")
        fleet_key = sec.fleet_key
        agent = _start_agent("fleet-token")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0,
        )
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(fleet_key, "legacy-pinger", nonce)
        writer.write(f"APING legacy-pinger {nonce.hex()} {sig.hex()}\n".encode())
        await writer.drain()

        response = await asyncio.wait_for(reader.readline(), timeout=2.0)
        assert response.startswith(b"APONG")
        parts = response.decode().strip().split(" ")
        assert len(parts) == 4, f"legacy ping should get 4-field APONG, got: {parts}"

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    async def test_v2_wrong_key_no_response(self):
        wrong_key = SecurityConfig(token="wrong-token").fleet_key
        agent = _start_agent("correct-token")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0,
        )
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        nonce = secrets.token_bytes(16)
        hw_json = HardwareExchange(gpu_cores=1).to_json_bytes()
        sig = sign_heartbeat_with_hw(wrong_key, "attacker", nonce, hw_json)
        writer.write(
            f"APING attacker {nonce.hex()} {sig.hex()} {hw_json.hex()}\n".encode()
        )
        await writer.drain()

        # Expect connection closed with no APONG
        response = await asyncio.wait_for(reader.read(1024), timeout=1.0)
        assert not response.startswith(b"APONG")

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    async def test_v2_tampered_hw_no_response(self):
        """Signature over one HW, wire carrying a different HW → rejected."""
        sec = SecurityConfig(token="fleet-token")
        fleet_key = sec.fleet_key
        agent = _start_agent("fleet-token")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0,
        )
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        nonce = secrets.token_bytes(16)
        real_hw = HardwareExchange(gpu_cores=8).to_json_bytes()
        lying_hw = HardwareExchange(gpu_cores=128).to_json_bytes()
        sig = sign_heartbeat_with_hw(fleet_key, "pinger", nonce, real_hw)
        # Wire the inflated HW but use the signature over the honest HW
        writer.write(
            f"APING pinger {nonce.hex()} {sig.hex()} {lying_hw.hex()}\n".encode()
        )
        await writer.drain()

        response = await asyncio.wait_for(reader.read(1024), timeout=1.0)
        assert not response.startswith(b"APONG")

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    async def test_v2_oversize_hw_rejected(self):
        """HW payload past HW_HANDSHAKE_MAX_JSON_BYTES → silently dropped."""
        sec = SecurityConfig(token="fleet-token")
        fleet_key = sec.fleet_key
        agent = _start_agent("fleet-token")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0,
        )
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        nonce = secrets.token_bytes(16)
        # Craft a valid signature over a giant HW blob (still rejected by bounds)
        giant_hw = b"x" * (HW_HANDSHAKE_MAX_JSON_BYTES + 100)
        sig = sign_heartbeat_with_hw(fleet_key, "pinger", nonce, giant_hw)
        writer.write(
            f"APING pinger {nonce.hex()} {sig.hex()} {giant_hw.hex()}\n".encode()
        )
        await writer.drain()

        response = await asyncio.wait_for(reader.read(1024), timeout=1.0)
        assert not response.startswith(b"APONG")

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()

    async def test_v2_malformed_hex_rejected(self):
        agent = _start_agent("fleet-token")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0,
        )
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"APING attacker not-hex not-hex not-hex\n")
        await writer.drain()

        response = await asyncio.wait_for(reader.read(1024), timeout=1.0)
        assert not response.startswith(b"APONG")

        writer.close()
        await writer.wait_closed()
        server.close()
        await server.wait_closed()


# ------------------------------------------------------------------ #
# Integration: _add_manual_peer registers with real HW from APONG v2  #
# ------------------------------------------------------------------ #


class TestAddManualPeerCapabilityExchange:
    async def test_manual_peer_registered_with_real_hw(self):
        """Alice `--peer`s Bob; after APING/APONG v2 Alice's registry has Bob's real HW."""
        sec = SecurityConfig(token="fleet-token")
        sec.tls = False  # test uses plain TCP server; skip TLS

        # Bob: the peer Alice will connect to
        bob = _start_agent("fleet-token")
        bob._security.tls = False
        # Override Bob's HW so we can detect that Alice got the real values
        from macfleet.engines.base import HardwareProfile
        bob.hardware = HardwareProfile(
            hostname="bob",
            node_id="bob-deadbeef",
            gpu_cores=32,
            ram_gb=128.0,
            memory_bandwidth_gbps=800.0,
            has_ane=True,
            chip_name="Apple M3 Ultra",
            mps_available=True,
            mlx_available=True,
        )
        bob.data_port = 60052

        bob_server = await asyncio.start_server(
            bob._handle_heartbeat_ping, "127.0.0.1", 0,
        )
        bob_port = bob_server.sockets[0].getsockname()[1]

        # Alice: the caller. We need her background components up to use
        # _add_manual_peer. Minimum: registry + heartbeat bookkeeping.
        alice = _start_agent("fleet-token")
        alice._security.tls = False
        alice.hardware.node_id = "alice-aaaaaaaa"
        alice.hardware.hostname = "alice"

        from macfleet.pool.heartbeat import GossipHeartbeat, HeartbeatConfig
        from macfleet.pool.registry import ClusterRegistry, NodeRecord

        alice._registry = ClusterRegistry(alice.hardware.node_id)
        alice._registry.register(NodeRecord(
            node_id=alice.hardware.node_id,
            hostname=alice.hardware.hostname,
            ip_address="127.0.0.1",
            port=alice.port,
            data_port=alice.data_port,
            hardware=alice.hardware,
        ))
        alice._heartbeat = GossipHeartbeat(
            node_id=alice.hardware.node_id,
            config=HeartbeatConfig(interval_sec=60.0),
            security=sec,
        )

        # Manual-peer dial Bob
        await alice._add_manual_peer(f"127.0.0.1:{bob_port}")

        # Alice's registry should now carry Bob with REAL HW from the APONG v2 payload
        bob_record = alice._registry.get_node("bob-deadbeef")
        assert bob_record is not None, "Bob not registered in Alice's registry"
        assert bob_record.hardware.gpu_cores == 32
        assert bob_record.hardware.ram_gb == 128.0
        assert bob_record.hardware.chip_name == "Apple M3 Ultra"
        assert bob_record.hardware.mps_available is True
        # data_port from the HardwareExchange payload, not port+1 fallback
        assert bob_record.data_port == 60052

        bob_server.close()
        await bob_server.wait_closed()

    async def test_manual_peer_falls_back_when_bob_is_v2_1(self):
        """If Bob responds with 4-field APONG v1, Alice registers zero-HW placeholder."""
        from macfleet.security.auth import sign_heartbeat

        sec = SecurityConfig(token="fleet-token")
        fleet_key = sec.fleet_key

        # Fake v2.1 Bob: only understands 4-field APING
        async def v2_1_bob(reader, writer):
            data = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if data.startswith(b"APING"):
                parts = data.decode().strip().split(" ")
                # v2.1 server: only accepts 4-field, silently drops 5-field as
                # "unknown format" and responds v1 when it sees 4-field.
                # We simulate the worst case: Bob doesn't parse the 5-field
                # ping at all, but for the sake of testing Alice's fallback
                # path we emulate a Bob that IS on v2.1 and trims to 4 fields.
                if len(parts) >= 4:
                    peer_id = parts[1]
                    nonce = bytes.fromhex(parts[2])
                    sig_bytes = bytes.fromhex(parts[3])
                    # v2.1 Bob would verify without HW — which won't match our
                    # v2 signature. But per the PR 5 compat story we ALSO accept
                    # a graceful-degrade where Bob treats the 5-field ping as
                    # "valid enough to respond v1". We emulate that branch here.
                    # (If the client sent a v1 ping, Bob's verify would pass.)
                    _ = nonce, sig_bytes, peer_id
                    # Regardless of verification, emit v1 APONG so Alice exercises
                    # her v1 fallback branch.
                    resp_nonce = secrets.token_bytes(16)
                    resp_sig = sign_heartbeat(fleet_key, "bob-v21", resp_nonce)
                    writer.write(
                        f"APONG bob-v21 {resp_nonce.hex()} {resp_sig.hex()}\n".encode()
                    )
                    await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(v2_1_bob, "127.0.0.1", 0)
        bob_port = server.sockets[0].getsockname()[1]

        alice = _start_agent("fleet-token")
        alice._security.tls = False
        alice.hardware.node_id = "alice-aaaaaaaa"

        from macfleet.pool.heartbeat import GossipHeartbeat, HeartbeatConfig
        from macfleet.pool.registry import ClusterRegistry, NodeRecord

        alice._registry = ClusterRegistry(alice.hardware.node_id)
        alice._registry.register(NodeRecord(
            node_id=alice.hardware.node_id,
            hostname=alice.hardware.hostname,
            ip_address="127.0.0.1",
            port=alice.port,
            data_port=alice.data_port,
            hardware=alice.hardware,
        ))
        alice._heartbeat = GossipHeartbeat(
            node_id=alice.hardware.node_id,
            config=HeartbeatConfig(interval_sec=60.0),
            security=sec,
        )

        await alice._add_manual_peer(f"127.0.0.1:{bob_port}")

        # Alice should have registered Bob with the zero-HW v2.1 fallback
        bob_record = alice._registry.get_node("bob-v21")
        assert bob_record is not None, "v2.1 Bob not registered in Alice's registry"
        assert bob_record.hardware.gpu_cores == 0
        assert bob_record.hardware.chip_name == "unknown (manual peer)"
        # data_port defaults to port + 1 when no HW payload is available
        assert bob_record.data_port == bob_port + 1

        server.close()
        await server.wait_closed()


# Pytest-asyncio auto mode handles async tests via pyproject asyncio_mode setting.
# No explicit `@pytest.mark.asyncio` needed.
_ = pytest  # silence unused-import warning if pytest not referenced elsewhere
