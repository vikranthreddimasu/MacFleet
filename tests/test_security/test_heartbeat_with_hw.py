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
import json
import secrets
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from macfleet.comm.transport import HardwareExchange
from macfleet.pool.agent import PoolAgent
from macfleet.pool.network import NetworkTopology
from macfleet.security.auth import (
    HW_HANDSHAKE_MAX_JSON_BYTES,
    SecurityConfig,
    create_client_ssl_context,
    sign_heartbeat_response,
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
        # v2.3: the APONG signature is bound to OUR request nonce.
        from macfleet.security.auth import verify_heartbeat_response
        assert verify_heartbeat_response(
            fleet_key, peer_id, resp_nonce, nonce, resp_sig, hw_json=peer_hw_json,
        )
        # An unbound (pre-v2.3) verification must NOT accept it.
        assert not verify_heartbeat_with_hw(
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

    async def test_manual_peer_accepts_request_bound_apong_without_hw(self):
        """A 4-field APONG with a request-bound signature registers a zero-HW
        placeholder (HW arrives later via gossip)."""
        from macfleet.security.auth import sign_heartbeat_response

        sec = SecurityConfig(token="fleet-token")
        fleet_key = sec.fleet_key

        # Bob answers the (5-field) APING with a 4-field APONG whose
        # signature is correctly bound to Alice's request nonce.
        async def hw_less_bob(reader, writer):
            data = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if data.startswith(b"APING"):
                parts = data.decode().strip().split(" ")
                if len(parts) >= 4:
                    req_nonce = bytes.fromhex(parts[2])
                    resp_nonce = secrets.token_bytes(16)
                    resp_sig = sign_heartbeat_response(
                        fleet_key, "bob-v21", resp_nonce, req_nonce,
                    )
                    writer.write(
                        f"APONG bob-v21 {resp_nonce.hex()} {resp_sig.hex()}\n".encode()
                    )
                    await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(hw_less_bob, "127.0.0.1", 0)
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

        # Alice should have registered Bob with the zero-HW fallback
        bob_record = alice._registry.get_node("bob-v21")
        assert bob_record is not None, "HW-less Bob not registered in Alice's registry"
        assert bob_record.hardware.gpu_cores == 0
        assert bob_record.hardware.chip_name == "unknown (manual peer)"
        # data_port defaults to port + 1 when no HW payload is available
        assert bob_record.data_port == bob_port + 1

        server.close()
        await server.wait_closed()

    async def test_manual_peer_rejects_legacy_unbound_apong(self):
        """v2.3 security: an APONG signed with the OLD (request-unbound)
        scheme — e.g. a replayed response or a pre-v2.3 peer — is rejected
        and the peer is NOT registered."""
        from macfleet.security.auth import sign_heartbeat

        sec = SecurityConfig(token="fleet-token")
        fleet_key = sec.fleet_key

        async def legacy_bob(reader, writer):
            data = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if data.startswith(b"APING"):
                # Old-style signature: covers only Bob's own nonce, not the
                # request nonce. Exactly what a captured/replayed APONG or a
                # pre-v2.3 peer would produce.
                resp_nonce = secrets.token_bytes(16)
                resp_sig = sign_heartbeat(fleet_key, "legacy-bob", resp_nonce)
                writer.write(
                    f"APONG legacy-bob {resp_nonce.hex()} {resp_sig.hex()}\n".encode()
                )
                await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(legacy_bob, "127.0.0.1", 0)
        bob_port = server.sockets[0].getsockname()[1]

        alice = _start_agent("fleet-token")
        alice._security.tls = False
        alice.hardware.node_id = "alice-bbbbbbbb"

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

        assert alice._registry.get_node("legacy-bob") is None, (
            "request-unbound APONG must be rejected, peer must not register"
        )

        server.close()
        await server.wait_closed()


# Pytest-asyncio auto mode handles async tests via pyproject asyncio_mode setting.
# No explicit `@pytest.mark.asyncio` needed.
_ = pytest  # silence unused-import warning if pytest not referenced elsewhere


@pytest.fixture
async def manual_agents(monkeypatch):
    """Run real TLS heartbeat servers with mDNS entirely disabled."""
    token = secrets.token_hex(32)
    profile = _start_agent(token).hardware
    monkeypatch.setattr("macfleet.pool.agent.profile_hardware", lambda: replace(profile))
    monkeypatch.setattr("macfleet.pool.agent.get_network_topology", NetworkTopology)
    agents = []

    async def start(name, peers=None):
        agent = PoolAgent(name=name, port=0, data_port=0, token=token, peers=peers)
        monkeypatch.setattr(agent._discovery, "async_register_node", AsyncMock())
        monkeypatch.setattr(agent._discovery, "start_discovery", Mock())
        agents.append(agent)
        await agent.start()
        return agent

    yield start
    for agent in reversed(agents):
        await agent.stop()


async def test_one_manual_peer_forms_mutual_membership_without_mdns(manual_agents):
    bob = await manual_agents("bob")
    alice = await manual_agents("alice", [f"127.0.0.1:{bob.port}"])
    async with asyncio.timeout(2):
        while alice.registry.world_size != 2 or bob.registry.world_size != 2:
            await asyncio.sleep(0.01)
    for receiver, peer in ((alice, bob), (bob, alice)):
        record = receiver.registry.get_node(peer.node_id)
        assert record.ip_address == "127.0.0.1"
        assert record.port == peer.port
        assert record.data_port == peer.data_port
        assert record.port != record.data_port
        assert record.is_alive
        assert record.hardware.gpu_cores == peer.hardware.gpu_cores


@pytest.mark.parametrize("invalid_reply", ["identity", "replayed_nonce", "invalid_hardware", "nan_hardware"])
async def test_reverse_probe_requires_fresh_matching_identity(manual_agents, invalid_reply):
    bob = await manual_agents("bob")
    original_nonce = secrets.token_bytes(16)
    received = asyncio.Event()
    reply = asyncio.Event()
    probes = []

    async def endpoint(reader, writer):
        try:
            request = await reader.readline()
            request_nonce = bytes.fromhex(request.decode().split(" ")[2])
            probes.append(request_nonce)
            received.set()
            await reply.wait()
            node_id = "different-peer" if invalid_reply == "identity" else "claimed-peer"
            signed_nonce = original_nonce if invalid_reply == "replayed_nonce" else request_nonce
            response_nonce = secrets.token_bytes(16)
            hardware = None
            if invalid_reply == "invalid_hardware":
                hardware = b'{"gpu_cores":"oops"}'
            elif invalid_reply == "nan_hardware":
                hardware = b'{"ram_gb":NaN}'
            signature = sign_heartbeat_response(
                bob._security.fleet_key, node_id, response_nonce, signed_nonce, hw_json=hardware,
            )
            suffix = f" {hardware.hex()}" if hardware is not None else ""
            writer.write(f"APONG {node_id} {response_nonce.hex()} {signature.hex()}{suffix}\n".encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(endpoint, "127.0.0.1", 0, ssl=bob._heartbeat_ssl_ctx)
    port = server.sockets[0].getsockname()[1]
    hardware = json.dumps({"heartbeat_port": port, "data_port": 50052}).encode()
    signature = sign_heartbeat_with_hw(bob._security.fleet_key, "claimed-peer", original_nonce, hardware)
    captured_ping = f"APING claimed-peer {original_nonce.hex()} {signature.hex()} {hardware.hex()}\n".encode()

    async def replay_ping():
        reader, writer = await asyncio.open_connection("127.0.0.1", bob.port, ssl=create_client_ssl_context())
        try:
            writer.write(captured_ping)
            await writer.drain()
            assert (await reader.readline()).startswith(b"APONG")
        finally:
            writer.close()
            await writer.wait_closed()

    try:
        await asyncio.gather(replay_ping(), replay_ping())
        await asyncio.wait_for(received.wait(), 1)
        assert bob.registry.get_node("claimed-peer") is None
        assert len(probes) == 1
        assert probes[0] != original_nonce
        reply.set()
        async with asyncio.timeout(2):
            while bob._peer_probes:
                await asyncio.sleep(0.01)
        assert bob.registry.world_size == 1
        assert bob.registry.get_node("different-peer") is None
        assert bob.registry.get_node("claimed-peer") is None
    finally:
        reply.set()
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize("start_probes", [True, False])
async def test_reverse_probes_are_bounded_coalesced_and_cancelled_on_stop(manual_agents, monkeypatch, start_probes):
    bob = await manual_agents("bob")

    async def blocked_probe(*args, **kwargs):
        await asyncio.Event().wait()

    probe = AsyncMock(side_effect=blocked_probe)
    monkeypatch.setattr(bob, "_add_manual_peer", probe)
    hardware = HardwareExchange(heartbeat_port=50051, data_port=50052)
    for index in range(40):
        for _ in range(2):
            bob._queue_peer_probe(f"peer-{index}", "127.0.0.1", hardware)
    assert len(bob._peer_probes) == 32
    tasks = list(bob._peer_probes.values())
    if start_probes:
        async with asyncio.timeout(1):
            while probe.await_count != 32:
                await asyncio.sleep(0)
    await asyncio.wait_for(bob.stop(), 1)
    assert not bob._peer_probes
    assert all(task.done() for task in tasks)
