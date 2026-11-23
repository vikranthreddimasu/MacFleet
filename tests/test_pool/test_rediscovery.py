"""Secure rediscovery preserves authenticated state only for the same live endpoint."""

import asyncio
from dataclasses import replace

import pytest

from macfleet.comm.transport import HardwareExchange
from macfleet.engines.base import HardwareProfile
from macfleet.pool.agent import PoolAgent
from macfleet.pool.discovery import DiscoveredNode
from macfleet.pool.heartbeat import GossipHeartbeat, NodeStatus
from macfleet.pool.registry import ClusterRegistry, NodeRecord


def make_agent(node_id, secure=True):
    agent = PoolAgent(token="isolated-rediscovery-test-token" if secure else None)
    agent.hardware = HardwareProfile(node_id, node_id, 8, 16, 100, True, "test chip")
    agent._registry = ClusterRegistry(node_id)
    agent._registry.register(NodeRecord(node_id, node_id, "127.0.0.1", agent.port, agent.hardware))
    agent._heartbeat = GossipHeartbeat(
        node_id, security=agent._security, local_hw_provider=agent._gossip_local_hw_bytes,
        on_peer_hw=agent._on_peer_hw_received, on_recovered=agent._on_peer_recovered,
    )
    return agent


def advertisement():
    return DiscoveredNode("peer-host", "peer", "127.0.0.1", 50051, 0, 0, "unknown", "", "test", data_port=50052)


def test_repeat_advertisement_preserves_membership_hardware_and_tracking():
    agent = make_agent("local")
    node = advertisement()
    agent._on_peer_discovered(node)
    agent._on_peer_hw_received("peer", HardwareExchange(gpu_cores=40, ram_gb=64, data_port=51999).to_json_bytes())
    agent.registry.update_throughput("peer", 123.5)
    verified = agent.registry.get_node("peer")
    tracker = agent._heartbeat.peers["peer"]
    tracker.missed_rounds = 2
    agent._on_peer_discovered(replace(node))
    assert agent.registry.get_node("peer") is verified
    assert agent.registry.world_size == 2
    assert agent.registry.coordinator_id == "peer"
    assert (verified.hardware.gpu_cores, verified.data_port, verified.throughput_samples_sec) == (40, 51999, 123.5)
    assert agent._heartbeat.peers["peer"] is tracker
    assert tracker.missed_rounds == 2

    # An announcement that catches up with the already verified data port is also safe.
    node = replace(node, data_port=verified.data_port)
    agent._on_peer_discovered(node)
    assert agent.registry.get_node("peer") is verified

    # Authentication can recover a failed peer; discovery alone cannot.
    agent._on_peer_failed("peer")
    agent._on_peer_discovered(replace(node))
    assert agent.registry.get_node("peer").status == NodeStatus.FAILED


@pytest.mark.parametrize("change", [{"ip_address": "127.0.0.2"}, {"port": 51051}, {"data_port": 51052}])
def test_changed_discovery_endpoint_requires_fresh_authentication(change):
    agent = make_agent("local")
    node = advertisement()
    agent._on_peer_discovered(node)
    agent._on_peer_hw_received("peer", HardwareExchange(gpu_cores=32, data_port=node.data_port).to_json_bytes())
    agent.registry.update_throughput("peer", 45)
    tracker = agent._heartbeat.peers["peer"]
    agent._on_peer_discovered(replace(node, **change))
    pending = agent.registry.get_node("peer")
    assert pending.status == NodeStatus.SUSPECTED
    assert pending.hardware.gpu_cores == pending.throughput_samples_sec == 0
    assert agent.registry.world_size == 1
    assert agent._heartbeat.peers["peer"] is not tracker


def test_open_discovery_updates_advertised_hardware():
    agent = make_agent("local", secure=False)
    node = advertisement()
    agent._on_peer_discovered(node)
    agent._on_peer_discovered(replace(node, gpu_cores=12))
    assert agent.registry.world_size == 2
    assert agent.registry.get_node("peer").hardware.gpu_cores == 12


@pytest.mark.parametrize("probe", ["manual", "gossip"])
@pytest.mark.parametrize("change", ["unchanged", "data port", "leave", "rejoin"])
async def test_heartbeat_reply_belongs_to_current_endpoint(probe, change):
    """Pause a legitimate TLS heartbeat, deliver discovery, then release the real peer."""
    agent, peer = make_agent("local"), make_agent("peer")
    accepted, resume, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def respond(reader, writer):
        accepted.set()
        await resume.wait()
        try:
            await peer._handle_heartbeat_ping(reader, writer)
        finally:
            finished.set()

    async with await asyncio.start_server(respond, "127.0.0.1", 0, ssl=peer._heartbeat_ssl_ctx) as server:
        port = server.sockets[0].getsockname()[1]
        node = replace(advertisement(), port=port)
        agent._on_peer_discovered(node)
        work = (agent._heartbeat._gossip_round() if probe == "gossip"
                else agent._add_manual_peer(f"127.0.0.1:{port}"))
        task = asyncio.create_task(work)
        try:
            await asyncio.wait_for(accepted.wait(), 3)
            if change == "unchanged":
                agent._on_peer_discovered(replace(node))
            elif change == "data port":
                agent._on_peer_discovered(replace(node, data_port=51052))
            else:
                agent._on_peer_removed("peer")
                if change == "rejoin":
                    agent._on_peer_discovered(node)
            resume.set()
            await asyncio.wait_for(task, 3)
            await asyncio.wait_for(finished.wait(), 3)
            record = agent.registry.get_node("peer")
            expected = {"unchanged": NodeStatus.ALIVE, "data port": NodeStatus.SUSPECTED,
                        "leave": NodeStatus.LEFT, "rejoin": NodeStatus.SUSPECTED}
            assert record.status == expected[change]
            assert record.hardware.gpu_cores == (peer.hardware.gpu_cores if change == "unchanged" else 0)
            assert record.data_port == (51052 if change == "data port" else node.data_port)
        finally:
            resume.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("change", ["unchanged", "replaced"])
async def test_inbound_hardware_from_current_membership_only_queues_reverse_probe(change):
    from unittest.mock import Mock

    from macfleet.pool.heartbeat import request_heartbeat
    from macfleet.security.auth import create_client_ssl_context

    agent, peer = make_agent("local"), make_agent("peer")
    node = advertisement()
    agent._on_peer_discovered(node)
    queued = Mock()
    agent._queue_peer_probe = queued
    finished = asyncio.Event()

    async def handle(reader, writer):
        try:
            await agent._handle_heartbeat_ping(reader, writer)
        finally:
            finished.set()

    async with await asyncio.start_server(handle, "127.0.0.1", 0, ssl=agent._heartbeat_ssl_ctx) as server:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server.sockets[0].getsockname()[1], ssl=create_client_ssl_context(),
        )
        class DiscoveryDuringReply:
            async def readline(self):
                result = await reader.readline()
                updated = node if change == "unchanged" else replace(node, data_port=51052)
                agent._on_peer_discovered(updated)
                return result

        try:
            authenticated, _ = await request_heartbeat(
                DiscoveryDuringReply(), writer, peer._security.fleet_key, peer.node_id,
                peer._local_hw_exchange().to_json_bytes(), agent.node_id,
            )
            assert authenticated == agent.node_id
            await asyncio.wait_for(finished.wait(), 3)
            assert queued.call_count == int(change == "unchanged")
        finally:
            writer.close()
            await writer.wait_closed()


async def test_discovery_replacement_retires_only_the_prior_reverse_probe(monkeypatch):
    agent = make_agent("local")
    agent._running = True
    node = advertisement()
    agent._on_peer_discovered(node)
    first_started = asyncio.Event()
    finished = asyncio.Event()
    async def wait_for_peer(*args, **kwargs):
        first_started.set()
        await finished.wait()

    monkeypatch.setattr(agent, "_add_manual_peer", wait_for_peer)
    hw = HardwareExchange(heartbeat_port=node.port, data_port=node.data_port)
    agent._queue_peer_probe(node.node_id, node.ip_address, hw)
    old = agent._peer_probes[node.node_id]
    await asyncio.wait_for(first_started.wait(), 3)
    agent._on_peer_discovered(replace(node))
    assert agent._peer_probes[node.node_id] is old

    moved = replace(node, port=51051)
    agent._on_peer_discovered(moved)
    agent._queue_peer_probe(moved.node_id, moved.ip_address, replace(hw, heartbeat_port=moved.port))
    new = agent._peer_probes[moved.node_id]
    try:
        await asyncio.gather(old, return_exceptions=True)
        assert old.cancelled()
        assert agent._peer_probes[moved.node_id] is new
        finished.set()
        await asyncio.wait_for(new, 3)
    finally:
        finished.set()
        await asyncio.gather(old, new, return_exceptions=True)
