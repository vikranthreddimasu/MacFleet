"""mDNS peer removal must clear registry + heartbeat state."""

from __future__ import annotations

from unittest.mock import MagicMock

from macfleet.engines.base import HardwareProfile
from macfleet.pool.agent import PoolAgent
from macfleet.pool.registry import ClusterRegistry, NodeRecord, NodeStatus


def _make_node(node_id: str) -> NodeRecord:
    return NodeRecord(
        node_id=node_id,
        hostname=f"host-{node_id}",
        ip_address="127.0.0.1",
        port=50051,
        hardware=HardwareProfile(
            hostname=f"host-{node_id}",
            node_id=node_id,
            gpu_cores=10,
            ram_gb=16.0,
            memory_bandwidth_gbps=100.0,
            has_ane=True,
            chip_name="Apple M4",
        ),
    )


def test_on_peer_removed_deregisters_and_drops_heartbeat():
    agent = PoolAgent()  # open fleet (no token)
    registry = ClusterRegistry("local-node")
    registry.register(_make_node("local-node"))
    registry.register(_make_node("peer-1"))
    heartbeat = MagicMock()
    agent._registry = registry
    agent._heartbeat = heartbeat

    agent._on_peer_removed("peer-1")

    peer = registry.get_node("peer-1")
    assert peer is not None
    assert peer.status == NodeStatus.LEFT
    heartbeat.remove_peer.assert_called_once_with("peer-1")
