"""Tests for gossip heartbeat failure detection."""

import time

from macfleet.pool.heartbeat import (
    GossipHeartbeat,
    HeartbeatConfig,
    NodeStatus,
    PeerState,
)


class TestPeerState:
    def test_initial_state(self):
        peer = PeerState(node_id="test", ip_address="127.0.0.1", port=50051)
        assert peer.status == NodeStatus.ALIVE
        assert peer.is_alive
        assert peer.missed_rounds == 0

    def test_status_values(self):
        assert NodeStatus.ALIVE.value == "alive"
        assert NodeStatus.SUSPECTED.value == "suspected"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.LEFT.value == "left"


class TestGossipHeartbeat:
    def test_add_peer(self):
        hb = GossipHeartbeat(node_id="self")
        hb.add_peer("peer1", "127.0.0.1", 50051)
        assert len(hb.peers) == 1
        assert hb.alive_count == 1

    def test_ignore_self(self):
        hb = GossipHeartbeat(node_id="self")
        hb.add_peer("self", "127.0.0.1", 50051)
        assert len(hb.peers) == 0

    def test_remove_peer(self):
        hb = GossipHeartbeat(node_id="self")
        hb.add_peer("peer1", "127.0.0.1", 50051)
        hb.remove_peer("peer1")
        assert hb.peers["peer1"].status == NodeStatus.LEFT

    def test_record_heartbeat(self):
        hb = GossipHeartbeat(node_id="self")
        hb.add_peer("peer1", "127.0.0.1", 50051)
        hb._peers["peer1"].missed_rounds = 2
        hb.record_heartbeat("peer1")
        assert hb._peers["peer1"].missed_rounds == 0
        assert hb._peers["peer1"].is_alive

    def test_recovery_callback(self):
        recovered = []
        hb = GossipHeartbeat(
            node_id="self",
            on_recovered=lambda nid: recovered.append(nid),
        )
        hb.add_peer("peer1", "127.0.0.1", 50051)
        hb._peers["peer1"].status = NodeStatus.SUSPECTED
        hb.record_heartbeat("peer1")
        assert "peer1" in recovered
        assert hb._peers["peer1"].is_alive

    def test_status_summary(self):
        hb = GossipHeartbeat(node_id="self")
        hb.add_peer("a", "127.0.0.1", 50051)
        hb.add_peer("b", "127.0.0.1", 50052)
        hb._peers["b"].status = NodeStatus.FAILED

        summary = hb.get_status_summary()
        assert summary["alive"] == 1
        assert summary["failed"] == 1

    def test_config_defaults(self):
        config = HeartbeatConfig()
        assert config.interval_sec == 1.0
        assert config.suspicion_rounds == 3
        assert config.failure_timeout_sec == 10.0
        assert config.fanout == 3
