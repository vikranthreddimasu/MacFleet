"""Tests for the PoolAgent → Dashboard NodeHealth adapter (E2 / PR 11)."""

from __future__ import annotations

from macfleet.monitoring.agent_adapter import (
    build_node_health_for_peers,
    build_node_health_for_self,
    classify_health,
    snapshot_all,
)
from macfleet.monitoring.health import HealthStatus
from macfleet.sdk.pool import Pool


class TestClassifyHealth:
    def test_healthy_bucket(self):
        assert classify_health(1.0) == HealthStatus.HEALTHY
        assert classify_health(0.8) == HealthStatus.HEALTHY

    def test_degraded_bucket(self):
        assert classify_health(0.7) == HealthStatus.DEGRADED
        assert classify_health(0.5) == HealthStatus.DEGRADED

    def test_unhealthy_bucket(self):
        assert classify_health(0.49) == HealthStatus.UNHEALTHY
        assert classify_health(0.0) == HealthStatus.UNHEALTHY


class TestBuildNodeHealthForSelf:
    def test_snapshot_populates_basics(self):
        with Pool(
            name="dash-self",
            open=True,
            port=0,
            data_port=0,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            health = build_node_health_for_self(
                pool._agent,
                loss_trend="decreasing",
                throughput_samples_sec=123.4,
                avg_sync_time_sec=0.05,
            )
            assert health.node_id == pool._agent.node_id
            assert health.loss_trend == "decreasing"
            assert health.throughput_samples_sec == 123.4
            assert health.avg_sync_time_sec == 0.05
            # timestamp set, status classified (not UNKNOWN)
            assert health.timestamp > 0
            assert health.status != HealthStatus.UNKNOWN

    def test_connection_failures_tracked(self):
        with Pool(
            name="dash-failures",
            open=True,
            port=0,
            data_port=0,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            health = build_node_health_for_self(
                pool._agent, connection_failures=5,
            )
            assert health.connection_failures == 5


class TestBuildNodeHealthForPeers:
    def test_empty_when_solo(self):
        """No peers in registry → empty list."""
        with Pool(
            name="dash-solo",
            open=True,
            port=0,
            data_port=0,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            peers = build_node_health_for_peers(pool._agent)
            assert peers == []

    def test_self_excluded(self):
        """build_node_health_for_peers must not include self."""
        with Pool(
            name="dash-no-self",
            open=True,
            port=0,
            data_port=0,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            peer_healths = build_node_health_for_peers(pool._agent)
            self_id = pool._agent.node_id
            assert all(p.node_id != self_id for p in peer_healths)


class TestSnapshotAll:
    def test_includes_self_first(self):
        with Pool(
            name="dash-all",
            open=True,
            port=0,
            data_port=0,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            nodes = snapshot_all(pool._agent)
            assert len(nodes) == 1  # just self
            assert nodes[0].node_id == pool._agent.node_id

    def test_pool_dashboard_snapshot_helper(self):
        """Pool.dashboard_snapshot() is the one-call entry point."""
        with Pool(
            name="dash-helper",
            open=True,
            port=0,
            data_port=0,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            snap = pool.dashboard_snapshot()
            assert len(snap) == 1
            assert snap[0].node_id == pool._agent.node_id

    def test_pool_dashboard_snapshot_empty_when_not_distributed(self):
        """Without a live agent, Pool.dashboard_snapshot returns []."""
        with Pool(open=True) as pool:  # flag off by default
            assert pool.dashboard_snapshot() == []
