"""Tests for v2.2 PR 2 (Issue 5) — heartbeat / data transport port split.

Heartbeat on 50051 speaks line-delimited APING/APONG. Data transport on 50052
speaks binary WireMessage. They must not share a port — handshakes collide.
"""

from __future__ import annotations

import pytest

from macfleet.engines.base import HardwareProfile
from macfleet.pool.agent import PoolAgent
from macfleet.pool.discovery import DiscoveredNode
from macfleet.pool.registry import NodeRecord


class TestDiscoveredNodeDataPortDefault:
    """data_port backward-compat: missing or 0 falls back to port + 1."""

    def test_data_port_zero_falls_back_to_port_plus_one(self):
        node = DiscoveredNode(
            hostname="h", node_id="n", ip_address="1.1.1.1", port=50051,
            gpu_cores=0, ram_gb=0, chip_name="x", link_types="",
            pool_version="2.2.0", compute_score=0.0, data_port=0,
        )
        assert node.data_port == 50052

    def test_data_port_explicit_value_preserved(self):
        node = DiscoveredNode(
            hostname="h", node_id="n", ip_address="1.1.1.1", port=50051,
            gpu_cores=0, ram_gb=0, chip_name="x", link_types="",
            pool_version="2.2.0", compute_score=0.0, data_port=60000,
        )
        assert node.data_port == 60000

    def test_non_default_heartbeat_port_derives_data_port(self):
        node = DiscoveredNode(
            hostname="h", node_id="n", ip_address="1.1.1.1", port=9000,
            gpu_cores=0, ram_gb=0, chip_name="x", link_types="",
            pool_version="2.1.1", compute_score=0.0, data_port=0,
        )
        # 2.1.x peer without data_port in mDNS TXT → derive 9001
        assert node.data_port == 9001


class TestNodeRecordDataPortDefault:
    """NodeRecord data_port backward-compat: 0 falls back to port + 1."""

    def _hw(self) -> HardwareProfile:
        return HardwareProfile(
            hostname="h", node_id="n", gpu_cores=0, ram_gb=0.0,
            memory_bandwidth_gbps=0.0, has_ane=True, chip_name="x",
        )

    def test_unset_data_port_derives(self):
        rec = NodeRecord(
            node_id="n", hostname="h", ip_address="1.1.1.1",
            port=50051, hardware=self._hw(),
        )
        assert rec.data_port == 50052

    def test_explicit_data_port(self):
        rec = NodeRecord(
            node_id="n", hostname="h", ip_address="1.1.1.1",
            port=50051, data_port=50055, hardware=self._hw(),
        )
        assert rec.data_port == 50055


class TestPoolAgentPortSplit:
    """PoolAgent enforces port / data_port distinctness."""

    def test_default_ports(self):
        agent = PoolAgent(token="secret-token-long-enough-to-pass-min")
        assert agent.port == 50051
        assert agent.data_port == 50052

    def test_custom_heartbeat_port_derives_data_port(self):
        agent = PoolAgent(port=9000, token="secret-token-long-enough-to-pass-min")
        assert agent.port == 9000
        assert agent.data_port == 9001

    def test_explicit_data_port_overrides_derivation(self):
        agent = PoolAgent(
            port=9000, data_port=9500,
            token="secret-token-long-enough-to-pass-min",
        )
        assert agent.port == 9000
        assert agent.data_port == 9500

    def test_same_port_for_both_rejected(self):
        with pytest.raises(ValueError, match="must differ"):
            PoolAgent(
                port=9000, data_port=9000,
                token="secret-token-long-enough-to-pass-min",
            )

    def test_open_fleet_port_split(self):
        """Port split works with --open (no token) too."""
        agent = PoolAgent()  # no token = open fleet
        assert agent.port == 50051
        assert agent.data_port == 50052
