"""Tests for network detection and link scoring."""

import pytest

from macfleet.pool import network
from macfleet.pool.network import (
    LinkType,
    NetworkLink,
    NetworkTopology,
    detect_interfaces,
    get_network_topology,
)


class TestNetworkLink:
    def test_score_calculation(self):
        link = NetworkLink(
            interface="en0",
            link_type=LinkType.WIFI,
            ip_address="192.168.1.100",
            bandwidth_mbps=200.0,
            latency_ms=5.0,
            loss_rate=0.01,
        )
        # score = 200 * (1 - 0.01) / 5 = 200 * 0.99 / 5 = 39.6
        assert abs(link.score - 39.6) < 0.1

    def test_score_zero_bandwidth(self):
        link = NetworkLink(
            interface="en0",
            link_type=LinkType.WIFI,
            ip_address="192.168.1.100",
            bandwidth_mbps=0.0,
            latency_ms=5.0,
        )
        assert link.score == 0.0

    def test_theoretical_bandwidth(self):
        assert NetworkLink("bridge0", LinkType.THUNDERBOLT, "10.0.0.1").theoretical_bandwidth_mbps == 40_000.0
        assert NetworkLink("en0", LinkType.WIFI, "192.168.1.1").theoretical_bandwidth_mbps == 300.0
        assert NetworkLink("en1", LinkType.ETHERNET, "192.168.1.1").theoretical_bandwidth_mbps == 1_000.0


class TestNetworkTopology:
    def test_best_link(self):
        topo = NetworkTopology(
            links=[
                NetworkLink("en0", LinkType.WIFI, "192.168.1.100", bandwidth_mbps=200, latency_ms=5),
                NetworkLink("bridge0", LinkType.THUNDERBOLT, "10.0.0.1", bandwidth_mbps=32000, latency_ms=0.5),
            ],
            hostname="test-mac",
        )
        best = topo.best_link
        assert best is not None
        assert best.link_type == LinkType.THUNDERBOLT

    def test_best_link_prefers_fastest_theoretical_link_without_measurements(self):
        wifi = NetworkLink("en0", LinkType.WIFI, "192.168.1.10")
        ethernet = NetworkLink("en1", LinkType.ETHERNET, "192.168.1.11")

        assert NetworkTopology(links=[wifi, ethernet]).best_link == ethernet

    def test_has_link_types(self):
        topo = NetworkTopology(
            links=[
                NetworkLink("en0", LinkType.WIFI, "192.168.1.100"),
                NetworkLink("bridge0", LinkType.THUNDERBOLT, "10.0.0.1"),
            ],
        )
        assert topo.has_wifi
        assert topo.has_thunderbolt
        assert not topo.has_ethernet

    def test_empty_topology(self):
        topo = NetworkTopology()
        assert topo.best_link is None
        assert not topo.has_wifi


class TestDetectInterfaces:
    def test_subprocess_probe_errors_fall_back_cleanly(self, monkeypatch):
        def denied(*args, **kwargs):
            raise PermissionError("restricted environment")

        monkeypatch.setattr(network.subprocess, "run", denied)

        assert not network._is_wifi_interface("en0")
        assert network._ip_to_interface("192.168.1.10") is None
        assert network._parse_ifconfig() == []

    def test_detect_returns_list(self):
        links = detect_interfaces()
        assert isinstance(links, list)
        # On any Mac, we should find at least one interface
        # (but in CI this may be empty, so don't assert > 0)

    def test_get_topology(self):
        topo = get_network_topology()
        assert isinstance(topo, NetworkTopology)
        assert topo.hostname != ""


class TestNetworkProbes:
    @pytest.mark.parametrize(
        "timeout",
        [0, -1, True, float("nan"), float("inf"), "soon"],
    )
    async def test_latency_rejects_invalid_timeout(self, timeout):
        with pytest.raises(ValueError, match="timeout"):
            await network.measure_latency("127.0.0.1", 9, timeout=timeout)

    @pytest.mark.parametrize(
        "timeout",
        [0, -1, True, float("nan"), float("inf"), "soon"],
    )
    async def test_bandwidth_rejects_invalid_timeout(self, timeout):
        with pytest.raises(ValueError, match="timeout"):
            await network.measure_bandwidth("127.0.0.1", 9, timeout=timeout)

    @pytest.mark.parametrize(
        "payload_mb",
        [0, -1, True, float("nan"), float("inf"), "large"],
    )
    async def test_bandwidth_rejects_invalid_payload_size(self, payload_mb):
        with pytest.raises(ValueError, match="payload_mb"):
            await network.measure_bandwidth("127.0.0.1", 9, payload_mb=payload_mb)
