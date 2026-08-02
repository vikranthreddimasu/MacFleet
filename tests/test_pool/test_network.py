"""Tests for network detection and link scoring."""

import asyncio
import time

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

    async def test_latency_timeout_bounds_connection_close(self, monkeypatch):
        writer = _ProbeWriter(wait_closed_delay=None)

        async def open_connection(host, port):
            return _ProbeReader(), writer

        monkeypatch.setattr(network.asyncio, "open_connection", open_connection)

        started = time.monotonic()
        result = await asyncio.wait_for(
            network.measure_latency("peer", 5000, timeout=0.01),
            timeout=0.2,
        )

        assert result == -1.0
        assert time.monotonic() - started < 0.15
        assert writer.closed

    async def test_bandwidth_requires_receiver_acknowledgement(self, monkeypatch):
        writer = _ProbeWriter()

        async def open_connection(host, port):
            return _ProbeReader(read_delay=None), writer

        monkeypatch.setattr(network.asyncio, "open_connection", open_connection)

        result = await network.measure_bandwidth(
            "peer",
            5000,
            payload_mb=0.001,
            timeout=0.01,
        )

        assert result == 0.0
        assert writer.closed

    async def test_bandwidth_timeout_is_end_to_end(self, monkeypatch):
        writer = _ProbeWriter()

        async def open_connection(host, port):
            await asyncio.sleep(0.03)
            return _ProbeReader(read_delay=0.03), writer

        monkeypatch.setattr(network.asyncio, "open_connection", open_connection)

        started = time.monotonic()
        result = await network.measure_bandwidth(
            "peer",
            5000,
            payload_mb=0.001,
            timeout=0.05,
        )

        assert result == 0.0
        assert time.monotonic() - started < 0.09
        assert writer.closed

    async def test_bandwidth_timeout_bounds_stalled_drain(self, monkeypatch):
        writer = _ProbeWriter(drain_delay=None)

        async def open_connection(host, port):
            return _ProbeReader(), writer

        monkeypatch.setattr(network.asyncio, "open_connection", open_connection)

        result = await asyncio.wait_for(
            network.measure_bandwidth(
                "peer",
                5000,
                payload_mb=0.001,
                timeout=0.01,
            ),
            timeout=0.2,
        )

        assert result == 0.0
        assert writer.closed

    async def test_bandwidth_reports_measurement_after_acknowledgement(self, monkeypatch):
        writer = _ProbeWriter()

        async def open_connection(host, port):
            return _ProbeReader(), writer

        monkeypatch.setattr(network.asyncio, "open_connection", open_connection)

        result = await network.measure_bandwidth(
            "peer",
            5000,
            payload_mb=0.001,
            timeout=0.1,
        )

        assert result > 0.0
        assert writer.bytes_written == int(0.001 * 1024 * 1024)
        assert writer.eof_written
        assert writer.closed


class _ProbeReader:
    def __init__(self, read_delay=0.0):
        self.read_delay = read_delay

    async def read(self):
        if self.read_delay is None:
            await asyncio.Future()
        await asyncio.sleep(self.read_delay)
        return b""


class _ProbeWriter:
    def __init__(self, drain_delay=0.0, wait_closed_delay=0.0):
        self.drain_delay = drain_delay
        self.wait_closed_delay = wait_closed_delay
        self.bytes_written = 0
        self.eof_written = False
        self.closed = False

    def write(self, data):
        self.bytes_written += len(data)

    async def drain(self):
        if self.drain_delay is None:
            await asyncio.Future()
        await asyncio.sleep(self.drain_delay)

    def can_write_eof(self):
        return True

    def write_eof(self):
        self.eof_written = True

    def close(self):
        self.closed = True

    async def wait_closed(self):
        if self.wait_closed_delay is None:
            await asyncio.Future()
        await asyncio.sleep(self.wait_closed_delay)
