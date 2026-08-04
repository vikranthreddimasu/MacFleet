"""Validation tests for mDNS peer discovery metadata."""

from __future__ import annotations

import socket

import pytest
from zeroconf import ServiceInfo

from macfleet.pool.discovery import PoolServiceListener

SERVICE_TYPE = "_macfleet._tcp.local."


def _service_info(
    *,
    properties: dict[bytes, bytes] | None = None,
    port: int = 50051,
) -> ServiceInfo:
    base_properties = {
        b"node_id": b"peer-1",
        b"gpu_cores": b"20",
        b"ram_gb": b"64",
        b"chip_name": b"Apple M4 Ultra",
        b"link_types": b"ethernet",
        b"pool_version": b"2.2.1",
        b"compute_score": b"512.0",
        b"data_port": b"50052",
    }
    if properties:
        base_properties.update(properties)
    return ServiceInfo(
        SERVICE_TYPE,
        f"peer-1.{SERVICE_TYPE}",
        addresses=[socket.inet_aton("192.0.2.10")],
        port=port,
        properties=base_properties,
        server="peer-1.local.",
    )


class TestDiscoveryNumericValidation:
    def test_valid_metadata_is_parsed(self):
        node = PoolServiceListener()._parse_service_info(_service_info())

        assert node is not None
        assert node.node_id == "peer-1"
        assert node.port == 50051
        assert node.data_port == 50052
        assert node.gpu_cores == 20
        assert node.ram_gb == 64
        assert node.compute_score == 512.0

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            (b"gpu_cores", b"-1"),
            (b"gpu_cores", b"1025"),
            (b"gpu_cores", b"1.5"),
            (b"ram_gb", b"-1"),
            (b"ram_gb", b"65537"),
            (b"ram_gb", b"16.5"),
        ],
    )
    def test_invalid_hardware_counts_reject_peer(self, key, value):
        info = _service_info(properties={key: value})

        assert PoolServiceListener()._parse_service_info(info) is None

    @pytest.mark.parametrize("value", [b"-1", b"nan", b"inf", b"-inf"])
    def test_invalid_compute_score_rejects_peer(self, value):
        info = _service_info(properties={b"compute_score": value})

        assert PoolServiceListener()._parse_service_info(info) is None

    @pytest.mark.parametrize("value", [b"-1", b"65536", b"1.5"])
    def test_invalid_data_port_rejects_peer(self, value):
        info = _service_info(properties={b"data_port": value})

        assert PoolServiceListener()._parse_service_info(info) is None

    def test_same_control_and_data_port_rejects_peer(self):
        info = _service_info(properties={b"data_port": b"50051"})

        assert PoolServiceListener()._parse_service_info(info) is None

    def test_missing_data_port_uses_next_control_port(self):
        info = _service_info(properties={b"data_port": b"0"}, port=9000)

        node = PoolServiceListener()._parse_service_info(info)

        assert node is not None
        assert node.data_port == 9001

    def test_missing_data_port_rejects_control_port_at_maximum(self):
        info = _service_info(properties={b"data_port": b"0"}, port=65535)

        assert PoolServiceListener()._parse_service_info(info) is None


class TestDiscoveryTextValidation:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            (b"node_id", b"n" * 129),
            (b"node_id", b"peer\x1b[31m"),
            (b"chip_name", b"c" * 129),
            (b"chip_name", b"M4\nforged-output"),
            (b"link_types", b"l" * 129),
            (b"pool_version", b"v" * 65),
        ],
    )
    def test_unsafe_text_property_rejects_peer(self, key, value):
        info = _service_info(properties={key: value})

        assert PoolServiceListener()._parse_service_info(info) is None

    def test_empty_node_id_falls_back_to_hostname(self):
        info = _service_info(properties={b"node_id": b""})

        node = PoolServiceListener()._parse_service_info(info)

        assert node is not None
        assert node.node_id == "peer-1.local"

    def test_unsafe_hostname_rejects_peer(self):
        info = _service_info()
        info.server = "peer\nforged.local."

        assert PoolServiceListener()._parse_service_info(info) is None
