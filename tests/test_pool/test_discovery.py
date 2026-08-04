"""Validation tests for mDNS peer discovery metadata."""

from __future__ import annotations

import socket

import pytest
from zeroconf import ServiceInfo

from macfleet.pool.discovery import PoolServiceListener, ServiceRegistry
from macfleet.pool.network import LinkType, NetworkLink
from macfleet.pool.topology import deserialize_network_links

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


class TestDiscoveryAdvertisementValidation:
    def _properties(self, **overrides):
        values = {
            "node_id": "node-1",
            "gpu_cores": 20,
            "ram_gb": 64,
            "chip_name": "Apple M4 Ultra",
            "link_types": "ethernet",
            "compute_score": 512.0,
            "data_port": 50052,
        }
        values.update(overrides)
        return ServiceRegistry()._build_properties(**values)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"node_id": "n" * 64},
            {"gpu_cores": -1},
            {"gpu_cores": 1025},
            {"ram_gb": 65537},
            {"chip_name": "M4\nforged-output"},
            {"link_types": "l" * 129},
            {"compute_score": float("nan")},
            {"data_port": 0},
            {"data_port": 65536},
        ],
    )
    def test_invalid_local_property_is_rejected(self, overrides):
        with pytest.raises(ValueError):
            self._properties(**overrides)

    def test_rich_topology_is_trimmed_to_mdns_txt_limit(self):
        links = tuple(
            NetworkLink(
                interface=f"en{index}",
                link_type=LinkType.ETHERNET,
                ip_address=f"192.0.2.{index + 1}",
                bandwidth_mbps=1000.0,
                latency_ms=1.0,
                loss_rate=0.01,
                mtu=9000,
            )
            for index in range(8)
        )

        properties = self._properties(network_links=links)
        encoded_links = properties[b"network_links"]

        assert len(b"network_links=") + len(encoded_links) <= 255
        restored = deserialize_network_links(encoded_links.decode())
        assert 0 < len(restored) < len(links)
        ServiceInfo(
            SERVICE_TYPE,
            f"node-1.{SERVICE_TYPE}",
            addresses=[socket.inet_aton("192.0.2.100")],
            port=50051,
            properties=properties,
            server="node-1.local.",
        )
