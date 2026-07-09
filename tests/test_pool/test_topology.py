"""Tests for pool topology helpers."""

import pytest

from macfleet.pool.network import LinkType, NetworkLink
from macfleet.pool.topology import (
    NodeTopology,
    best_peer_address,
    deserialize_network_links,
    serialize_network_links,
)


def test_best_peer_address_prefers_shared_thunderbolt():
    local = NodeTopology(
        node_id="a",
        default_ip="192.168.1.10",
        links=(
            NetworkLink("en0", LinkType.WIFI, "192.168.1.10"),
            NetworkLink("bridge0", LinkType.THUNDERBOLT, "10.0.0.10"),
        ),
    )
    peer = NodeTopology(
        node_id="b",
        default_ip="192.168.1.20",
        links=(
            NetworkLink("en0", LinkType.WIFI, "192.168.1.20"),
            NetworkLink("bridge0", LinkType.THUNDERBOLT, "10.0.0.20"),
        ),
    )

    assert best_peer_address(local, peer) == "10.0.0.20"


@pytest.mark.parametrize(
    ("node_id", "default_ip", "message"),
    [("", "192.168.1.10", "node_id"), ("node", "0.0.0.0", "default_ip"), ("node", "bad", "default_ip")],
)
def test_topology_requires_dialable_identity(node_id, default_ip, message):
    with pytest.raises(ValueError, match=message):
        NodeTopology(node_id=node_id, default_ip=default_ip)


def test_best_peer_address_falls_back_to_peer_default_ip():
    local = NodeTopology(node_id="a", default_ip="192.168.1.10")
    peer = NodeTopology(node_id="b", default_ip="192.168.1.20")

    assert best_peer_address(local, peer) == "192.168.1.20"


def test_best_peer_address_prefers_shared_type_over_peer_only_ethernet():
    local = NodeTopology(
        node_id="a",
        default_ip="192.168.1.10",
        links=(NetworkLink("en0", LinkType.WIFI, "192.168.1.10"),),
    )
    peer = NodeTopology(
        node_id="b",
        default_ip="192.168.1.20",
        links=(
            NetworkLink("en0", LinkType.WIFI, "192.168.1.20"),
            NetworkLink("en5", LinkType.ETHERNET, "172.16.0.20"),
        ),
    )

    assert best_peer_address(local, peer) == "192.168.1.20"


def test_best_peer_address_prefers_same_lan_over_off_subnet_faster_link():
    local = NodeTopology(
        node_id="a",
        default_ip="192.168.1.10",
        links=(
            NetworkLink("en0", LinkType.WIFI, "192.168.1.10"),
            NetworkLink("en5", LinkType.ETHERNET, "10.1.0.10"),
        ),
    )
    peer = NodeTopology(
        node_id="b",
        default_ip="192.168.1.20",
        links=(
            NetworkLink("en0", LinkType.WIFI, "192.168.1.20"),
            NetworkLink("en5", LinkType.ETHERNET, "10.2.0.20"),
        ),
    )

    assert best_peer_address(local, peer) == "192.168.1.20"


def test_best_peer_address_skips_ipv6_link_local_without_scope():
    local = NodeTopology(
        node_id="a",
        default_ip="192.168.1.10",
        links=(
            NetworkLink("en0", LinkType.WIFI, "192.168.1.10"),
            NetworkLink("en0", LinkType.WIFI, "fe80::1"),
        ),
    )
    peer = NodeTopology(
        node_id="b",
        default_ip="192.168.1.20",
        links=(
            NetworkLink("en0", LinkType.WIFI, "fe80::2"),
            NetworkLink("en0", LinkType.WIFI, "192.168.1.20"),
        ),
    )

    assert best_peer_address(local, peer) == "192.168.1.20"


def test_network_link_serialization_round_trips_and_skips_loopback():
    links = (
        NetworkLink(
            "en0",
            LinkType.WIFI,
            "192.168.1.10",
            bandwidth_mbps=100,
            latency_ms=4,
            loss_rate=0.01,
            mtu=1500,
        ),
        NetworkLink("lo0", LinkType.LOOPBACK, "127.0.0.1"),
        NetworkLink("en0", LinkType.WIFI, "fe80::1"),
    )

    restored = deserialize_network_links(serialize_network_links(links))

    assert len(restored) == 1
    assert restored[0].interface == "en0"
    assert restored[0].link_type == LinkType.WIFI
    assert restored[0].ip_address == "192.168.1.10"
    assert restored[0].bandwidth_mbps == 100


def test_deserialize_network_links_ignores_bad_payloads():
    assert deserialize_network_links("not-json") == ()
    assert deserialize_network_links('{"not":"a list"}') == ()
