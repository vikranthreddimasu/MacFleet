"""Lightweight pool topology helpers.

This module stays framework-agnostic and dependency-free. It turns the
per-node interface facts already collected by ``pool.network`` into stable
address choices for the training mesh.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Iterable, Sequence

from macfleet.pool.network import LinkType, NetworkLink

_MAX_SERIALIZED_LINKS = 8

_LINK_PRIORITY = {
    LinkType.LOOPBACK: 50,
    LinkType.THUNDERBOLT: 40,
    LinkType.ETHERNET: 30,
    LinkType.WIFI: 20,
    LinkType.UNKNOWN: 10,
}


def _parsed_address(value: str) -> IPv4Address | IPv6Address | None:
    try:
        return ip_address(value)
    except ValueError:
        return None


def _is_dialable_address(value: str) -> bool:
    parsed = _parsed_address(value)
    if parsed is None:
        return False
    if parsed.is_unspecified or parsed.is_multicast:
        return False
    # macOS requires a scope id for IPv6 link-local addresses. We strip the
    # zone suffix in network detection, so do not select or advertise them.
    return not (parsed.version == 6 and parsed.is_link_local)


def _same_lan_score(left: str, right: str) -> int:
    left_ip = _parsed_address(left)
    right_ip = _parsed_address(right)
    if left_ip is None or right_ip is None or left_ip.version != right_ip.version:
        return 0
    if left_ip == right_ip:
        return 400
    if left_ip.version == 4 and isinstance(left_ip, IPv4Address):
        right_v4 = right_ip
        if not isinstance(right_v4, IPv4Address):
            return 0
        if left_ip.is_link_local and right_v4.is_link_local:
            return 250
        if int(left_ip) >> 8 == int(right_v4) >> 8:
            return 300
    elif left_ip.version == 6 and isinstance(left_ip, IPv6Address):
        right_v6 = right_ip
        if (
            isinstance(right_v6, IPv6Address)
            and left_ip.packed[:8] == right_v6.packed[:8]
        ):
            return 300
    return 0


@dataclass(frozen=True)
class NodeTopology:
    """Network facts known for one node."""

    node_id: str
    default_ip: str
    links: tuple[NetworkLink, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty string")
        if not isinstance(self.default_ip, str) or not _is_dialable_address(self.default_ip):
            raise ValueError("default_ip must be a dialable IP address")

    @property
    def link_types(self) -> frozenset[LinkType]:
        return frozenset(link.link_type for link in self.links)

    def candidate_addresses(
        self,
        *,
        shared_link_types: Iterable[LinkType] = (),
    ) -> list[NetworkLink]:
        """Return peer addresses ordered from most to least preferred."""
        shared = set(shared_link_types)

        def key(link: NetworkLink) -> tuple[int, float, float, str, str]:
            shared_bonus = 100 if link.link_type in shared else 0
            measured = link.score
            theoretical = link.theoretical_bandwidth_mbps
            return (
                shared_bonus + _LINK_PRIORITY[link.link_type],
                measured,
                theoretical,
                link.interface,
                link.ip_address,
            )

        candidates = [
            link for link in self.links if _is_dialable_address(link.ip_address)
        ]
        ordered = sorted(candidates, key=key, reverse=True)
        if not any(link.ip_address == self.default_ip for link in ordered):
            ordered.append(
                NetworkLink(
                    interface="default",
                    link_type=LinkType.UNKNOWN,
                    ip_address=self.default_ip,
                )
            )
        return ordered


def best_peer_address(local: NodeTopology, peer: NodeTopology) -> str:
    """Choose the best address for ``local`` to dial on ``peer``.

    Preference is deterministic and based on link classes both nodes appear
    to share. If richer topology is unavailable, this falls back to
    ``peer.default_ip``.
    """
    if not local.links or not peer.links:
        return peer.default_ip

    local_candidates = [
        link for link in local.links if _is_dialable_address(link.ip_address)
    ]
    peer_candidates = peer.candidate_addresses()
    if not local_candidates or not peer_candidates:
        return peer.default_ip

    def pair_score(local_link: NetworkLink, peer_link: NetworkLink) -> int:
        score = _same_lan_score(local_link.ip_address, peer_link.ip_address)
        if local_link.link_type == peer_link.link_type:
            score += 100
        return score

    def candidate_key(
        link: NetworkLink,
    ) -> tuple[int, int, int, float, float, str, str]:
        best_pair = max(pair_score(local_link, link) for local_link in local_candidates)
        default_bonus = 25 if link.ip_address == peer.default_ip else 0
        return (
            best_pair,
            _LINK_PRIORITY[link.link_type],
            default_bonus,
            link.score,
            link.theoretical_bandwidth_mbps,
            link.interface,
            link.ip_address,
        )

    selected = max(peer_candidates, key=candidate_key)
    if _is_dialable_address(selected.ip_address):
        return selected.ip_address
    return peer.default_ip


def serialize_network_links(links: Sequence[NetworkLink]) -> str:
    """Serialize links for mDNS TXT records or registry snapshots."""
    payload = []
    for link in links:
        if link.link_type == LinkType.LOOPBACK:
            continue
        if not _is_dialable_address(link.ip_address):
            continue
        item: dict[str, str | float | int] = {
            "i": link.interface,
            "t": link.link_type.value,
            "ip": link.ip_address,
        }
        if link.bandwidth_mbps:
            item["b"] = link.bandwidth_mbps
        if link.latency_ms:
            item["l"] = link.latency_ms
        if link.loss_rate:
            item["r"] = link.loss_rate
        if link.mtu != 1500:
            item["m"] = link.mtu
        payload.append(item)
        if len(payload) >= _MAX_SERIALIZED_LINKS:
            break
    return json.dumps(payload, separators=(",", ":"))


def deserialize_network_links(payload: str) -> tuple[NetworkLink, ...]:
    """Parse serialized links, ignoring malformed entries."""
    try:
        raw = json.loads(payload)
    except (TypeError, ValueError):
        return ()
    if not isinstance(raw, list):
        return ()

    links: list[NetworkLink] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            link_type = LinkType(str(item.get("t", item.get("link_type"))))
            interface = item.get("i", item.get("interface"))
            ip_address = item.get("ip", item.get("ip_address"))
            if (
                not isinstance(interface, str)
                or not interface
                or not isinstance(ip_address, str)
                or not _is_dialable_address(ip_address)
                or link_type == LinkType.LOOPBACK
            ):
                continue
            bandwidth_mbps = float(item.get("b") or item.get("bandwidth_mbps") or 0.0)
            latency_ms = float(item.get("l") or item.get("latency_ms") or 0.0)
            loss_rate = float(item.get("r") or item.get("loss_rate") or 0.0)
            mtu = int(item.get("m") or item.get("mtu") or 1500)
            if (
                not all(math.isfinite(value) and value >= 0 for value in (
                    bandwidth_mbps,
                    latency_ms,
                    loss_rate,
                ))
                or mtu < 1
            ):
                continue
            links.append(
                NetworkLink(
                    interface=interface,
                    link_type=link_type,
                    ip_address=ip_address,
                    bandwidth_mbps=bandwidth_mbps,
                    latency_ms=latency_ms,
                    loss_rate=loss_rate,
                    mtu=mtu,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(links)
