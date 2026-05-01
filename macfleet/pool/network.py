"""Network interface detection and link scoring.

Detects WiFi, Ethernet, and Thunderbolt interfaces, measures bandwidth/latency,
and produces composite scores for routing tensor traffic through the best link.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LinkType(Enum):
    """Network link types, ordered by typical bandwidth."""
    WIFI = "wifi"
    ETHERNET = "ethernet"
    THUNDERBOLT = "thunderbolt"
    LOOPBACK = "loopback"
    UNKNOWN = "unknown"


@dataclass
class NetworkLink:
    """A single network interface with measured performance."""
    interface: str          # "en0", "bridge0", etc.
    link_type: LinkType
    ip_address: str
    bandwidth_mbps: float = 0.0    # measured, not theoretical
    latency_ms: float = 0.0        # measured RTT to peer
    loss_rate: float = 0.0         # measured packet loss %
    mtu: int = 1500

    @property
    def score(self) -> float:
        """Composite score for link selection. Higher is better."""
        if self.latency_ms <= 0 or self.bandwidth_mbps <= 0:
            return 0.0
        return self.bandwidth_mbps * (1.0 - self.loss_rate) / max(self.latency_ms, 0.1)

    @property
    def theoretical_bandwidth_mbps(self) -> float:
        """Theoretical max bandwidth based on link type."""
        return {
            LinkType.THUNDERBOLT: 40_000.0,
            LinkType.ETHERNET: 1_000.0,
            LinkType.WIFI: 300.0,
            LinkType.LOOPBACK: 100_000.0,
            LinkType.UNKNOWN: 100.0,
        }[self.link_type]


@dataclass
class NetworkTopology:
    """Complete network topology of this node."""
    links: list[NetworkLink] = field(default_factory=list)
    hostname: str = ""

    @property
    def best_link(self) -> Optional[NetworkLink]:
        """The highest-scored available link."""
        scored = [l for l in self.links if l.score > 0]
        if not scored:
            return self.links[0] if self.links else None
        return max(scored, key=lambda l: l.score)

    @property
    def has_thunderbolt(self) -> bool:
        return any(l.link_type == LinkType.THUNDERBOLT for l in self.links)

    @property
    def has_ethernet(self) -> bool:
        return any(l.link_type == LinkType.ETHERNET for l in self.links)

    @property
    def has_wifi(self) -> bool:
        return any(l.link_type == LinkType.WIFI for l in self.links)


def _classify_interface(name: str, ip: str) -> LinkType:
    """Classify a network interface by name and IP patterns."""
    if name == "lo0" or ip == "127.0.0.1":
        return LinkType.LOOPBACK

    # Thunderbolt bridge interfaces
    if name.startswith("bridge") or name.startswith("tb"):
        return LinkType.THUNDERBOLT

    # Thunderbolt IPs (link-local or 10.x.x.x on bridge)
    if ip.startswith("169.254.") or (ip.startswith("10.0.0.") and name.startswith("bridge")):
        return LinkType.THUNDERBOLT

    # macOS en0 is typically WiFi on laptops, but Ethernet on desktops
    # Use system_profiler or networksetup to determine
    if name.startswith("en"):
        if _is_wifi_interface(name):
            return LinkType.WIFI
        return LinkType.ETHERNET

    return LinkType.UNKNOWN


def _is_wifi_interface(interface: str) -> bool:
    """Check if an interface is WiFi using networksetup.

    When networksetup is unavailable (sandboxed runs, restricted user,
    timeout), we no longer assume en0 is WiFi. On Mac mini / Studio /
    iMac the en0 jack is wired Ethernet, and an over-eager WiFi
    classification would push the AdaptiveCompressor into AGGRESSIVE
    mode on what is actually a Gbit/s+ link. Return False on the
    fallback so the caller treats the interface as Ethernet.
    """
    try:
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for i, line in enumerate(lines):
                if f"Device: {interface}" in line:
                    # Check the hardware port line above
                    if i > 0 and "wi-fi" in lines[i - 1].lower():
                        return True
                    return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def detect_interfaces() -> list[NetworkLink]:
    """Detect all active network interfaces and classify them.

    Returns:
        List of NetworkLink objects with IP and link type.
    """
    links = []
    try:
        # Use socket to get all interfaces with IPs
        hostname = socket.gethostname()
        # Get all IPs for this host
        addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)
        seen_ips = set()

        for addr_info in addrs:
            ip = addr_info[4][0]
            if ip in seen_ips or ip == "127.0.0.1":
                continue
            seen_ips.add(ip)

            # We don't have interface name from getaddrinfo,
            # so use ifconfig to map IPs to interfaces
            interface = _ip_to_interface(ip)
            if interface:
                link_type = _classify_interface(interface, ip)
                links.append(NetworkLink(
                    interface=interface,
                    link_type=link_type,
                    ip_address=ip,
                    bandwidth_mbps=0.0,
                    latency_ms=0.0,
                ))
    except (socket.gaierror, OSError):
        pass

    # Fallback: parse ifconfig directly
    if not links:
        links = _parse_ifconfig()

    return links


def _ip_to_interface(ip: str) -> Optional[str]:
    """Map an IP address to its interface name using ifconfig."""
    try:
        result = subprocess.run(
            ["ifconfig"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            current_iface = None
            for line in result.stdout.split("\n"):
                if not line.startswith("\t") and ":" in line:
                    current_iface = line.split(":")[0]
                if f"inet {ip} " in line and current_iface:
                    return current_iface
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _parse_ifconfig() -> list[NetworkLink]:
    """Parse ifconfig output to get all interfaces."""
    links = []
    try:
        result = subprocess.run(
            ["ifconfig"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return links

        current_iface = None
        for line in result.stdout.split("\n"):
            if not line.startswith("\t") and ":" in line:
                current_iface = line.split(":")[0]
            if "inet " in line and current_iface and current_iface != "lo0":
                parts = line.strip().split()
                if len(parts) >= 2:
                    ip = parts[1]
                    link_type = _classify_interface(current_iface, ip)
                    links.append(NetworkLink(
                        interface=current_iface,
                        link_type=link_type,
                        ip_address=ip,
                    ))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return links


async def measure_latency(host: str, port: int, timeout: float = 5.0) -> float:
    """Measure TCP round-trip latency to a peer in milliseconds."""
    try:
        start = time.monotonic()
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        elapsed = (time.monotonic() - start) * 1000.0
        writer.close()
        await writer.wait_closed()
        return elapsed
    except (OSError, asyncio.TimeoutError):
        return -1.0


async def measure_bandwidth(
    host: str,
    port: int,
    payload_mb: float = 10.0,
    timeout: float = 30.0,
) -> float:
    """Measure TCP bandwidth to a peer in Mbps.

    Sends a payload and measures transfer time.
    Requires a listening peer that accepts and discards data.
    """
    payload_bytes = int(payload_mb * 1024 * 1024)
    chunk_size = 65536

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )

        data = b"\x00" * chunk_size
        start = time.monotonic()
        sent = 0
        while sent < payload_bytes:
            to_send = min(chunk_size, payload_bytes - sent)
            writer.write(data[:to_send])
            await writer.drain()
            sent += to_send

        elapsed = time.monotonic() - start
        writer.close()
        await writer.wait_closed()

        if elapsed > 0:
            return (payload_bytes * 8) / (elapsed * 1_000_000)  # Mbps
        return 0.0
    except (OSError, asyncio.TimeoutError):
        return 0.0


def get_network_topology() -> NetworkTopology:
    """Detect and return full network topology for this node."""
    hostname = socket.gethostname()
    links = detect_interfaces()
    return NetworkTopology(links=links, hostname=hostname)
