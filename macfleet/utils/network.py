"""Network utilities for MacFleet.

Handles Thunderbolt bridge detection, IP address discovery, and port management.
"""

import ipaddress
import re
import socket
import subprocess
from typing import Optional

# Default Thunderbolt bridge IP range (10.0.0.x or 169.254.x.x link-local)
TB_BRIDGE_PREFIXES = ("10.0.0.", "169.254.")

# Network interface names for Thunderbolt bridge on macOS
TB_INTERFACE_NAMES = ("bridge", "Thunderbolt", "en")

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def validate_port(port: int, field_name: str = "port") -> int:
    """Validate a TCP port number."""
    try:
        port_int = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {port!r}") from exc

    if not (1 <= port_int <= 65535):
        raise ValueError(f"{field_name} must be 1-65535, got {port_int}")
    return port_int


def validate_ip_address(ip_address: str, field_name: str = "ip_address") -> str:
    """Validate an IPv4 or IPv6 address string."""
    value = str(ip_address).strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")

    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid IP address, got {ip_address!r}") from exc

    return value


def validate_host(host: str, field_name: str = "host") -> str:
    """Validate an IP address or DNS hostname."""
    value = str(host).strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"{field_name} must not contain whitespace, got {host!r}")

    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass

    if len(value) > 253:
        raise ValueError(f"{field_name} is too long")

    labels = value.rstrip(".").split(".")
    if not labels or any(not _HOSTNAME_RE.match(label) for label in labels):
        raise ValueError(f"{field_name} must be a valid IP address or hostname, got {host!r}")

    return value


def parse_endpoint(
    endpoint: str,
    default_port: int,
    field_name: str = "endpoint",
) -> tuple[str, int]:
    """Parse and validate host[:port] endpoint strings.

    IPv6 literals with an explicit port must use bracket notation, for example
    ``[::1]:50051``. Bare IPv6 literals are accepted with the default port.
    """
    value = str(endpoint).strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")

    port = validate_port(default_port, f"{field_name} port")

    if value.startswith("["):
        closing = value.find("]")
        if closing == -1:
            raise ValueError(f"{field_name} has an invalid bracketed IPv6 host")
        host = value[1:closing]
        remainder = value[closing + 1:]
        if remainder:
            if not remainder.startswith(":"):
                raise ValueError(f"{field_name} has unexpected text after host")
            port = validate_port(remainder[1:], f"{field_name} port")
        return validate_host(host, f"{field_name} host"), port

    if value.count(":") == 1:
        host, raw_port = value.rsplit(":", 1)
        if not host:
            raise ValueError(f"{field_name} host must not be empty")
        port = validate_port(raw_port, f"{field_name} port")
        return validate_host(host, f"{field_name} host"), port

    return validate_host(value, f"{field_name} host"), port


def get_hostname() -> str:
    """Get the local hostname."""
    return socket.gethostname()


def get_local_ip() -> str:
    """Get the local IP address (non-loopback).

    Returns the first non-loopback IP found, preferring
    Thunderbolt bridge IPs if available.
    """
    # Try to find Thunderbolt bridge IP first
    tb_ip = get_thunderbolt_bridge_ip()
    if tb_ip:
        return tb_ip

    # Fall back to any non-loopback IP
    try:
        # Connect to an external address to find the default route IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def get_thunderbolt_bridge_ip() -> Optional[str]:
    """Detect the Thunderbolt bridge IP address on macOS.

    Looks for network interfaces with Thunderbolt bridge IPs
    (typically 10.0.0.x or 169.254.x.x link-local).

    Returns:
        Thunderbolt bridge IP if found, None otherwise.
    """
    try:
        # Use ifconfig to get all network interfaces
        result = subprocess.run(
            ["ifconfig"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        output = result.stdout
        for line in output.split("\n"):
            # Look for inet addresses
            if "inet " in line and "127.0.0.1" not in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    ip = parts[1]
                    # Check if it's a Thunderbolt bridge IP
                    if any(ip.startswith(prefix) for prefix in TB_BRIDGE_PREFIXES):
                        return ip

        return None

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_all_local_ips() -> list[str]:
    """Get all non-loopback local IP addresses.

    Returns:
        List of IP addresses.
    """
    ips = []
    try:
        hostname = socket.gethostname()
        # Get all addresses for this host
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip != "127.0.0.1" and ip not in ips:
                ips.append(ip)
    except socket.gaierror:
        pass

    # Also try ifconfig parsing
    try:
        result = subprocess.run(
            ["ifconfig"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "inet " in line and "127.0.0.1" not in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        if ip not in ips:
                            ips.append(ip)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return ips


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available for binding.

    Args:
        port: Port number to check.
        host: Host to bind to.

    Returns:
        True if the port is available.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(start_port: int = 50051, max_attempts: int = 100) -> int:
    """Find an available port starting from the given port.

    Args:
        start_port: Port to start searching from.
        max_attempts: Maximum number of ports to try.

    Returns:
        Available port number.

    Raises:
        RuntimeError: If no available port found.
    """
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port):
            return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def is_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a host:port is reachable.

    Args:
        host: Host to connect to.
        port: Port to connect to.
        timeout: Connection timeout in seconds.

    Returns:
        True if reachable.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            return True
    except (OSError, socket.timeout):
        return False


def ping_host(host: str, timeout: float = 2.0) -> bool:
    """Ping a host to check if it's reachable.

    Args:
        host: Host to ping.
        timeout: Timeout in seconds.

    Returns:
        True if host responds to ping.
    """
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(int(timeout * 1000)), host],
            capture_output=True,
            timeout=timeout + 1,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_gpu_info() -> dict:
    """Get GPU information on macOS (Apple Silicon).

    Returns:
        Dictionary with gpu_cores and gpu_name.
    """
    gpu_info = {"gpu_cores": 0, "gpu_name": "Unknown"}

    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            output = result.stdout
            for line in output.split("\n"):
                if "Total Number of Cores" in line:
                    # Extract number of cores
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            gpu_info["gpu_cores"] = int(parts[1].strip())
                        except ValueError:
                            pass
                elif "Chipset Model" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        gpu_info["gpu_name"] = parts[1].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return gpu_info


def get_memory_info() -> dict:
    """Get memory information on macOS.

    Returns:
        Dictionary with total_gb and available_gb.
    """
    memory_info = {"total_gb": 0, "available_gb": 0}

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            total_bytes = int(result.stdout.strip())
            memory_info["total_gb"] = total_bytes // (1024 ** 3)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return memory_info


def get_memory_bandwidth() -> float:
    """Estimate memory bandwidth based on chip type.

    Returns approximate bandwidth in GB/s based on Apple Silicon chip.
    """
    # Get chip name from system_profiler
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            chip_name = result.stdout.strip().lower()

            # Memory bandwidth estimates for different chips
            if "m4 pro" in chip_name or "m4 max" in chip_name:
                return 273.0
            elif "m4" in chip_name:
                return 120.0
            elif "m3 pro" in chip_name or "m3 max" in chip_name:
                return 200.0
            elif "m3" in chip_name:
                return 100.0
            elif "m2 pro" in chip_name or "m2 max" in chip_name:
                return 200.0
            elif "m2" in chip_name:
                return 100.0
            elif "m1 pro" in chip_name or "m1 max" in chip_name:
                return 200.0
            elif "m1" in chip_name:
                return 68.0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Default estimate
    return 100.0


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Human-readable string (e.g., "1.5 GB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_bandwidth(bytes_per_sec: float) -> str:
    """Format bandwidth as human-readable string.

    Args:
        bytes_per_sec: Bytes per second.

    Returns:
        Human-readable string (e.g., "1.5 GB/s").
    """
    return f"{format_bytes(int(bytes_per_sec))}/s"
