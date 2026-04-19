"""Tests for fleet-scoped mDNS discovery."""

from __future__ import annotations

from macfleet.pool.discovery import MACFLEET_SERVICE_TYPE, ServiceRegistry
from macfleet.security.auth import DEFAULT_SERVICE_TYPE, SecurityConfig


class TestFleetScopedServiceType:
    def test_default_service_type_constant(self):
        assert MACFLEET_SERVICE_TYPE == DEFAULT_SERVICE_TYPE

    def test_no_token_uses_default(self):
        cfg = SecurityConfig()
        assert cfg.mdns_service_type == DEFAULT_SERVICE_TYPE

    def test_token_produces_scoped_type(self):
        cfg = SecurityConfig(token="my-secret-token")
        stype = cfg.mdns_service_type
        assert stype != DEFAULT_SERVICE_TYPE
        assert stype.startswith("_mf-")
        assert stype.endswith("._tcp.local.")

    def test_same_token_same_service_type(self):
        a = SecurityConfig(token="same-long-token")
        b = SecurityConfig(token="same-long-token")
        assert a.mdns_service_type == b.mdns_service_type

    def test_different_tokens_different_service_types(self):
        a = SecurityConfig(token="token-a-long")
        b = SecurityConfig(token="token-b-long")
        assert a.mdns_service_type != b.mdns_service_type

    def test_different_fleet_ids_different_service_types(self):
        a = SecurityConfig(token="same-long-token", fleet_id="fleet-a")
        b = SecurityConfig(token="same-long-token", fleet_id="fleet-b")
        assert a.mdns_service_type != b.mdns_service_type

    def test_service_type_valid_mdns_format(self):
        """mDNS service types must match _<name>._tcp.local. format."""
        cfg = SecurityConfig(token="test-token")
        stype = cfg.mdns_service_type
        parts = stype.split(".")
        assert parts[0].startswith("_")
        assert parts[1] == "_tcp"
        assert parts[2] == "local"


class TestServiceRegistryWithSecurity:
    def test_registry_default_service_type(self):
        registry = ServiceRegistry()
        assert registry._service_type == DEFAULT_SERVICE_TYPE

    def test_registry_with_token(self):
        sec = SecurityConfig(token="my-secret-token")
        registry = ServiceRegistry(security=sec)
        assert registry._service_type == sec.mdns_service_type
        assert registry._service_type != DEFAULT_SERVICE_TYPE

    def test_registry_no_security_backward_compat(self):
        registry = ServiceRegistry()
        assert registry._service_type == "_macfleet._tcp.local."

    def test_two_registries_same_fleet(self):
        sec = SecurityConfig(token="fleet-token-long")
        r1 = ServiceRegistry(security=sec)
        r2 = ServiceRegistry(security=sec)
        assert r1._service_type == r2._service_type

    def test_two_registries_different_fleets(self):
        s1 = SecurityConfig(token="token-a-long")
        s2 = SecurityConfig(token="token-b-long")
        r1 = ServiceRegistry(security=s1)
        r2 = ServiceRegistry(security=s2)
        assert r1._service_type != r2._service_type


class TestMdnsInfoMinimization:
    """Verify that secure mode minimizes broadcast information."""

    def test_secure_properties_minimal(self):
        """Secure mode only broadcasts node_id, pool_version, and data_port."""
        sec = SecurityConfig(token="secret-token")
        registry = ServiceRegistry(security=sec)
        props = registry._build_properties(
            node_id="node-0", gpu_cores=32, ram_gb=64,
            chip_name="M2 Ultra", link_types="thunderbolt,ethernet",
            compute_score=42.0, data_port=50052,
        )
        assert b"node_id" in props
        assert b"pool_version" in props
        # data_port is broadcast even in secure mode — not sensitive, needed
        # by peers to initiate the authenticated handshake on the transport port
        assert b"data_port" in props
        assert props[b"data_port"] == b"50052"
        # Hardware details must NOT be broadcast in secure mode
        assert b"gpu_cores" not in props
        assert b"ram_gb" not in props
        assert b"chip_name" not in props
        assert b"link_types" not in props
        assert b"compute_score" not in props

    def test_open_properties_full(self):
        """Open mode broadcasts all hardware details and data_port."""
        registry = ServiceRegistry()
        props = registry._build_properties(
            node_id="node-0", gpu_cores=32, ram_gb=64,
            chip_name="M2 Ultra", link_types="thunderbolt",
            compute_score=42.0, data_port=50052,
        )
        assert b"node_id" in props
        assert b"gpu_cores" in props
        assert b"ram_gb" in props
        assert b"chip_name" in props
        assert b"link_types" in props
        assert b"compute_score" in props
        assert b"data_port" in props
        assert props[b"data_port"] == b"50052"
