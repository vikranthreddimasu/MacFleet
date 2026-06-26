"""Tests for `macfleet status` CLI behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass

from click.testing import CliRunner

from macfleet.cli.main import cli


@dataclass
class FakeNode:
    hostname: str
    node_id: str
    ip_address: str
    port: int
    data_port: int
    gpu_cores: int
    ram_gb: int
    chip_name: str
    link_types: str
    pool_version: str
    compute_score: float

    @property
    def link_type_list(self) -> list[str]:
        return [part.strip() for part in self.link_types.split(",") if part.strip()]


def test_status_reports_token_read_failure_without_traceback(monkeypatch):
    def fail_to_read_token(token=None):
        raise PermissionError(13, "permission denied", "/tmp/fleet-token")

    monkeypatch.setattr("macfleet.security.auth.resolve_token_with_file", fail_to_read_token)

    runner = CliRunner()
    result = runner.invoke(cli, ["status"])

    assert result.exit_code == 1
    assert "couldn't read fleet token" in result.output
    assert "status --open" in result.output
    assert "Traceback" not in result.output


def test_status_open_skips_saved_token_read(monkeypatch):
    def fail_to_read_token(token=None):
        raise AssertionError("open status should not read the saved token")

    class EmptyRegistry:
        def __init__(self, security=None):
            self.security = security

        def find_peers(self, timeout=3.0):
            return []

        def stop(self):
            pass

    monkeypatch.setattr("macfleet.security.auth.resolve_token_with_file", fail_to_read_token)
    monkeypatch.setattr("macfleet.pool.discovery.ServiceRegistry", EmptyRegistry)

    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--open"])

    assert result.exit_code == 0
    assert "No pool members found" in result.output


def test_status_json_outputs_empty_pool_without_human_banner(monkeypatch):
    class EmptyRegistry:
        def __init__(self, security=None):
            self.security = security

        def find_peers(self, timeout=3.0):
            return []

        def stop(self):
            pass

    monkeypatch.setattr("macfleet.pool.discovery.ServiceRegistry", EmptyRegistry)

    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--open", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "secure": False,
        "fleet_id": None,
        "count": 0,
        "nodes": [],
    }
    assert "Scanning" not in result.output
    assert "No pool members" not in result.output


def test_status_json_outputs_sorted_nodes(monkeypatch):
    nodes = [
        FakeNode(
            hostname="slow.local",
            node_id="slow",
            ip_address="192.168.1.20",
            port=50051,
            data_port=50052,
            gpu_cores=8,
            ram_gb=16,
            chip_name="Apple M1",
            link_types="wifi",
            pool_version="2.2.0",
            compute_score=100.0,
        ),
        FakeNode(
            hostname="fast.local",
            node_id="fast",
            ip_address="192.168.1.10",
            port=50061,
            data_port=50062,
            gpu_cores=32,
            ram_gb=64,
            chip_name="Apple M3 Max",
            link_types="ethernet,thunderbolt",
            pool_version="2.2.0",
            compute_score=900.0,
        ),
    ]

    class NodeRegistry:
        def __init__(self, security=None):
            self.security = security

        def find_peers(self, timeout=3.0):
            return nodes

        def stop(self):
            pass

    monkeypatch.setattr("macfleet.pool.discovery.ServiceRegistry", NodeRegistry)

    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--open", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["count"] == 2
    assert [node["node_id"] for node in payload["nodes"]] == ["fast", "slow"]
    assert payload["nodes"][0] == {
        "hostname": "fast.local",
        "node_id": "fast",
        "ip_address": "192.168.1.10",
        "heartbeat_port": 50061,
        "data_port": 50062,
        "gpu_cores": 32,
        "ram_gb": 64,
        "chip_name": "Apple M3 Max",
        "link_types": ["ethernet", "thunderbolt"],
        "pool_version": "2.2.0",
        "compute_score": 900.0,
    }
