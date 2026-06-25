"""Tests for CLI input validation."""

from click.testing import CliRunner

from macfleet.cli.main import cli


def test_status_rejects_invalid_master_port():
    result = CliRunner().invoke(cli, ["status", "--master", "10.0.0.1:notaport"])

    assert result.exit_code != 0
    assert "Invalid --master" in result.output
    assert "port" in result.output


def test_launch_rejects_invalid_master_endpoint_before_running():
    result = CliRunner().invoke(
        cli,
        ["launch", "--role", "worker", "--master", "bad host:50051"],
    )

    assert result.exit_code != 0
    assert "Invalid --master" in result.output


def test_benchmark_rejects_invalid_sizes_before_running():
    result = CliRunner().invoke(cli, ["benchmark", "--sizes", "1,,10"])

    assert result.exit_code != 0
    assert "Invalid --sizes" in result.output


def test_diagnose_runs_local_checks(monkeypatch):
    monkeypatch.setattr("macfleet.cli.main.is_port_available", lambda port, host: True)

    result = CliRunner().invoke(cli, ["diagnose"])

    assert result.exit_code == 0
    assert "MacFleet Diagnostics" in result.output
    assert "Auth token" in result.output


def test_diagnose_rejects_invalid_host():
    result = CliRunner().invoke(cli, ["diagnose", "--host", "bad host"])

    assert result.exit_code != 0
    assert "--host" in result.output


def test_diagnose_checks_optional_master(monkeypatch):
    monkeypatch.setattr("macfleet.cli.main.is_port_available", lambda port, host: True)
    monkeypatch.setattr("macfleet.cli.main.is_reachable", lambda host, port, timeout: True)

    result = CliRunner().invoke(cli, ["diagnose", "--master", "10.0.0.1:50051"])

    assert result.exit_code == 0
    assert "Coordinator" in result.output
    assert "10.0.0.1:50051 is reachable" in result.output
