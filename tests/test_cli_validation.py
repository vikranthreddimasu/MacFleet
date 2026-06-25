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
