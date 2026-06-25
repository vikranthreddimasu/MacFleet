"""Tests for `macfleet status` CLI behavior."""

from __future__ import annotations

from click.testing import CliRunner

from macfleet.cli.main import cli


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
