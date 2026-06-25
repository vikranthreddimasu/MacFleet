"""Tests for `macfleet join` startup failure handling."""

from __future__ import annotations

import errno

from click.testing import CliRunner

from macfleet.cli.main import cli


def test_join_rejects_invalid_port_before_runtime(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("PoolAgent should not be constructed for invalid CLI input")

    monkeypatch.setattr("macfleet.pool.agent.PoolAgent", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(cli, ["join", "--port", "-1"])

    assert result.exit_code == 2
    assert "Invalid value for '--port'" in result.output


def test_join_rejects_invalid_data_port_before_runtime(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("PoolAgent should not be constructed for invalid CLI input")

    monkeypatch.setattr("macfleet.pool.agent.PoolAgent", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(cli, ["join", "--data-port", "70000"])

    assert result.exit_code == 2
    assert "Invalid value for '--data-port'" in result.output


def test_join_rejects_nonpositive_enrollment_options_before_runtime(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("PoolAgent should not be constructed for invalid CLI input")

    monkeypatch.setattr("macfleet.pool.agent.PoolAgent", fail_if_called)

    runner = CliRunner()
    ttl_result = runner.invoke(cli, ["join", "--enroll-ttl", "0"])
    uses_result = runner.invoke(cli, ["join", "--enroll-uses", "0"])

    assert ttl_result.exit_code == 2
    assert "Invalid value for '--enroll-ttl'" in ttl_result.output
    assert uses_result.exit_code == 2
    assert "Invalid value for '--enroll-uses'" in uses_result.output


def test_join_reports_agent_start_failure_and_cleans_up(monkeypatch):
    events: list[str] = []

    class FailingAgent:
        def __init__(self, **kwargs):
            self.node_id = "fake-node"
            events.append("init")

        async def start(self):
            events.append("start")
            raise OSError(errno.EADDRINUSE, "Address already in use")

        async def stop(self):
            events.append("stop")

    monkeypatch.setattr("macfleet.pool.agent.PoolAgent", FailingAgent)
    monkeypatch.setattr("macfleet.security.audit.audit_event", lambda *args, **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["join", "--open", "--allow-insecure-open"])

    assert result.exit_code == 1
    assert "couldn't start MacFleet agent" in result.output
    assert "Port conflict detected" in result.output
    assert "Traceback" not in result.output
    assert events == ["init", "start", "stop"]


def test_join_reports_enrollment_start_failure_and_stops_agent(monkeypatch):
    events: list[str] = []

    class StartedAgent:
        def __init__(self, **kwargs):
            self.node_id = "fake-node"
            events.append("agent-init")

        async def start(self):
            events.append("agent-start")

        async def stop(self):
            events.append("agent-stop")

    class FailingEnrollmentServer:
        def __init__(self, **kwargs):
            events.append("enrollment-init")

        async def start(self):
            events.append("enrollment-start")
            raise OSError(errno.EADDRINUSE, "Address already in use")

        async def stop(self):
            events.append("enrollment-stop")

    monkeypatch.setattr("macfleet.pool.agent.PoolAgent", StartedAgent)
    monkeypatch.setattr("macfleet.security.enrollment.EnrollmentServer", FailingEnrollmentServer)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["join", "--token", "this-token-is-long-enough", "--bootstrap"],
    )

    assert result.exit_code == 1
    assert "couldn't start enrollment server" in result.output
    assert "Traceback" not in result.output
    assert events == [
        "agent-init",
        "agent-start",
        "enrollment-init",
        "enrollment-start",
        "enrollment-stop",
        "agent-stop",
    ]
