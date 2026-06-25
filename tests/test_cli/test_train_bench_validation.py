"""Tests for training and benchmark CLI input validation."""

from __future__ import annotations

from click.testing import CliRunner

from macfleet.cli.main import cli


def test_train_rejects_invalid_numeric_options_before_runtime(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("training runtime should not run for invalid CLI input")

    monkeypatch.setattr("macfleet.cli.main._train_demo", fail_if_called)
    monkeypatch.setattr("macfleet.cli.main._train_from_script", fail_if_called)

    runner = CliRunner()
    cases = [
        (["train", "--epochs", "0"], "--epochs"),
        (["train", "--batch-size", "0"], "--batch-size"),
        (["train", "--lr", "0"], "--lr"),
        (["train", "--lr", "-0.1"], "--lr"),
    ]

    for args, option in cases:
        result = runner.invoke(cli, args)
        assert result.exit_code == 2
        assert f"Invalid value for '{option}'" in result.output


def test_bench_rejects_invalid_numeric_options_before_runtime(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("benchmark runtime should not run for invalid CLI input")

    monkeypatch.setattr("macfleet.cli.main._bench_compute", fail_if_called)
    monkeypatch.setattr("macfleet.cli.main._bench_network", fail_if_called)
    monkeypatch.setattr("macfleet.cli.main._bench_allreduce", fail_if_called)

    runner = CliRunner()
    cases = [
        (["bench", "--iterations", "0"], "--iterations"),
        (["bench", "--iterations", "-1"], "--iterations"),
        (["bench", "--size-mb", "0"], "--size-mb"),
        (["bench", "--size-mb", "4097"], "--size-mb"),
    ]

    for args, option in cases:
        result = runner.invoke(cli, args)
        assert result.exit_code == 2
        assert f"Invalid value for '{option}'" in result.output
