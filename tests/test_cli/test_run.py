"""Tests for `macfleet run` CLI safety behavior."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from macfleet.cli.main import cli


def test_run_open_requires_explicit_insecure_ack():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "missing.py", "--open"])

    assert result.exit_code == 1
    assert "--allow-insecure-open" in result.output
    assert "authentication" in result.output
    assert "Script not found" not in result.output


def test_run_open_and_token_are_mutually_exclusive():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "missing.py", "--open", "--token", "this-token-is-long-enough"],
    )

    assert result.exit_code == 1
    assert "mutually exclusive" in result.output
    assert "Script not found" not in result.output


def test_run_open_with_ack_passes_open_mode_to_pool(
    tmp_path: Path,
    monkeypatch,
):
    script = tmp_path / "job.py"
    script.write_text("def main():\n    return 'done'\n")
    created: list[dict[str, object]] = []

    class FakePool:
        def __init__(self, *, token=None, open=False):
            created.append({"token": token, "open": open})

        def __enter__(self):
            return self

        def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
            return None

        def run(self, fn):
            return fn()

    monkeypatch.setattr("macfleet.sdk.pool.Pool", FakePool)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", str(script), "--open", "--allow-insecure-open"],
    )

    assert result.exit_code == 0
    assert created == [{"token": None, "open": True}]
    assert "Result: done" in result.output


def test_run_help_includes_insecure_ack_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--allow-insecure-open" in result.output
