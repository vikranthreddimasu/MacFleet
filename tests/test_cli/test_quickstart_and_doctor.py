"""Tests for the v2.2 PR 16 CLI additions: doctor + quickstart."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from macfleet.cli.main import cli


class TestQuickstart:
    def test_writes_demo_script(self, tmp_path: Path):
        target = tmp_path / "demo.py"
        runner = CliRunner()
        result = runner.invoke(cli, ["quickstart", "--output", str(target)])
        assert result.exit_code == 0, result.output
        assert target.exists()
        content = target.read_text()
        assert "macfleet.Pool" in content
        assert "TinyMLP" in content
        assert "enable_pool_distributed=False" in content
        assert "quorum_size=2" in content
        assert "stop the CLI agents on both Macs" in content

    def test_missing_parent_is_actionable(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["quickstart", "--output", str(tmp_path / "missing" / "demo.py")])
        assert result.exit_code == 1
        assert "Cannot write" in result.output
        assert "Traceback" not in result.output

    def test_writes_correct_filename_in_script(self, tmp_path: Path):
        """The generated script references its own filename in the docstring."""
        target = tmp_path / "my_run.py"
        runner = CliRunner()
        runner.invoke(cli, ["quickstart", "--output", str(target)])
        assert "python my_run.py" in target.read_text()

    def test_refuses_to_overwrite_without_force(self, tmp_path: Path):
        target = tmp_path / "demo.py"
        target.write_text("# original contents")

        runner = CliRunner()
        result = runner.invoke(cli, ["quickstart", "--output", str(target)])
        assert result.exit_code == 1
        # Rich may wrap output; normalize whitespace before assertion
        assert "already exists" in " ".join(result.output.split())
        # File must not have been clobbered
        assert target.read_text() == "# original contents"

    def test_force_overwrites(self, tmp_path: Path):
        target = tmp_path / "demo.py"
        target.write_text("# stale")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["quickstart", "--output", str(target), "--force"],
        )
        assert result.exit_code == 0
        assert "macfleet.Pool" in target.read_text()
        assert "# stale" not in target.read_text()

    def test_next_steps_shown(self, tmp_path: Path):
        target = tmp_path / "demo.py"
        runner = CliRunner()
        result = runner.invoke(cli, ["quickstart", "--output", str(target)])
        assert "pip install" in result.output
        assert "macfleet[torch]" in result.output
        # Rich may wrap long paths; check for the filename stem at minimum
        assert target.name in result.output
        assert "pair" in result.output


class TestDoctor:
    def test_doctor_runs(self):
        """`macfleet doctor` runs the same checks as diagnose and exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code in (0, 1)
        # Output includes the expected section headers
        assert "Hardware" in result.output
        assert "ML Frameworks" in result.output
        assert "Thermal" in result.output
        assert "Network" in result.output

    def test_doctor_json_outputs_machine_readable_report(self, monkeypatch):
        """`macfleet doctor --json` is parseable and does not leak tokens."""
        secret = "super-secret-token-for-json-doctor"
        monkeypatch.setenv("MACFLEET_TOKEN", secret)

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])

        assert "Running diagnostics" not in result.output
        assert secret not in result.output
        payload = json.loads(result.output)
        assert result.exit_code == (0 if payload["ready"] else 1)
        assert payload["total"] + payload["informational"] == len(payload["checks"])
        assert payload["passed"] + payload["failed"] == payload["total"]
        assert payload["status"] in {"ok", "fail"}
        assert isinstance(payload["ready"], bool)
        sections = {check["section"] for check in payload["checks"]}
        assert {
            "Runtime",
            "Hardware",
            "ML Frameworks",
            "Thermal",
            "Network",
            "Security",
        } <= sections
        assert all(
            {"id", "section", "name", "status", "passed", "detail"} <= set(check)
            for check in payload["checks"]
        )
        assert any(check["name"] == "Fleet token configured" for check in payload["checks"])

    def test_diagnose_json_alias_outputs_machine_readable_report(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["diagnose", "--json"])

        payload = json.loads(result.output)
        assert result.exit_code == (0 if payload["ready"] else 1)
        assert payload["total"] + payload["informational"] == len(payload["checks"])

    def test_optional_engine_and_fleet_checks_are_scoped(self, monkeypatch):
        monkeypatch.setattr("macfleet.pool.agent._check_mlx_available", lambda: False)
        base = json.loads(CliRunner().invoke(cli, ["doctor", "--json"]).output)
        mlx = json.loads(CliRunner().invoke(cli, ["doctor", "--json", "--engine", "mlx"]).output)
        assert next(c for c in base["checks"] if c["name"] == "MLX available")["status"] == "info"
        assert next(c for c in mlx["checks"] if c["name"] == "MLX available")["status"] == "fail"
        assert not mlx["ready"]
        assert all(not c["required"] for c in base["checks"] if c["section"] in {"Network", "Security"})
        fleet = json.loads(CliRunner().invoke(cli, ["doctor", "--json", "--fleet"]).output)
        assert all(c["required"] for c in fleet["checks"] if c["section"] in {"Network", "Security"})

    def test_torch_runtime_failure_is_required_only_when_selected(self, monkeypatch):
        monkeypatch.setattr("macfleet.pool.agent._probe_framework", lambda script: False)
        base = json.loads(CliRunner().invoke(cli, ["doctor", "--json"]).output)
        selected_result = CliRunner().invoke(cli, ["doctor", "--json", "--engine", "torch"])
        selected = json.loads(selected_result.output)
        assert next(c for c in base["checks"] if c["name"] == "Torch available")["status"] == "info"
        assert next(c for c in selected["checks"] if c["name"] == "Torch available")["status"] == "fail"
        assert selected_result.exit_code == 1

        monkeypatch.setattr("macfleet.pool.agent._probe_framework", lambda script: script == "import torch")
        cpu = json.loads(CliRunner().invoke(cli, ["doctor", "--json", "--engine", "torch"]).output)
        assert next(c for c in cpu["checks"] if c["name"] == "Torch available")["status"] == "ok"
        assert next(c for c in cpu["checks"] if c["name"] == "MPS available")["status"] == "info"

    def test_doctor_alias_for_diagnose(self):
        """Both commands should produce the same section structure."""
        runner = CliRunner()
        doctor_out = runner.invoke(cli, ["doctor"]).output
        diagnose_out = runner.invoke(cli, ["diagnose"]).output
        # Headers match; detailed content may differ (random hw_id etc)
        for section in ("Runtime", "Hardware", "ML Frameworks", "Thermal", "Network"):
            assert section in doctor_out
            assert section in diagnose_out

    def test_doctor_reports_symlinked_token_file(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "target-token"
        target.write_text("existing-token")
        token_path = tmp_path / "fleet-token"
        token_path.symlink_to(target)
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))
        monkeypatch.setattr("macfleet.security.auth.TOKEN_DIR", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--fleet"])

        assert result.exit_code == 1
        clean = " ".join(result.output.split())
        assert "Token file is regular file" in clean
        assert "must not be a symlink" in clean


class TestCliHelp:
    def test_quickstart_in_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "quickstart" in result.output
        assert "doctor" in result.output
