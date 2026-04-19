"""Tests for the v2.2 PR 16 CLI additions: doctor + quickstart."""

from __future__ import annotations

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
        # Exit code 0 regardless of how many checks pass (doctor is informational)
        assert result.exit_code == 0
        # Output includes the expected section headers
        assert "Hardware" in result.output
        assert "ML Frameworks" in result.output
        assert "Thermal" in result.output
        assert "Network" in result.output

    def test_doctor_alias_for_diagnose(self):
        """Both commands should produce the same section structure."""
        runner = CliRunner()
        doctor_out = runner.invoke(cli, ["doctor"]).output
        diagnose_out = runner.invoke(cli, ["diagnose"]).output
        # Headers match; detailed content may differ (random hw_id etc)
        for section in ("Hardware", "ML Frameworks", "Thermal", "Network"):
            assert section in doctor_out
            assert section in diagnose_out


class TestCliHelp:
    def test_quickstart_in_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "quickstart" in result.output
        assert "doctor" in result.output
