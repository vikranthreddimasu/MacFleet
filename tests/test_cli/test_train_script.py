"""Tests for `macfleet train SCRIPT` option forwarding."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from macfleet.cli.main import cli


def test_train_script_passes_cli_options_to_matching_main_parameters(tmp_path: Path):
    script = tmp_path / "train_job.py"
    output = tmp_path / "seen.json"
    script.write_text(
        """
import json

def main(engine, epochs, batch_size, lr, compression, config_path):
    with open(config_path, "w") as f:
        json.dump(
            {
                "engine": engine,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "compression": compression,
            },
            f,
            sort_keys=True,
        )
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train",
            str(script),
            "--engine",
            "mlx",
            "--epochs",
            "3",
            "--batch-size",
            "16",
            "--lr",
            "0.02",
            "--compression",
            "adaptive",
            "--config",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text()) == {
        "engine": "mlx",
        "epochs": 3,
        "batch_size": 16,
        "lr": 0.02,
        "compression": "adaptive",
    }


def test_train_script_preserves_no_arg_main_compatibility(tmp_path: Path):
    script = tmp_path / "train_job.py"
    output = tmp_path / "ran.txt"
    script.write_text(
        f"""
def main():
    with open({str(output)!r}, "w") as f:
        f.write("ran")
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(script), "--epochs", "2"])

    assert result.exit_code == 0
    assert output.read_text() == "ran"


def test_train_script_can_receive_config_alias(tmp_path: Path):
    script = tmp_path / "train_job.py"
    output = tmp_path / "config-path.txt"
    config_file = tmp_path / "settings.yaml"
    script.write_text(
        f"""
def main(config):
    with open({str(output)!r}, "w") as f:
        f.write(config)
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["train", str(script), "--config", str(config_file)],
    )

    assert result.exit_code == 0
    assert output.read_text() == str(config_file)


def test_train_script_reports_required_unsupported_parameters(tmp_path: Path):
    script = tmp_path / "train_job.py"
    script.write_text(
        """
def main(dataset):
    raise AssertionError("should not run")
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(script)])

    assert result.exit_code == 1
    assert "cannot provide" in result.output
    assert "dataset" in result.output
    assert "Traceback" not in result.output


def test_train_script_reports_non_callable_main(tmp_path: Path):
    script = tmp_path / "train_job.py"
    script.write_text("main = 123\n")

    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(script)])

    assert result.exit_code == 1
    assert "main" in result.output
    assert "not callable" in result.output
    assert "Traceback" not in result.output
