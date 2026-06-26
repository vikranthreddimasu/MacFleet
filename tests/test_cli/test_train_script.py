"""Tests for `macfleet train SCRIPT` option forwarding."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from macfleet.cli.main import cli


def test_train_script_passes_cli_options_to_matching_main_parameters(tmp_path: Path):
    script = tmp_path / "train_job.py"
    config = tmp_path / "train.json"
    output = tmp_path / "seen.json"
    config.write_text("{}")
    script.write_text(
        f"""
import json

def main(engine, epochs, batch_size, lr, compression, config_path):
    with open({str(output)!r}, "w") as f:
        json.dump(
            {{
                "engine": engine,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "compression": compression,
                "config_path": config_path,
            }},
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
            str(config),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text()) == {
        "engine": "mlx",
        "epochs": 3,
        "batch_size": 16,
        "lr": 0.02,
        "compression": "adaptive",
        "config_path": str(config),
    }


def test_train_config_populates_script_options(tmp_path: Path):
    script = tmp_path / "train_job.py"
    config = tmp_path / "train.json"
    output = tmp_path / "seen.json"
    config.write_text(
        json.dumps(
            {
                "engine": "mlx",
                "epochs": 4,
                "batch-size": 24,
                "learning_rate": 0.03,
                "compression": "light",
            }
        )
    )
    script.write_text(
        f"""
import json

def main(engine, epochs, batch_size, lr, compression):
    with open({str(output)!r}, "w") as f:
        json.dump(
            {{
                "engine": engine,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "compression": compression,
            }},
            f,
            sort_keys=True,
        )
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(script), "--config", str(config)])

    assert result.exit_code == 0
    assert json.loads(output.read_text()) == {
        "engine": "mlx",
        "epochs": 4,
        "batch_size": 24,
        "lr": 0.03,
        "compression": "light",
    }


def test_train_cli_options_override_config_values(tmp_path: Path):
    script = tmp_path / "train_job.py"
    config = tmp_path / "train.json"
    output = tmp_path / "seen.json"
    config.write_text(json.dumps({"epochs": 99, "batch_size": 128, "lr": 0.5}))
    script.write_text(
        f"""
import json

def main(epochs, batch_size, lr):
    with open({str(output)!r}, "w") as f:
        json.dump({{"epochs": epochs, "batch_size": batch_size, "lr": lr}}, f)
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train",
            str(script),
            "--config",
            str(config),
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--lr",
            "0.01",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text()) == {
        "epochs": 2,
        "batch_size": 16,
        "lr": 0.01,
    }


def test_train_config_rejects_unknown_keys_before_script_runs(tmp_path: Path):
    script = tmp_path / "train_job.py"
    config = tmp_path / "train.json"
    marker = tmp_path / "should-not-exist"
    config.write_text(json.dumps({"epocs": 3}))
    script.write_text(
        f"""
def main():
    with open({str(marker)!r}, "w") as f:
        f.write("ran")
""".lstrip()
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["train", str(script), "--config", str(config)])

    assert result.exit_code == 1
    assert "Unknown training config key" in result.output
    assert "epocs" in result.output
    assert not marker.exists()


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
    config_file.write_text("{}")
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
