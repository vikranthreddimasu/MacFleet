"""Tests for `macfleet train` CLI option wiring."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from macfleet.cli.main import cli


class TestTrainScriptInvocation:
    def test_passes_supported_kwargs_to_script_main(self, tmp_path: Path):
        output_path = tmp_path / "out.json"
        script_path = tmp_path / "train_script.py"
        script_path.write_text(
            """
import json
import os

def main(engine, epochs, batch_size, lr, compression):
    with open(os.environ["OUT"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "engine": engine,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "compression": compression,
            },
            f,
        )
"""
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "train",
                str(script_path),
                "--engine",
                "mlx",
                "--epochs",
                "7",
                "--batch-size",
                "16",
                "--lr",
                "0.02",
                "--compression",
                "adaptive",
            ],
            env={"OUT": str(output_path)},
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(output_path.read_text())
        assert payload == {
            "engine": "mlx",
            "epochs": 7,
            "batch_size": 16,
            "lr": 0.02,
            "compression": "adaptive",
        }

    def test_zero_arg_main_still_supported(self, tmp_path: Path):
        output_path = tmp_path / "ok.txt"
        script_path = tmp_path / "zero_arg_script.py"
        script_path.write_text(
            """
import os

def main():
    with open(os.environ["OUT"], "w", encoding="utf-8") as f:
        f.write("ok")
"""
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "train",
                str(script_path),
                "--epochs",
                "3",
                "--compression",
                "none",
            ],
            env={"OUT": str(output_path)},
        )
        assert result.exit_code == 0, result.output
        assert output_path.read_text() == "ok"

    def test_legacy_compression_alias_is_mapped(self, tmp_path: Path):
        output_path = tmp_path / "compression.txt"
        script_path = tmp_path / "compression_script.py"
        script_path.write_text(
            """
import os

def main(compression):
    with open(os.environ["OUT"], "w", encoding="utf-8") as f:
        f.write(compression)
"""
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["train", str(script_path), "--compression", "topk_fp16"],
            env={"OUT": str(output_path)},
        )
        assert result.exit_code == 0, result.output
        assert output_path.read_text() == "moderate"
