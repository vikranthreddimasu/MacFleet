"""Tests for training and benchmark CLI input validation."""

from __future__ import annotations

import builtins

import click
import pytest
from click.testing import CliRunner

from macfleet.cli.main import _train_demo_mlx, _train_demo_torch, cli


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
        (["train", "--compression", "topk"], "--compression"),
    ]

    for args, option in cases:
        result = runner.invoke(cli, args)
        assert result.exit_code == 2
        assert f"Invalid value for '{option}'" in result.output


def test_train_demo_dispatches_to_selected_engine(monkeypatch):
    calls = []

    def fake_torch(epochs, batch_size, lr):
        calls.append(("torch", epochs, batch_size, lr))

    def fake_mlx(epochs, batch_size, lr):
        calls.append(("mlx", epochs, batch_size, lr))

    monkeypatch.setattr("macfleet.cli.main._train_demo_torch", fake_torch)
    monkeypatch.setattr("macfleet.cli.main._train_demo_mlx", fake_mlx)

    runner = CliRunner()
    torch_result = runner.invoke(cli, ["train", "--epochs", "2"])
    mlx_result = runner.invoke(
        cli,
        [
            "train",
            "--engine",
            "mlx",
            "--epochs",
            "3",
            "--batch-size",
            "16",
            "--lr",
            "0.02",
        ],
    )

    assert torch_result.exit_code == 0
    assert mlx_result.exit_code == 0
    assert calls == [
        ("torch", 2, 128, 0.001),
        ("mlx", 3, 16, 0.02),
    ]


def test_train_demo_torch_missing_dependency_has_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def fail_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_torch_import)

    with pytest.raises(click.ClickException) as exc_info:
        _train_demo_torch(epochs=1, batch_size=8, lr=0.001)

    assert "macfleet[torch]" in exc_info.value.message
    assert "Traceback" not in exc_info.value.message


def test_train_demo_mlx_missing_dependency_has_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def fail_mlx_import(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx."):
            raise ImportError("missing mlx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_mlx_import)

    with pytest.raises(click.ClickException) as exc_info:
        _train_demo_mlx(epochs=1, batch_size=8, lr=0.001)

    assert "macfleet[mlx]" in exc_info.value.message
    assert "Traceback" not in exc_info.value.message


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
