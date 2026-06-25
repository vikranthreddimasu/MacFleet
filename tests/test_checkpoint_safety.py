"""Tests for checkpoint write and load safety."""

import pytest
import torch
import torch.nn as nn

from macfleet.training import checkpoint as checkpoint_module
from macfleet.training.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint


def _model_and_optimizer():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_checkpoint_manager_writes_checkpoint_and_metadata_atomically(tmp_path):
    model, optimizer = _model_and_optimizer()
    manager = CheckpointManager(str(tmp_path), max_checkpoints=2)

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=10,
        loss=0.25,
        accuracy=0.75,
    )

    assert (tmp_path / "checkpoint_epoch001_step000010.pt").exists()
    assert (tmp_path / "checkpoint_epoch001_step000010.pt.json").exists()
    assert checkpoint_path.endswith("checkpoint_epoch001_step000010.pt")
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_manager_rejects_invalid_retention(tmp_path):
    with pytest.raises(ValueError, match="max_checkpoints"):
        CheckpointManager(str(tmp_path), max_checkpoints=0)


def test_checkpoint_roundtrip_uses_safe_loader(tmp_path):
    model, optimizer = _model_and_optimizer()
    path = tmp_path / "single.pt"

    save_checkpoint(str(path), model, optimizer, epoch=2, step=20)

    loaded_model, loaded_optimizer = _model_and_optimizer()
    checkpoint = load_checkpoint(str(path), loaded_model, loaded_optimizer)

    assert checkpoint["epoch"] == 2
    assert checkpoint["step"] == 20
    assert list(tmp_path.glob("*.tmp")) == []


def test_safe_torch_load_defaults_to_weights_only(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    result = checkpoint_module.safe_torch_load("checkpoint.pt", map_location="cpu")

    assert result == {"ok": True}
    assert calls[0]["weights_only"] is True


def test_safe_torch_load_allows_explicit_trusted_mode(monkeypatch):
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    checkpoint_module.safe_torch_load(
        "checkpoint.pt",
        map_location="cpu",
        trusted=True,
    )

    assert calls[0]["weights_only"] is False
