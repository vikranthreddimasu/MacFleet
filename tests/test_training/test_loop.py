"""Tests for training loop configuration validation."""

from __future__ import annotations

import pytest

from macfleet.training.guards import TrainingConfigError
from macfleet.training.loop import TrainingConfig


class TestTrainingConfig:
    def test_defaults_are_valid(self):
        config = TrainingConfig()

        assert config.epochs == 10
        assert config.log_every_n_steps == 10
        assert config.checkpoint_every_n_steps == 0
        assert config.max_grad_norm == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"epochs": 0}, "epochs"),
            ({"epochs": True}, "epochs"),
            ({"log_every_n_steps": -1}, "log_every_n_steps"),
            ({"checkpoint_every_n_steps": -1}, "checkpoint_every_n_steps"),
            ({"max_grad_norm": -0.1}, "max_grad_norm"),
            ({"max_grad_norm": float("nan")}, "max_grad_norm"),
        ],
    )
    def test_invalid_values_fail_fast(self, kwargs, field):
        with pytest.raises(TrainingConfigError, match=field):
            TrainingConfig(**kwargs)
