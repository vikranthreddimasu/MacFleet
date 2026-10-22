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


async def test_gradient_clipping_applies_before_optimizer_step():
    from unittest.mock import AsyncMock, Mock

    import numpy as np

    from macfleet.training.loop import training_loop

    engine = Mock()
    engine.forward.return_value = 1.0
    engine.get_flat_gradients.return_value = np.array([3., 4.], dtype=np.float32)
    dp = Mock(world_size=1, setup=AsyncMock(), sync_gradients=AsyncMock(return_value=0.01))
    await training_loop(engine, dp, [1], TrainingConfig(epochs=1, max_grad_norm=1))
    np.testing.assert_allclose(engine.apply_flat_gradients.call_args.args[0], [0.6, 0.8])
    calls = [call[0] for call in engine.mock_calls]
    assert calls.index("apply_flat_gradients") < calls.index("step")
    engine.step.assert_called_once()


def test_unimplemented_checkpoint_setting_is_rejected():
    with pytest.raises(TrainingConfigError, match="not implemented"):
        TrainingConfig(checkpoint_every_n_steps=1)


async def test_distributed_loop_rejects_unsized_loader_before_training():
    from unittest.mock import AsyncMock, Mock

    from macfleet.training.loop import training_loop

    engine = Mock()
    dp = Mock(world_size=2, setup=AsyncMock(), check_consistent=AsyncMock())
    with pytest.raises(TrainingConfigError, match="sized dataloader"):
        await training_loop(engine, dp, iter([1]), TrainingConfig(epochs=1))
    engine.forward.assert_not_called()
