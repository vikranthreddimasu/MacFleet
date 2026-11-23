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


@pytest.mark.parametrize("optimizer_kind", ["SGD", "AdamW"])
async def test_weighted_gradient_and_unused_parameters_match_single_mac(optimizer_kind):
    import asyncio

    torch = pytest.importorskip("torch")
    from macfleet.engines.torch_engine import TorchEngine
    from macfleet.training.data_parallel import DataParallel
    from macfleet.training.loop import training_loop
    from macfleet.training.sampler import DistributedBatchSampler
    from tests.test_comm.test_collectives import _make_groups, _setup_mesh, _teardown_mesh

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.))
            self.branch = torch.nn.Parameter(torch.tensor(1.))
            self.unused = torch.nn.Parameter(torch.tensor(2.))

        def forward(self, x):
            y = self.scale * x
            if (x > 2).any():
                y = y + self.branch * (x > 2)
            return y.mean()

    def build():
        model = Model()
        optimizer = getattr(torch.optim, optimizer_kind)(model.parameters(), lr=1.0)
        return model, optimizer

    data = torch.tensor([1.] * 7 + [3.] * 3)
    reference, optimizer = build()
    reference(data).backward()
    clip = 0.5 if optimizer_kind == "AdamW" else 0.0
    if clip:
        torch.nn.utils.clip_grad_norm_(reference.parameters(), clip)
    optimizer.step()
    if optimizer_kind == "SGD":
        assert reference.scale.item() == pytest.approx(-1.6)
    transports, _ = await _setup_mesh(2)
    try:
        ranks = [build(), build()]
        runs = []
        for rank, ((model, opt), group) in enumerate(zip(ranks, _make_groups(transports))):
            engine = TorchEngine(device="cpu")
            engine.load_model(model, opt)
            sampler = DistributedBatchSampler(data, 10, 2, rank, weights=[0.7, 0.3], shuffle=False)
            loader = torch.utils.data.DataLoader(data, batch_sampler=sampler)
            cfg = TrainingConfig(epochs=1, max_grad_norm=clip)
            runs.append(training_loop(engine, DataParallel(engine, group), loader, cfg))
        await asyncio.gather(*runs)
        for model, opt in ranks:
            for p, ref in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p, ref)
                if p in opt.state:
                    for name, value in opt.state[p].items():
                        torch.testing.assert_close(value, optimizer.state[ref][name])
            assert model.unused.grad is None and model.unused not in opt.state
    finally:
        await _teardown_mesh(transports)


@pytest.mark.parametrize("optimizer_name", ["SGD", "AdamW"])
async def test_clipped_conditional_gradients_match_single_mac(optimizer_name):
    import asyncio

    torch = pytest.importorskip("torch")
    from macfleet.engines.torch_engine import TorchEngine
    from macfleet.training.data_parallel import DataParallel
    from macfleet.training.loop import training_loop
    from tests.test_comm.test_collectives import _make_groups, _setup_mesh, _teardown_mesh

    class Conditional(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Parameter(torch.ones(()))
            self.branch = torch.nn.Parameter(torch.ones(()))
            self.unused = torch.nn.Parameter(torch.ones(()))

        def forward(self, x):
            value = self.used * x
            if (x > 0).any():
                value = value + self.branch * (x > 0)
            return value.mean()

    def build():
        model = Conditional()
        opt = getattr(torch.optim, optimizer_name)(model.parameters(), lr=0.1, weight_decay=0.1)
        return model, opt

    shards = [torch.tensor([0., 0.]), torch.tensor([2., 2.])]
    expected, optimizer = build()
    expected(torch.cat(shards)).backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 0.5)
    optimizer.step()

    transports, _ = await _setup_mesh(2)
    try:
        ranks = [build(), build()]
        runs = []
        for group, (model, opt), shard in zip(_make_groups(transports), ranks, shards):
            engine = TorchEngine(device="cpu")
            engine.load_model(model, opt)
            runs.append(training_loop(engine, DataParallel(engine, group), [shard],
                                      TrainingConfig(epochs=1, max_grad_norm=0.5)))
        await asyncio.gather(*runs)
        for model, opt in ranks:
            for param, ref in zip(model.parameters(), expected.parameters()):
                torch.testing.assert_close(param, ref)
            assert model.unused.grad is None and model.unused not in opt.state
            torch.testing.assert_close(model.branch.grad, expected.branch.grad)
    finally:
        await _teardown_mesh(transports)
