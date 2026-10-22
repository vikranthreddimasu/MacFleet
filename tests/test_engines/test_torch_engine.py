"""Tests for the PyTorch training engine."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from macfleet.engines.base import EngineType
from macfleet.engines.torch_engine import TorchEngine


class _TinyModel(nn.Module):
    """Minimal model for testing (10 + 5 = 15 params)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 5, bias=False)  # 10 params

    def forward(self, x):
        return self.linear(x).sum()


class _TwoLayerModel(nn.Module):
    """Two-layer model for gradient bucketing tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8, bias=False)  # 32 params
        self.layer2 = nn.Linear(8, 2, bias=False)  # 16 params

    def forward(self, x):
        return self.layer2(self.layer1(x)).sum()


class TestTorchEngineBasic:
    def test_init_cpu(self):
        engine = TorchEngine(device="cpu")
        assert engine.device == torch.device("cpu")

    def test_load_model(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        engine.load_model(model, optimizer)
        assert engine.param_count() == 10

    def test_forward_backward(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        engine.load_model(model, optimizer)

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)

        # Gradients should be populated
        assert model.linear.weight.grad is not None

    def test_step(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        engine.load_model(model, optimizer)

        params_before = model.linear.weight.data.clone()

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)
        engine.step()

        # Parameters should have changed
        assert not torch.equal(model.linear.weight.data, params_before)

    def test_zero_grad(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)
        assert model.linear.weight.grad is not None

        engine.zero_grad()
        assert model.linear.weight.grad is None or model.linear.weight.grad.abs().sum() == 0


class TestTorchEngineGradients:
    def test_get_flat_gradients(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)

        flat = engine.get_flat_gradients()
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.float32
        assert len(flat) == 10  # 5x2 weight matrix

    def test_flat_gradients_roundtrip(self):
        """get_flat_gradients → apply_flat_gradients preserves values."""
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)

        # Extract, modify (simulate allreduce), and apply
        flat = engine.get_flat_gradients()
        modified = flat * 2.0  # simulate averaged gradients
        engine.apply_flat_gradients(modified)

        # Verify the gradients match the modified values
        result = engine.get_flat_gradients()
        np.testing.assert_allclose(result, modified, rtol=1e-6)

    def test_flat_gradients_two_layers(self):
        """Flat gradients include all layers in parameter order."""
        engine = TorchEngine(device="cpu")
        model = _TwoLayerModel()
        engine.load_model(model)

        x = torch.randn(3, 4)
        loss = engine.forward(x)
        engine.backward(loss)

        flat = engine.get_flat_gradients()
        assert len(flat) == 32 + 16  # layer1 (4x8) + layer2 (8x2)

    def test_no_model_returns_empty(self):
        engine = TorchEngine(device="cpu")
        assert engine.param_count() == 0
        assert engine.memory_usage_gb() == 0.0


class TestTorchEngineParameters:
    def test_get_flat_parameters(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        flat = engine.get_flat_parameters()
        assert isinstance(flat, np.ndarray)
        assert len(flat) == 10

    def test_flat_parameters_roundtrip(self):
        """Broadcast simulation: get → apply preserves parameters."""
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)

        original = engine.get_flat_parameters().copy()

        # Simulate receiving broadcast parameters
        engine.apply_flat_parameters(original)

        result = engine.get_flat_parameters()
        np.testing.assert_allclose(result, original, rtol=1e-6)


class TestTorchEngineState:
    def test_state_dict_roundtrip(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        engine.load_model(model, optimizer)

        # Modify model
        x = torch.randn(3, 2)
        loss = engine.forward(x)
        engine.backward(loss)
        engine.step()

        # Checkpoint
        params_after_step = engine.get_flat_parameters().copy()
        state_bytes = engine.state_dict()

        # Reset model
        restored_model = _TinyModel()
        engine.load_model(restored_model, torch.optim.SGD(restored_model.parameters(), lr=0.01))

        # Restore
        engine.load_state_dict(state_bytes)
        restored_params = engine.get_flat_parameters()
        np.testing.assert_allclose(restored_params, params_after_step, rtol=1e-6)

    def test_capabilities(self):
        engine = TorchEngine(device="cpu")
        caps = engine.capabilities
        assert caps.engine_type == EngineType.TORCH
        assert "float32" in caps.supported_dtypes

    def test_memory_usage(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)
        assert engine.memory_usage_gb() > 0

    def test_estimated_memory(self):
        engine = TorchEngine(device="cpu")
        model = _TinyModel()
        engine.load_model(model)
        assert engine.estimated_model_memory_gb() > 0


def test_parameter_updates_validate_before_mutation():
    model = nn.Linear(2, 2)
    engine = TorchEngine(device="cpu")
    engine.load_model(model)
    original = engine.get_flat_parameters()
    for invalid in [np.ones(original.size + 1, dtype=np.float32), np.full_like(original, np.nan)]:
        with pytest.raises(ValueError):
            engine.apply_flat_parameters(invalid)
        np.testing.assert_array_equal(engine.get_flat_parameters(), original)


def test_frozen_parameters_are_included_in_initial_broadcast():
    model = nn.Linear(2, 2)
    model.weight.requires_grad_(False)
    engine = TorchEngine(device="cpu")
    engine.load_model(model)
    assert engine.get_flat_parameters().size == 6
    engine.apply_flat_parameters(np.ones(6, dtype=np.float32))
    assert torch.equal(model.weight, torch.ones_like(model.weight))


def test_half_precision_gradient_update_preserves_parameter_dtype():
    model = nn.Linear(2, 1).half()
    engine = TorchEngine(device="cpu")
    engine.load_model(model)
    engine.apply_flat_gradients(np.ones(3, dtype=np.float32))
    assert model.weight.grad.dtype == torch.float16


def test_optimizer_order_mismatch_rejects_checkpoint_before_mutation():
    model = torch.nn.Linear(1, 1)
    engine = TorchEngine(device="cpu")
    engine.load_model(model, torch.optim.SGD([model.weight, model.bias], lr=.1, momentum=.9))
    checkpoint = engine.state_dict()
    other = torch.nn.Linear(1, 1)
    restored = TorchEngine(device="cpu")
    restored.load_model(other, torch.optim.SGD([other.bias, other.weight], lr=.1, momentum=.9))
    before = restored.get_flat_parameters().copy()
    assert restored.model_signature() != engine.model_signature()
    with pytest.raises(ValueError, match="optimizer parameter order"):
        restored.load_state_dict(checkpoint)
    np.testing.assert_array_equal(restored.get_flat_parameters(), before)


def test_optimizer_for_another_model_is_rejected():
    model = torch.nn.Linear(1, 1)
    other = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="outside the loaded model"):
        TorchEngine(device="cpu").load_model(model, torch.optim.SGD(other.parameters(), lr=.1))


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("optimizer_type", [torch.optim.Adam, torch.optim.SGD])
def test_loading_initialized_optimizer_preserves_training_on_target_device(device, optimizer_type):
    import copy

    from macfleet.engines.torch_engine import _mps_actually_works

    if device == "mps" and (not torch.backends.mps.is_available() or not _mps_actually_works()):
        pytest.skip("MPS device is unavailable")
    torch.manual_seed(52)
    model = nn.Linear(2, 1)
    options = {"momentum": 0.9} if optimizer_type is torch.optim.SGD else {}
    optimizer = optimizer_type([
        {"params": [model.weight], "lr": 0.02},
        {"params": [model.bias], "lr": 0.01},
    ], **options)
    batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model(batch).square().mean().backward()
    optimizer.step()
    reference, reference_optimizer = copy.deepcopy((model, optimizer))

    engine = TorchEngine(device=device)
    engine.load_model(model, optimizer)
    assert [group["lr"] for group in optimizer.param_groups] == [0.02, 0.01]
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and name != "step":
                assert value.device == parameter.device
    engine.zero_grad()
    engine.backward(engine.forward(batch.to(device)).square().mean())
    engine.step()
    reference_optimizer.zero_grad()
    reference(batch).square().mean().backward()
    reference_optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("invalid", ["nan", "inf"])
def test_nonfinite_gradients_leave_model_and_optimizer_unchanged(invalid):
    import copy

    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    engine = TorchEngine(device="cpu")
    engine.load_model(model, optimizer)
    batch = torch.ones(1, 1)
    engine.backward(engine.forward(batch).square().mean())
    engine.step()
    with torch.no_grad():
        model.weight.zero_()
    engine.zero_grad()
    weight = model.weight[0, 0]
    loss = (weight.square().sqrt() if invalid == "nan" else weight.sqrt()) + model.bias.sum()
    engine.backward(loss)
    before = engine.get_flat_parameters().copy()
    state_before = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(ValueError, match="Non-finite gradients"):
        engine.step()
    np.testing.assert_array_equal(engine.get_flat_parameters(), before)
    state_after = optimizer.state_dict()
    assert state_after["param_groups"] == state_before["param_groups"]
    for parameter, state in state_before["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(state_after["state"][parameter][name], value, rtol=0, atol=0)

    engine.zero_grad()
    engine.backward(engine.forward(batch).square().mean())
    engine.step()
    assert np.isfinite(engine.get_flat_parameters()).all()
    assert all(state["step"].item() == 2 for state in optimizer.state.values())


def test_finite_sparse_gradients_still_train():
    model = nn.Embedding(4, 2, sparse=True)
    nn.init.zeros_(model.weight)
    engine = TorchEngine(device="cpu")
    engine.load_model(model, torch.optim.SGD(model.parameters(), lr=0.1))
    engine.backward(engine.forward(torch.tensor([1, 1, 2])).sum())
    engine.step()
    torch.testing.assert_close(model.weight, torch.tensor([[0., 0.], [-.2, -.2], [-.1, -.1], [0., 0.]]))
