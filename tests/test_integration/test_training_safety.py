"""Public training must reject invalid derivatives before corrupting a model."""

import numpy as np
import pytest

from macfleet.sdk.pool import Pool


@pytest.mark.parametrize("engine", ["torch", "mlx"])
def test_finite_loss_with_nonfinite_gradients_preserves_model(engine):
    x = np.ones((4, 2), dtype=np.float32)
    y = np.ones((4, 1), dtype=np.float32)
    if engine == "torch":
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        def loss_fn(outputs, _):
            return outputs.square().sum().sqrt()

        def valid_loss(outputs, targets):
            return (outputs - targets).square().mean()

        def weights():
            return model.weight.detach().cpu().numpy().copy()
    else:
        mx = pytest.importorskip("mlx.core")
        nn = pytest.importorskip("mlx.nn")
        optim = pytest.importorskip("mlx.optimizers")
        model = nn.Linear(2, 1, bias=False)
        model.weight = mx.zeros_like(model.weight)
        optimizer = optim.Adam(learning_rate=0.1)

        def loss_fn(model, inputs, _):
            return mx.sqrt(mx.sum(mx.square(model(inputs))))

        def valid_loss(model, inputs, targets):
            return mx.mean(mx.square(model(inputs) - targets))

        def weights():
            return np.array(model.weight)

    before = weights()
    with Pool(engine=engine, open=True) as pool:
        with pytest.raises(ValueError, match="gradients"):
            pool.train(
                model, (x, y), optimizer=optimizer, loss_fn=loss_fn,
                epochs=1, batch_size=4, device="cpu",
            )
        np.testing.assert_array_equal(weights(), before)
        result = pool.train(
            model, (x, y), optimizer=optimizer, loss_fn=valid_loss,
            epochs=1, batch_size=2, device="cpu",
        )
    assert result["steps"] == 2
    assert np.isfinite(result["loss"])
    assert np.isfinite(weights()).all()
    assert (weights() > before).all()
