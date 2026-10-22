"""MLX SDK tests; optional dependency skips must not hide Torch tests."""

import numpy as np
import pytest

from macfleet.sdk.pool import Pool

mlx = pytest.importorskip("mlx.core", reason="MLX not installed")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")


class SimpleMLXModel(mlx_nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = mlx_nn.Linear(4, 8)
        self.linear2 = mlx_nn.Linear(8, 2)

    def __call__(self, x):
        x = mlx_nn.relu(self.linear1(x))
        return self.linear2(x)


def mlx_loss_fn(model, x, y):
    logits = model(x)
    return mlx_nn.losses.cross_entropy(logits, y, reduction="mean")


class TestPoolMLX:
    @pytest.mark.parametrize("counts", [(4, 1), (1, 4)])
    def test_paired_dataset_rejects_mismatched_lengths(self, counts):
        model = SimpleMLXModel()
        dataset = (np.ones((counts[0], 4), dtype=np.float32), np.zeros(counts[1], dtype=np.int32))
        with Pool(engine="mlx", open=True) as pool:
            with pytest.raises(ValueError, match="matching sample counts"):
                pool.train(model, dataset, epochs=1, batch_size=1, loss_fn=mlx_loss_fn)

    def test_pool_train_mlx(self):
        model = SimpleMLXModel()
        mlx.eval(model.parameters())

        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=50).astype(np.int32)

        with Pool(engine="mlx") as pool:
            result = pool.train(
                model=model,
                dataset=(X, y),
                epochs=3,
                batch_size=16,
                loss_fn=mlx_loss_fn,
            )

        assert result["epochs"] == 3
        assert result["loss"] > 0
        assert len(result["loss_history"]) == 3
