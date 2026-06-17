"""Tests for Pool.map(), Pool.submit(), Pool.run() — single-node fast path."""

import pytest

from macfleet.sdk.pool import Pool


def _square(x):
    return x * x


def _add_ten(x):
    return x + 10


def _greet(name):
    return f"hello {name}"


def _failing_fn(x):
    raise ValueError(f"cannot process {x}")


class TestPoolMap:
    def test_map_basic(self):
        with Pool(open=True) as pool:
            results = pool.map(_square, [1, 2, 3, 4, 5])
        assert results == [1, 4, 9, 16, 25]

    def test_map_empty(self):
        with Pool(open=True) as pool:
            results = pool.map(_square, [])
        assert results == []

    def test_map_single_item(self):
        with Pool(open=True) as pool:
            results = pool.map(_add_ten, [5])
        assert results == [15]

    def test_map_strings(self):
        with Pool(open=True) as pool:
            results = pool.map(_greet, ["alice", "bob"])
        assert results == ["hello alice", "hello bob"]

    def test_map_preserves_order(self):
        with Pool(open=True) as pool:
            results = pool.map(_square, range(10))
        assert results == [i * i for i in range(10)]

    def test_map_not_joined_raises(self):
        pool = Pool(open=True)
        with pytest.raises(RuntimeError, match="Must join"):
            pool.map(_square, [1])


class TestPoolSubmit:
    def test_submit_basic(self):
        with Pool(open=True) as pool:
            result = pool.submit(_square, 7)
        assert result == 49

    def test_submit_with_kwargs(self):
        def power(base, exp=2):
            return base ** exp

        with Pool(open=True) as pool:
            result = pool.submit(power, 3, exp=4)
        assert result == 81

    def test_submit_not_joined_raises(self):
        pool = Pool(open=True)
        with pytest.raises(RuntimeError, match="Must join"):
            pool.submit(_square, 5)


class TestPoolRun:
    def test_run_basic(self):
        with Pool(open=True) as pool:
            result = pool.run(_square, 6)
        assert result == 36

    def test_run_no_args(self):
        def get_value():
            return 42

        with Pool(open=True) as pool:
            result = pool.run(get_value)
        assert result == 42


class TestPoolTrainRegression:
    """Verify that Pool.train() still works after adding compute methods."""

    def test_train_still_works(self):
        """ML training path is unaffected by compute additions."""
        torch = pytest.importorskip("torch")
        nn = pytest.importorskip("torch.nn")

        model = nn.Sequential(nn.Linear(4, 2))
        X = torch.randn(50, 4)
        y = (X[:, 0] > 0).long()

        with Pool(open=True) as pool:
            result = pool.train(
                model=model,
                dataset=(X, y),
                epochs=2,
                batch_size=25,
                lr=0.01,
                loss_fn=nn.CrossEntropyLoss(),
            )

        assert "loss" in result
        assert result["epochs"] == 2
        assert result["time_sec"] > 0
