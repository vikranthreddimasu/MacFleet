"""Tests for Pool.map(), Pool.submit(), Pool.run() — single-node fast path."""

import time

import pytest

from macfleet import task
from macfleet.sdk.pool import Pool


@task
def _square(x):
    return x * x


@task
def _add_ten(x):
    return x + 10


@task
def _greet(name):
    return f"hello {name}"


@task
def _power(base, exp=2):
    return base ** exp


@task
def _get_value():
    return 42


@task
def _slow_value(delay):
    time.sleep(delay)
    return "done"


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
        with Pool(open=True) as pool:
            result = pool.submit(_power, 3, exp=4)
        assert result == 81

    def test_submit_not_joined_raises(self):
        pool = Pool(open=True)
        with pytest.raises(RuntimeError, match="Must join"):
            pool.submit(_square, 5)

    def test_submit_timeout_is_enforced_for_registered_task(self):
        with Pool(open=True) as pool:
            start = time.monotonic()
            with pytest.raises(TimeoutError, match="timed out after 0.05s"):
                pool.submit(_slow_value, 0.5, timeout=0.05)
            elapsed = time.monotonic() - start

        assert elapsed < 0.25


class TestPoolRun:
    def test_run_basic(self):
        with Pool(open=True) as pool:
            result = pool.run(_square, 6)
        assert result == 36

    def test_run_no_args(self):
        with Pool(open=True) as pool:
            result = pool.run(_get_value)
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
