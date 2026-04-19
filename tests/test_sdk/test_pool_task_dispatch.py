"""Tests for Pool.submit/map routing through @macfleet.task (Issue 25).

v2.2 PR 10: `Pool.submit` and `Pool.map` detect @task-decorated callables
and invoke them via the task registry (no cloudpickle). Undecorated fns
fall through to the legacy ProcessPoolExecutor + cloudpickle path so
existing user code keeps working.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from macfleet import task
from macfleet.sdk.pool import Pool

# --------------------------------------------------------------------------- #
# Tasks registered at module level (so ProcessPool workers can re-import them)
# --------------------------------------------------------------------------- #


@task
def _double(n: int) -> int:
    return n * 2


@task
def _square(x: int, offset: int = 0) -> int:
    return x * x + offset


@task
def _boom(x: int) -> int:
    raise ValueError(f"boom on {x}")


class _Cfg(BaseModel):
    n: int
    scale: float


@task(schema=_Cfg)
def _scaled(cfg: _Cfg) -> dict:
    return {"value": cfg.n * cfg.scale}


class TestPoolSubmitRegisteredTask:
    def test_submit_routes_registered_task(self):
        """@task fn goes through the registry, not cloudpickle."""
        with Pool(open=True) as pool:
            result = pool.submit(_double, 21)
        assert result == 42

    def test_submit_with_kwargs(self):
        with Pool(open=True) as pool:
            result = pool.submit(_square, 5, offset=10)
        assert result == 35

    def test_submit_pydantic_schema(self):
        """Pydantic schema validation runs through the task registry path."""
        cfg = _Cfg(n=4, scale=2.5)
        with Pool(open=True) as pool:
            result = pool.submit(_scaled, cfg)
        assert result == {"value": 10.0}

    def test_submit_exception_propagates(self):
        """Exceptions in @task fn surface to the caller."""
        with Pool(open=True) as pool:
            with pytest.raises(ValueError, match="boom on 7"):
                pool.submit(_boom, 7)


class TestPoolMapRegisteredTask:
    def test_map_routes_registered_task(self):
        with Pool(open=True) as pool:
            results = pool.map(_double, [1, 2, 3, 4])
        assert results == [2, 4, 6, 8]

    def test_map_preserves_order(self):
        with Pool(open=True) as pool:
            results = pool.map(_square, [5, 3, 7, 1])
        assert results == [25, 9, 49, 1]

    def test_map_empty_iterable(self):
        with Pool(open=True) as pool:
            assert pool.map(_double, []) == []


class TestPoolLegacyFallback:
    def test_undecorated_fn_still_works(self):
        """Bare lambdas fall through to the old ProcessPool path."""
        with Pool(open=True) as pool:
            # Plain lambda (no task_name) → ProcessPool + cloudpickle
            result = pool.submit(lambda x: x + 100, 5)
        assert result == 105

    def test_undecorated_map_still_works(self):
        with Pool(open=True) as pool:
            results = pool.map(lambda x: x * 3, [1, 2, 3])
        assert results == [3, 6, 9]


class TestIsDistributedProperty:
    def test_single_node_not_distributed(self):
        """Pool with flag off is NOT distributed, even if joined."""
        with Pool(open=True) as pool:
            assert pool.is_distributed is False

    def test_before_join_not_distributed(self):
        """Pool without join is never distributed."""
        pool = Pool(open=True, enable_pool_distributed=True)
        # Before join, no agent, is_distributed is False
        assert pool.is_distributed is False

    def test_distributed_flag_on_with_agent(self):
        """With flag on AND agent joined, is_distributed reports True."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        with Pool(
            name="pool-is-dist",
            open=True,
            port=port,
            data_port=port + 1,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            assert pool.is_distributed is True
