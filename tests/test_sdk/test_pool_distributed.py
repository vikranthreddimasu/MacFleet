"""Tests for v2.2 PR 8 (Issue 1a): Pool.join with distributed flag.

Covers:
    - Default (flag off): Pool.join is a no-op, world_size=1, nodes=[]
    - Flag on: Pool.join spins up a real PoolAgent, registry is live
    - Pool.world_size reads from the live registry (not hardcoded 1)
    - Pool.nodes dumps alive_nodes as dicts
    - Pool.leave stops the agent + background event loop cleanly
    - Timeout: if quorum isn't met within quorum_timeout_sec, raises
      TimeoutError with remediation text
"""

from __future__ import annotations

import socket

import pytest


def _free_port() -> int:
    """Grab an ephemeral port for this test run."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestPoolFeatureFlagDefault:
    """Default behavior (flag off) must remain legacy single-node."""

    def test_join_is_noop_by_default(self):
        """Without the flag, Pool.join doesn't spin up an agent."""
        from macfleet.sdk.pool import Pool

        pool = Pool(open=True)  # no token
        pool.join()
        try:
            assert pool._agent is None
            assert pool.world_size == 1
            assert pool.nodes == []
        finally:
            pool.leave()

    def test_context_manager_single_node(self):
        from macfleet.sdk.pool import Pool

        with Pool(open=True) as pool:
            assert pool.world_size == 1
            assert pool.nodes == []

    def test_enter_exit_idempotent(self):
        """Calling leave() without join() is safe."""
        from macfleet.sdk.pool import Pool

        pool = Pool(open=True)
        pool.leave()  # no-op
        pool.leave()  # still a no-op


class TestPoolDistributedFlag:
    """With `enable_pool_distributed=True`, Pool owns a PoolAgent."""

    def test_join_starts_agent(self):
        from macfleet.sdk.pool import Pool

        port = _free_port()
        pool = Pool(
            name="pool-test-1",
            open=True,
            port=port,
            data_port=port + 1,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        )
        try:
            pool.join()
            assert pool._agent is not None
            # self is the only node → world_size == 1
            assert pool.world_size == 1
            nodes = pool.nodes
            assert len(nodes) == 1
            self_node = nodes[0]
            assert self_node["port"] == port
            assert self_node["data_port"] == port + 1
            assert self_node["is_coordinator"] is True
        finally:
            pool.leave()

    def test_context_manager_distributed(self):
        from macfleet.sdk.pool import Pool

        port = _free_port()
        with Pool(
            name="pool-test-2",
            open=True,
            port=port,
            data_port=port + 1,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            assert pool._agent is not None
            assert pool.world_size >= 1

    def test_quorum_timeout_raises(self):
        """quorum_size above observed → TimeoutError with remediation text."""
        from macfleet.sdk.pool import Pool

        port = _free_port()
        pool = Pool(
            name="pool-test-3",
            open=True,
            port=port,
            data_port=port + 1,
            enable_pool_distributed=True,
            quorum_size=2,  # need 2, will only see 1
            quorum_timeout_sec=1.0,  # fail fast
        )
        with pytest.raises(TimeoutError) as excinfo:
            pool.join()
        # Remediation text should mention `macfleet status` and manual peers
        assert "macfleet status" in str(excinfo.value)
        assert "peers=" in str(excinfo.value)
        # Agent should be cleaned up on failure
        assert pool._agent is None

    def test_leave_stops_agent(self):
        from macfleet.sdk.pool import Pool

        port = _free_port()
        pool = Pool(
            name="pool-test-4",
            open=True,
            port=port,
            data_port=port + 1,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        )
        pool.join()
        assert pool._agent is not None
        pool.leave()
        assert pool._agent is None
        assert pool._loop is None
        assert pool._loop_thread is None

    def test_nodes_exposes_hardware(self):
        """Pool.nodes dumps the local hardware profile fields the user cares about."""
        from macfleet.sdk.pool import Pool

        port = _free_port()
        with Pool(
            name="pool-test-5",
            open=True,
            port=port,
            data_port=port + 1,
            enable_pool_distributed=True,
            quorum_size=1,
            quorum_timeout_sec=5.0,
        ) as pool:
            nodes = pool.nodes
            assert len(nodes) == 1
            n = nodes[0]
            # Required fields for downstream consumers (TUI, Pool.train)
            for field in [
                "node_id", "hostname", "ip_address", "port", "data_port",
                "chip_name", "gpu_cores", "ram_gb", "compute_score",
                "is_coordinator",
            ]:
                assert field in n, f"Pool.nodes missing {field!r}: {n}"
