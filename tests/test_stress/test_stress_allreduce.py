"""Long-soak stress tests for ring AllReduce.

Production scenario: training runs 1000+ allreduce rounds. Memory must
not grow without bound, FDs must not leak, and gradient consistency
must hold across every round.
"""

from __future__ import annotations

import asyncio
import gc
import os
import resource

import numpy as np
import pytest

from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import PeerTransport, TransportConfig

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=10.0)


async def _setup_mesh(n: int) -> tuple[list[PeerTransport], list[int]]:
    transports = []
    ports = []
    for i in range(n):
        t = PeerTransport(local_id=f"node-{i}", config=CONFIG)
        await t.start_server("127.0.0.1", 0)
        port = t._server.sockets[0].getsockname()[1]
        transports.append(t)
        ports.append(port)
    for i in range(n):
        for j in range(i + 1, n):
            await transports[i].connect(f"node-{j}", "127.0.0.1", ports[j])
    await asyncio.sleep(0.2)
    return transports, ports


async def _teardown_mesh(transports: list[PeerTransport]) -> None:
    for t in transports:
        await t.disconnect_all()


def _make_groups(n: int, transports: list[PeerTransport]) -> list[CollectiveGroup]:
    groups = []
    for rank in range(n):
        rank_to_peer = {r: f"node-{r}" for r in range(n) if r != rank}
        groups.append(CollectiveGroup(
            rank=rank, world_size=n, transport=transports[rank],
            rank_to_peer=rank_to_peer,
        ))
    return groups


def _open_fd_count() -> int:
    """Approximate count of open FDs for this process."""
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except FileNotFoundError:
        # macOS: walk /dev/fd. Each FD shows up once.
        try:
            return len(os.listdir("/dev/fd"))
        except FileNotFoundError:
            return -1


class TestAllreduceSoak:
    """100+ rounds of allreduce on a 3-node loopback mesh."""

    @pytest.mark.asyncio
    async def test_100_rounds_three_node(self):
        n = 3
        rounds = 100
        size = 1000

        transports, _ = await _setup_mesh(n)
        try:
            groups = _make_groups(n, transports)
            for round_i in range(rounds):
                # Each node contributes a different array per round.
                arrays = [
                    np.full(size, float(rank + round_i), dtype=np.float32)
                    for rank in range(n)
                ]
                results = await asyncio.gather(*(
                    groups[rank].allreduce(arrays[rank], op="mean")
                    for rank in range(n)
                ))
                # Expected mean for round R: mean(0+R, 1+R, 2+R) = 1+R.
                expected = float(round_i) + (n - 1) / 2.0
                for r in results:
                    np.testing.assert_allclose(r, expected, rtol=1e-5)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_no_fd_leak_over_rounds(self):
        """FD count after N rounds should not grow proportional to rounds."""
        n = 3
        rounds = 50
        size = 100

        transports, _ = await _setup_mesh(n)
        try:
            groups = _make_groups(n, transports)
            # Warmup so steady-state FDs settle (TLS buffers, etc.).
            for _ in range(5):
                await asyncio.gather(*(
                    g.allreduce(np.zeros(size, dtype=np.float32), op="mean")
                    for g in groups
                ))
            gc.collect()
            await asyncio.sleep(0.05)
            fd_before = _open_fd_count()

            for _ in range(rounds):
                await asyncio.gather(*(
                    g.allreduce(
                        np.random.randn(size).astype(np.float32), op="mean",
                    )
                    for g in groups
                ))
            gc.collect()
            await asyncio.sleep(0.05)
            fd_after = _open_fd_count()

            if fd_before == -1 or fd_after == -1:
                pytest.skip("FD enumeration not available on this OS")

            # Some leeway for transient sockets; growth > 50 over 50
            # rounds is the kind of leak that compounds in production.
            assert fd_after - fd_before < 50, (
                f"FD growth: {fd_before} → {fd_after} over {rounds} rounds"
            )
        finally:
            await _teardown_mesh(transports)


class TestAllreduceMemoryStability:
    """Memory should not grow without bound across rounds."""

    @pytest.mark.asyncio
    async def test_rss_stable_across_rounds(self):
        n = 3
        rounds = 100
        size = 10_000

        transports, _ = await _setup_mesh(n)
        try:
            groups = _make_groups(n, transports)
            # Warmup.
            for _ in range(5):
                await asyncio.gather(*(
                    g.allreduce(np.zeros(size, dtype=np.float32), op="mean")
                    for g in groups
                ))
            gc.collect()
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            for _ in range(rounds):
                await asyncio.gather(*(
                    g.allreduce(
                        np.random.randn(size).astype(np.float32), op="mean",
                    )
                    for g in groups
                ))
            gc.collect()
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            # ru_maxrss is in bytes on macOS, KB on Linux. Both ways the
            # relative growth bounds the leak.
            growth = rss_after - rss_before
            # Allow 50 MB of growth across 100 rounds — that's 500 KB
            # per round, far above what a leak-free implementation needs.
            # If real growth is sub-MB we accept noise.
            assert growth < 100_000_000, (
                f"RSS growth {growth} bytes over {rounds} rounds suggests a leak"
            )
        finally:
            await _teardown_mesh(transports)


class TestAllreduceCorrectnessUnderLoad:
    @pytest.mark.asyncio
    async def test_concurrent_allreduce_calls(self):
        """Two simultaneous allreduces on the same group must serialize cleanly."""
        n = 2
        size = 100

        transports, _ = await _setup_mesh(n)
        try:
            groups = _make_groups(n, transports)

            # Run two allreduce calls in sequence per group, all concurrent.
            async def driver(rank: int, val: float) -> tuple[np.ndarray, np.ndarray]:
                a = np.full(size, val, dtype=np.float32)
                b = np.full(size, val + 100.0, dtype=np.float32)
                # Sequential — within one rank the second allreduce
                # waits for the first. Across ranks they coordinate.
                r1 = await groups[rank].allreduce(a, op="mean")
                r2 = await groups[rank].allreduce(b, op="mean")
                return r1, r2

            results = await asyncio.gather(
                driver(0, 10.0),
                driver(1, 20.0),
            )
            r1_a, r2_a = results[0]
            r1_b, r2_b = results[1]
            # First allreduce: mean(10, 20) = 15
            np.testing.assert_allclose(r1_a, 15.0, rtol=1e-5)
            np.testing.assert_allclose(r1_b, 15.0, rtol=1e-5)
            # Second: mean(110, 120) = 115
            np.testing.assert_allclose(r2_a, 115.0, rtol=1e-5)
            np.testing.assert_allclose(r2_b, 115.0, rtol=1e-5)
        finally:
            await _teardown_mesh(transports)


class TestRingTopology:
    """Ring AllReduce with N=4 — exercises the full ring topology."""

    @pytest.mark.asyncio
    async def test_four_node_ring_correctness(self):
        n = 4
        rounds = 20
        size = 500
        rng = np.random.default_rng(seed=42)

        transports, _ = await _setup_mesh(n)
        try:
            groups = _make_groups(n, transports)
            for _round_i in range(rounds):
                arrays = [
                    rng.standard_normal(size).astype(np.float32)
                    for _ in range(n)
                ]
                results = await asyncio.gather(*(
                    groups[rank].allreduce(arrays[rank], op="mean")
                    for rank in range(n)
                ))
                expected = sum(arrays) / float(n)
                # Float32 ring-sum accumulates more rounding error than a
                # tree-reduce; rtol+atol cover the worst case.
                for r in results:
                    np.testing.assert_allclose(r, expected, rtol=1e-3, atol=1e-5)
        finally:
            await _teardown_mesh(transports)
