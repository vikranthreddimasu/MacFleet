"""Tests for N-node collective operations.

Uses loopback TCP connections to simulate multi-node clusters.
Each test creates N PeerTransport instances, connects them in a mesh,
and runs the collective operation concurrently on all ranks.
"""

import asyncio

import numpy as np
import pytest

from macfleet.comm.collectives import (
    CollectiveGroup,
    pack_array,
    unpack_array,
)
from macfleet.comm.transport import PeerTransport, TransportConfig


# --------------------------------------------------------------------------- #
# Array serialization tests                                                   #
# --------------------------------------------------------------------------- #


class TestArraySerialization:
    def test_float32_roundtrip(self):
        arr = np.random.randn(10, 20).astype(np.float32)
        data = pack_array(arr)
        result = unpack_array(data)
        np.testing.assert_array_equal(result, arr)

    def test_float64_roundtrip(self):
        arr = np.random.randn(5, 5).astype(np.float64)
        result = unpack_array(pack_array(arr))
        np.testing.assert_array_equal(result, arr)

    def test_int32_roundtrip(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = unpack_array(pack_array(arr))
        np.testing.assert_array_equal(result, arr)

    def test_1d_shape(self):
        arr = np.zeros(100, dtype=np.float32)
        result = unpack_array(pack_array(arr))
        assert result.shape == (100,)

    def test_3d_shape(self):
        arr = np.ones((2, 3, 4), dtype=np.float32)
        result = unpack_array(pack_array(arr))
        assert result.shape == (2, 3, 4)

    def test_large_array(self):
        arr = np.random.randn(1000, 1000).astype(np.float32)
        result = unpack_array(pack_array(arr))
        np.testing.assert_array_equal(result, arr)


# --------------------------------------------------------------------------- #
# Helpers for multi-node loopback tests                                       #
# --------------------------------------------------------------------------- #

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=10.0)


async def _setup_mesh(n: int) -> tuple[list[PeerTransport], list[int]]:
    """Create N transports connected in a full mesh over loopback.

    Returns (transports, ports) where transports[i] has local_id=f"node-{i}"
    and is connected to all other transports.
    """
    transports = []
    ports = []

    # Start servers
    for i in range(n):
        t = PeerTransport(local_id=f"node-{i}", config=CONFIG)
        await t.start_server("127.0.0.1", 0)
        port = t._server.sockets[0].getsockname()[1]
        transports.append(t)
        ports.append(port)

    # Connect each pair (i connects to j where j > i)
    for i in range(n):
        for j in range(i + 1, n):
            await transports[i].connect(f"node-{j}", "127.0.0.1", ports[j])

    # Wait for all connections to settle
    await asyncio.sleep(0.2)

    return transports, ports


async def _teardown_mesh(transports: list[PeerTransport]) -> None:
    """Disconnect and stop all transports."""
    for t in transports:
        await t.disconnect_all()


def _make_groups(
    transports: list[PeerTransport],
) -> list[CollectiveGroup]:
    """Create CollectiveGroups for each transport (rank = index)."""
    n = len(transports)
    groups = []
    for rank in range(n):
        rank_to_peer = {r: f"node-{r}" for r in range(n) if r != rank}
        groups.append(
            CollectiveGroup(
                rank=rank,
                world_size=n,
                transport=transports[rank],
                rank_to_peer=rank_to_peer,
            )
        )
    return groups


# --------------------------------------------------------------------------- #
# AllReduce tests                                                             #
# --------------------------------------------------------------------------- #


class TestAllReduce:
    @pytest.mark.asyncio
    async def test_single_node(self):
        """AllReduce with world_size=1 returns the input unchanged."""
        t = PeerTransport(local_id="solo", config=CONFIG)
        group = CollectiveGroup(rank=0, world_size=1, transport=t, rank_to_peer={})
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = await group.allreduce(arr)
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.asyncio
    async def test_two_nodes_mean(self):
        """AllReduce mean with 2 nodes produces the average."""
        transports, _ = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            arr0 = np.array([2.0, 4.0, 6.0], dtype=np.float32)
            arr1 = np.array([4.0, 8.0, 12.0], dtype=np.float32)

            r0, r1 = await asyncio.gather(
                groups[0].allreduce(arr0, op="mean"),
                groups[1].allreduce(arr1, op="mean"),
            )

            expected = np.array([3.0, 6.0, 9.0], dtype=np.float32)
            np.testing.assert_allclose(r0, expected, rtol=1e-5)
            np.testing.assert_allclose(r1, expected, rtol=1e-5)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_two_nodes_sum(self):
        """AllReduce sum with 2 nodes."""
        transports, _ = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            arr0 = np.array([1.0, 2.0], dtype=np.float32)
            arr1 = np.array([3.0, 4.0], dtype=np.float32)

            r0, r1 = await asyncio.gather(
                groups[0].allreduce(arr0, op="sum"),
                groups[1].allreduce(arr1, op="sum"),
            )

            expected = np.array([4.0, 6.0], dtype=np.float32)
            np.testing.assert_allclose(r0, expected, rtol=1e-5)
            np.testing.assert_allclose(r1, expected, rtol=1e-5)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_three_nodes_mean(self):
        """Ring AllReduce mean with 3 nodes (exercises ring topology)."""
        transports, _ = await _setup_mesh(3)
        try:
            groups = _make_groups(transports)

            arrays = [
                np.array([3.0, 6.0, 9.0], dtype=np.float32),
                np.array([6.0, 12.0, 18.0], dtype=np.float32),
                np.array([9.0, 18.0, 27.0], dtype=np.float32),
            ]

            results = await asyncio.gather(
                *(groups[i].allreduce(arrays[i], op="mean") for i in range(3))
            )

            expected = np.array([6.0, 12.0, 18.0], dtype=np.float32)
            for r in results:
                np.testing.assert_allclose(r, expected, rtol=1e-5)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_four_nodes_mean(self):
        """Ring AllReduce mean with 4 nodes."""
        transports, _ = await _setup_mesh(4)
        try:
            groups = _make_groups(transports)

            arrays = [np.full(10, float(i + 1), dtype=np.float32) for i in range(4)]

            results = await asyncio.gather(
                *(groups[i].allreduce(arrays[i], op="mean") for i in range(4))
            )

            expected = np.full(10, 2.5, dtype=np.float32)  # mean(1,2,3,4)
            for r in results:
                np.testing.assert_allclose(r, expected, rtol=1e-5)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_uneven_array_size(self):
        """AllReduce handles arrays not evenly divisible by world_size."""
        transports, _ = await _setup_mesh(3)
        try:
            groups = _make_groups(transports)

            # 7 elements, 3 nodes — not evenly divisible
            arrays = [
                np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
                np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32),
                np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0], dtype=np.float32),
            ]

            results = await asyncio.gather(
                *(groups[i].allreduce(arrays[i], op="mean") for i in range(3))
            )

            expected = (arrays[0] + arrays[1] + arrays[2]) / 3.0
            for r in results:
                np.testing.assert_allclose(r, expected, rtol=1e-4)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_2d_array(self):
        """AllReduce preserves multi-dimensional shape."""
        transports, _ = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            arr0 = np.ones((3, 4), dtype=np.float32) * 2
            arr1 = np.ones((3, 4), dtype=np.float32) * 4

            r0, r1 = await asyncio.gather(
                groups[0].allreduce(arr0, op="mean"),
                groups[1].allreduce(arr1, op="mean"),
            )

            expected = np.ones((3, 4), dtype=np.float32) * 3
            np.testing.assert_allclose(r0, expected, rtol=1e-5)
            assert r0.shape == (3, 4)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_gradient_like(self):
        """AllReduce with realistic gradient-sized arrays."""
        transports, _ = await _setup_mesh(3)
        try:
            groups = _make_groups(transports)

            # Simulate gradient: 1000 params
            np.random.seed(42)
            arrays = [np.random.randn(1000).astype(np.float32) for _ in range(3)]

            results = await asyncio.gather(
                *(groups[i].allreduce(arrays[i], op="mean") for i in range(3))
            )

            expected = sum(arrays) / 3.0
            for r in results:
                np.testing.assert_allclose(r, expected, rtol=5e-4)
        finally:
            await _teardown_mesh(transports)


# --------------------------------------------------------------------------- #
# Broadcast tests                                                             #
# --------------------------------------------------------------------------- #


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_single_node(self):
        """Broadcast with 1 node is a no-op."""
        t = PeerTransport(local_id="solo", config=CONFIG)
        group = CollectiveGroup(rank=0, world_size=1, transport=t, rank_to_peer={})
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = await group.broadcast(arr)
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.asyncio
    async def test_two_nodes(self):
        """Broadcast from rank 0 to rank 1."""
        transports, _ = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            source_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
            dummy_arr = np.zeros(3, dtype=np.float32)

            r0, r1 = await asyncio.gather(
                groups[0].broadcast(source_arr, src=0),
                groups[1].broadcast(dummy_arr, src=0),
            )

            np.testing.assert_array_equal(r0, source_arr)
            np.testing.assert_array_equal(r1, source_arr)
        finally:
            await _teardown_mesh(transports)

    @pytest.mark.asyncio
    async def test_three_nodes_from_rank1(self):
        """Broadcast from a non-zero source rank."""
        transports, _ = await _setup_mesh(3)
        try:
            groups = _make_groups(transports)

            source_arr = np.array([42.0, 99.0], dtype=np.float32)
            dummy = np.zeros(2, dtype=np.float32)

            results = await asyncio.gather(
                groups[0].broadcast(dummy, src=1),
                groups[1].broadcast(source_arr, src=1),
                groups[2].broadcast(dummy, src=1),
            )

            for r in results:
                np.testing.assert_array_equal(r, source_arr)
        finally:
            await _teardown_mesh(transports)


# --------------------------------------------------------------------------- #
# Scatter / Gather tests                                                      #
# --------------------------------------------------------------------------- #


class TestScatter:
    @pytest.mark.asyncio
    async def test_single_node(self):
        """Scatter with 1 node returns the full array."""
        t = PeerTransport(local_id="solo", config=CONFIG)
        group = CollectiveGroup(rank=0, world_size=1, transport=t, rank_to_peer={})
        arr = np.arange(12, dtype=np.float32)
        result = await group.scatter(arr)
        np.testing.assert_array_equal(result, arr)

    @pytest.mark.asyncio
    async def test_two_nodes(self):
        """Scatter splits array into 2 chunks."""
        transports, _ = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            full = np.arange(10, dtype=np.float32)

            r0, r1 = await asyncio.gather(
                groups[0].scatter(full, src=0),
                groups[1].scatter(None, src=0),
            )

            # array_split(10, 2) → [5, 5]
            np.testing.assert_array_equal(r0, np.arange(5, dtype=np.float32))
            np.testing.assert_array_equal(r1, np.arange(5, 10, dtype=np.float32))
        finally:
            await _teardown_mesh(transports)


class TestGather:
    @pytest.mark.asyncio
    async def test_single_node(self):
        """Gather with 1 node wraps in extra dimension."""
        t = PeerTransport(local_id="solo", config=CONFIG)
        group = CollectiveGroup(rank=0, world_size=1, transport=t, rank_to_peer={})
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = await group.gather(arr)
        assert result.shape == (1, 2)

    @pytest.mark.asyncio
    async def test_two_nodes(self):
        """Gather collects arrays from 2 nodes to rank 0."""
        transports, _ = await _setup_mesh(2)
        try:
            groups = _make_groups(transports)

            arr0 = np.array([1.0, 2.0], dtype=np.float32)
            arr1 = np.array([3.0, 4.0], dtype=np.float32)

            r0, r1 = await asyncio.gather(
                groups[0].gather(arr0, dst=0),
                groups[1].gather(arr1, dst=0),
            )

            assert r0.shape == (2, 2)
            np.testing.assert_array_equal(r0[0], arr0)
            np.testing.assert_array_equal(r0[1], arr1)
            assert r1 is None  # non-dst rank gets None
        finally:
            await _teardown_mesh(transports)
