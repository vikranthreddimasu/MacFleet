"""Tests for collective operations."""

import asyncio

import torch

from macfleet.comm.collectives import (
    AllReduce,
    Broadcast,
    CollectiveGroup,
    Gather,
    Scatter,
)
from macfleet.comm.transport import TensorTransport


class TestCollectiveGroup:
    """Tests for CollectiveGroup."""

    def test_group_creation(self):
        """Test creating a collective group."""
        transport = TensorTransport()
        group = CollectiveGroup(rank=0, world_size=2, transport=transport)

        assert group.rank == 0
        assert group.world_size == 2

    def test_set_device(self):
        """Test setting device."""
        transport = TensorTransport()
        group = CollectiveGroup(rank=0, world_size=2, transport=transport)

        group.set_device("cpu")
        assert group._device == "cpu"


class TestAllReduce:
    """Tests for AllReduce operation."""

    def test_single_node_allreduce(self):
        """Test AllReduce with single node (should be no-op)."""
        transport = TensorTransport()
        group = CollectiveGroup(rank=0, world_size=1, transport=transport)
        allreduce = AllReduce(group)

        tensor = torch.randn(100)

        async def run():
            result = await allreduce(tensor)
            return result

        result = asyncio.run(run())

        assert torch.equal(tensor, result)

    def test_allreduce_with_different_ops(self):
        """Test AllReduce with different ops (single node)."""
        transport = TensorTransport()
        group = CollectiveGroup(rank=0, world_size=1, transport=transport)
        allreduce = AllReduce(group)

        tensor = torch.randn(100)

        async def run_mean():
            return await allreduce(tensor.clone(), op="mean")

        async def run_sum():
            return await allreduce(tensor.clone(), op="sum")

        result_mean = asyncio.run(run_mean())
        result_sum = asyncio.run(run_sum())

        # For single node, both should return the original tensor
        assert torch.equal(tensor, result_mean)
        assert torch.equal(tensor, result_sum)


class TestBroadcast:
    """Tests for Broadcast operation."""

    def test_single_node_broadcast(self):
        """Test Broadcast with single node."""
        transport = TensorTransport()
        group = CollectiveGroup(rank=0, world_size=1, transport=transport)
        broadcast = Broadcast(group)

        tensor = torch.randn(100)

        async def run():
            return await broadcast(tensor, src=0)

        result = asyncio.run(run())

        assert torch.equal(tensor, result)


class TestScatter:
    """Tests for Scatter operation."""

    def test_single_node_scatter(self):
        """Test Scatter with single node."""
        transport = TensorTransport()
        group = CollectiveGroup(rank=0, world_size=1, transport=transport)
        scatter = Scatter(group)

        tensor = torch.randn(100)

        async def run():
            return await scatter(tensor, src=0)

        result = asyncio.run(run())

        assert torch.equal(tensor, result)


class TestGather:
    """Tests for Gather operation."""

    def test_single_node_gather(self):
        """Test Gather with single node."""
        transport = TensorTransport()
        group = CollectiveGroup(rank=0, world_size=1, transport=transport)
        gather = Gather(group)

        tensor = torch.randn(100)

        async def run():
            return await gather(tensor, dst=0)

        result = asyncio.run(run())

        # Should add batch dimension
        assert result.shape == (1, 100)
        assert torch.equal(result[0], tensor)
