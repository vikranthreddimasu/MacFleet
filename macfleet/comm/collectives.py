"""Collective operations for MacFleet distributed training.

Implements AllReduce, Broadcast, Scatter, and Gather operations
optimized for small clusters connected via Thunderbolt.
"""

import asyncio
from typing import Callable, Optional

import torch

from macfleet.comm.transport import TensorTransport
from macfleet.utils.tensor_utils import MessageType


class CollectiveGroup:
    """A group of nodes participating in collective operations.

    Manages connections and coordinates collective communication
    between nodes in the cluster.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        transport: TensorTransport,
    ):
        """Initialize the collective group.

        Args:
            rank: This node's rank (0 = master).
            world_size: Total number of nodes.
            transport: Tensor transport layer.
        """
        self.rank = rank
        self.world_size = world_size
        self._transport = transport
        self._peer_connections: dict[int, str] = {}  # rank -> conn_key
        self._device = "cpu"

    def set_device(self, device: str) -> None:
        """Set the device for received tensors."""
        self._device = device

    async def connect_to_peer(self, peer_rank: int, host: str, port: int) -> None:
        """Connect to a peer node.

        Args:
            peer_rank: Rank of the peer.
            host: Peer's IP address.
            port: Peer's tensor port.
        """
        conn_key = await self._transport.connect(host, port)
        self._peer_connections[peer_rank] = conn_key

    async def disconnect_from_peer(self, peer_rank: int) -> None:
        """Disconnect from a peer node."""
        conn_key = self._peer_connections.pop(peer_rank, None)
        if conn_key:
            await self._transport.disconnect(conn_key)

    async def disconnect_all(self) -> None:
        """Disconnect from all peers."""
        for peer_rank in list(self._peer_connections.keys()):
            await self.disconnect_from_peer(peer_rank)


class AllReduce:
    """AllReduce operation that averages tensors across all nodes.

    For 2 nodes: Uses direct exchange where each node sends its
    tensor to the other, then both compute the average locally.

    For N>2 nodes: Uses Ring-AllReduce where the tensor is split
    into chunks and passed around a ring.
    """

    def __init__(
        self,
        group: CollectiveGroup,
        compress_fn: Optional[Callable] = None,
        decompress_fn: Optional[Callable] = None,
    ):
        """Initialize AllReduce.

        Args:
            group: Collective group for this operation.
            compress_fn: Optional compression function.
            decompress_fn: Optional decompression function.
        """
        self._group = group
        self._compress_fn = compress_fn
        self._decompress_fn = decompress_fn

    async def __call__(
        self,
        tensor: torch.Tensor,
        op: str = "mean",
    ) -> torch.Tensor:
        """Execute AllReduce on the tensor.

        Args:
            tensor: Local tensor to reduce.
            op: Reduction operation ("mean" or "sum").

        Returns:
            Reduced tensor (same value on all nodes).
        """
        if self._group.world_size == 1:
            return tensor

        if self._group.world_size == 2:
            return await self._direct_exchange(tensor, op)
        else:
            return await self._ring_allreduce(tensor, op)

    async def _direct_exchange(
        self,
        tensor: torch.Tensor,
        op: str,
    ) -> torch.Tensor:
        """Direct exchange AllReduce for 2 nodes.

        Both nodes send their tensors to each other simultaneously,
        then compute the reduction locally.
        """
        # Determine peer rank
        peer_rank = 1 - self._group.rank  # 0->1, 1->0

        # Get peer connection
        peer_conn = self._group._peer_connections.get(peer_rank)
        if not peer_conn:
            raise RuntimeError(f"Not connected to peer rank {peer_rank}")

        # Prepare tensor for sending
        send_tensor = tensor.cpu().contiguous()

        # Apply compression if available
        if self._compress_fn:
            send_data = self._compress_fn(send_tensor)
        else:
            send_data = send_tensor

        # Create tasks for send and receive
        async def send():
            if isinstance(send_data, tuple):
                # Compressed gradient (indices, values, numel, dtype)
                await self._group._transport.send_compressed_gradient(
                    send_data[0], send_data[1], send_data[2], send_data[3],
                    peer_conn,
                )
            else:
                await self._group._transport.send_tensor(
                    send_data, peer_conn, MessageType.TENSOR_GRADIENT
                )

        async def recv():
            if self._compress_fn:
                # Receive on CPU — decompression uses scatter_ which
                # doesn't work on MPS. We move to device after reduction.
                return await self._group._transport.recv_compressed_gradient(
                    peer_conn, "cpu"
                )
            else:
                return await self._group._transport.recv_tensor(
                    peer_conn, self._group._device
                )

        # Send and receive concurrently to avoid deadlock on large tensors.
        # Both sides send while also reading, so TCP flow control works.
        _, recv_data = await asyncio.gather(send(), recv())

        # Decompress if needed
        if self._decompress_fn and isinstance(recv_data, tuple):
            indices, values, numel, dtype = recv_data
            recv_tensor = self._decompress_fn(indices, values, numel, dtype)
        elif isinstance(recv_data, tuple):
            recv_tensor = recv_data[0]
        else:
            recv_tensor = recv_data

        # Move to same device as input
        if recv_tensor.device != tensor.device:
            recv_tensor = recv_tensor.to(tensor.device)

        # Compute reduction
        if op == "mean":
            result = (tensor + recv_tensor) / 2.0
        elif op == "sum":
            result = tensor + recv_tensor
        else:
            raise ValueError(f"Unknown reduction op: {op}")

        return result

    async def _ring_allreduce(
        self,
        tensor: torch.Tensor,
        op: str,
    ) -> torch.Tensor:
        """Ring AllReduce for N>2 nodes.

        Splits tensor into world_size chunks and passes them
        around a ring in two phases:
        1. Scatter-reduce: Each chunk is reduced as it passes
        2. Allgather: Reduced chunks are broadcast
        """
        world_size = self._group.world_size
        rank = self._group.rank

        # Flatten tensor
        original_shape = tensor.shape
        flat = tensor.flatten()
        numel = flat.numel()

        # Pad to be divisible by world_size
        pad_size = (world_size - numel % world_size) % world_size
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size, device=tensor.device)])

        # Split into chunks
        chunk_size = flat.numel() // world_size
        chunks = list(flat.split(chunk_size))

        # Determine neighbors in ring
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        left_conn = self._group._peer_connections.get(left_rank)
        right_conn = self._group._peer_connections.get(right_rank)

        if not left_conn or not right_conn:
            raise RuntimeError("Ring topology not fully connected")

        # Phase 1: Scatter-reduce
        # Use asyncio.gather for concurrent send/recv to avoid deadlock
        # when N>2 (sequential send-then-recv fills TCP buffers and stalls).
        for step in range(world_size - 1):
            send_idx = (rank - step) % world_size
            recv_idx = (rank - step - 1) % world_size

            send_chunk = chunks[send_idx].cpu()

            async def _send_scatter(chunk=send_chunk):
                await self._group._transport.send_tensor(
                    chunk, right_conn, MessageType.TENSOR_GRADIENT
                )

            async def _recv_scatter():
                return await self._group._transport.recv_tensor(
                    left_conn, str(tensor.device)
                )

            _, (recv_chunk, _) = await asyncio.gather(
                _send_scatter(), _recv_scatter()
            )

            # Scatter-reduce always sums; division happens after the loop.
            chunks[recv_idx] = chunks[recv_idx] + recv_chunk

        # Apply mean reduction after scatter-reduce completes
        if op == "mean":
            for i in range(len(chunks)):
                chunks[i] = chunks[i] / world_size

        # Phase 2: Allgather
        for step in range(world_size - 1):
            send_idx = (rank - step + 1) % world_size
            recv_idx = (rank - step) % world_size

            send_chunk = chunks[send_idx].cpu()

            async def _send_gather(chunk=send_chunk):
                await self._group._transport.send_tensor(
                    chunk, right_conn, MessageType.TENSOR_GRADIENT
                )

            async def _recv_gather():
                return await self._group._transport.recv_tensor(
                    left_conn, str(tensor.device)
                )

            _, (recv_chunk, _) = await asyncio.gather(
                _send_gather(), _recv_gather()
            )

            chunks[recv_idx] = recv_chunk

        # Reassemble
        result = torch.cat(chunks)[:numel].view(original_shape)
        return result


class Broadcast:
    """Broadcast a tensor from source rank to all other nodes."""

    def __init__(self, group: CollectiveGroup):
        self._group = group

    async def __call__(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Broadcast tensor from src to all nodes.

        Args:
            tensor: Tensor to broadcast (only used on src rank).
            src: Source rank.

        Returns:
            Broadcast tensor (same on all nodes).
        """
        if self._group.world_size == 1:
            return tensor

        if self._group.rank == src:
            # Send to all other nodes concurrently to avoid deadlock
            send_tensor = tensor.cpu().contiguous()
            await asyncio.gather(*(
                self._group._transport.send_tensor(
                    send_tensor, conn_key, MessageType.TENSOR_WEIGHTS
                )
                for conn_key in self._group._peer_connections.values()
            ))
            return tensor
        else:
            # Receive from source
            src_conn = self._group._peer_connections.get(src)
            if not src_conn:
                raise RuntimeError(f"Not connected to source rank {src}")

            recv_tensor, _ = await self._group._transport.recv_tensor(
                src_conn, self._group._device
            )
            return recv_tensor


class Scatter:
    """Scatter tensor chunks from source to all nodes."""

    def __init__(self, group: CollectiveGroup):
        self._group = group

    async def __call__(
        self,
        tensor: Optional[torch.Tensor],
        src: int = 0,
    ) -> torch.Tensor:
        """Scatter tensor from src to all nodes.

        Args:
            tensor: Tensor to scatter (only on src, should be
                   divisible into world_size chunks).
            src: Source rank.

        Returns:
            This node's chunk of the tensor.
        """
        if self._group.world_size == 1:
            return tensor

        if self._group.rank == src:
            # Split and send chunks
            chunks = tensor.chunk(self._group.world_size)
            for peer_rank, conn_key in self._group._peer_connections.items():
                chunk = chunks[peer_rank].cpu().contiguous()
                await self._group._transport.send_tensor(
                    chunk, conn_key, MessageType.TENSOR_WEIGHTS
                )
            return chunks[src]
        else:
            # Receive my chunk
            src_conn = self._group._peer_connections.get(src)
            if not src_conn:
                raise RuntimeError(f"Not connected to source rank {src}")

            recv_chunk, _ = await self._group._transport.recv_tensor(
                src_conn, self._group._device
            )
            return recv_chunk


class Gather:
    """Gather tensors from all nodes to destination."""

    def __init__(self, group: CollectiveGroup):
        self._group = group

    async def __call__(
        self,
        tensor: torch.Tensor,
        dst: int = 0,
    ) -> Optional[torch.Tensor]:
        """Gather tensors to dst.

        Args:
            tensor: This node's tensor.
            dst: Destination rank.

        Returns:
            Concatenated tensor (only on dst), None otherwise.
        """
        if self._group.world_size == 1:
            return tensor.unsqueeze(0)

        if self._group.rank == dst:
            # Collect from all peers concurrently to avoid deadlock
            chunks = [None] * self._group.world_size
            chunks[dst] = tensor

            peer_items = list(self._group._peer_connections.items())

            async def _recv_from(conn_key):
                return await self._group._transport.recv_tensor(
                    conn_key, self._group._device
                )

            results = await asyncio.gather(*(
                _recv_from(conn_key) for _, conn_key in peer_items
            ))
            for (peer_rank, _), (recv_tensor, _) in zip(peer_items, results):
                chunks[peer_rank] = recv_tensor

            return torch.stack(chunks)
        else:
            # Send to destination
            dst_conn = self._group._peer_connections.get(dst)
            if not dst_conn:
                raise RuntimeError(f"Not connected to dst rank {dst}")

            send_tensor = tensor.cpu().contiguous()
            await self._group._transport.send_tensor(
                send_tensor, dst_conn, MessageType.TENSOR_GRADIENT
            )
            return None


async def allreduce(
    tensor: torch.Tensor,
    group: CollectiveGroup,
    op: str = "mean",
) -> torch.Tensor:
    """Convenience function for AllReduce.

    Args:
        tensor: Tensor to reduce.
        group: Collective group.
        op: Reduction operation.

    Returns:
        Reduced tensor.
    """
    allreduce_op = AllReduce(group)
    return await allreduce_op(tensor, op)


async def broadcast(
    tensor: torch.Tensor,
    group: CollectiveGroup,
    src: int = 0,
) -> torch.Tensor:
    """Convenience function for Broadcast.

    Args:
        tensor: Tensor to broadcast.
        group: Collective group.
        src: Source rank.

    Returns:
        Broadcast tensor.
    """
    broadcast_op = Broadcast(group)
    return await broadcast_op(tensor, src)
