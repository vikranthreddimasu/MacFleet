"""N-node collective operations for MacFleet v2.

Framework-agnostic: operates on numpy arrays, not torch/mlx tensors.
The engine layer converts between framework tensors and numpy.

Algorithms:
    N=1:  No-op (return local data)
    N=2:  Direct exchange (simultaneous send/recv)
    N>=3: Ring AllReduce (scatter-reduce + allgather)

Supported operations:
    allreduce  — Average/sum arrays across all nodes
    broadcast  — Send array from one rank to all
    scatter    — Split array from one rank to all
    gather     — Collect arrays from all ranks to one
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

from macfleet.comm.protocol import MessageType
from macfleet.comm.transport import PeerTransport

# --------------------------------------------------------------------------- #
# Array serialization (lightweight, no external deps beyond numpy)            #
# Format: [dtype_len:H][dtype_str][ndims:H][shape:ndims*I][data]              #
# --------------------------------------------------------------------------- #


def pack_array(array: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes with shape/dtype metadata."""
    dtype_str = str(array.dtype).encode("utf-8")
    ndims = len(array.shape)
    header = struct.pack(f"!HH{'I' * ndims}", len(dtype_str), ndims, *array.shape)
    return header + dtype_str + array.tobytes()


def unpack_array(data: bytes) -> np.ndarray:
    """Deserialize bytes back to a numpy array."""
    dtype_len, ndims = struct.unpack("!HH", data[:4])
    shape_start = 4
    shape_end = shape_start + ndims * 4
    shape = struct.unpack(f"!{'I' * ndims}", data[shape_start:shape_end])
    dtype_str = data[shape_end : shape_end + dtype_len].decode("utf-8")
    data_start = shape_end + dtype_len
    array = np.frombuffer(data[data_start:], dtype=np.dtype(dtype_str)).reshape(shape)
    return array.copy()  # own the memory (frombuffer returns read-only view)


# --------------------------------------------------------------------------- #
# Collective group                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class CollectiveConfig:
    """Configuration for collective operations."""

    recv_timeout_sec: float = 120.0


class CollectiveGroup:
    """A group of nodes participating in collective operations.

    Maps integer ranks (0 .. world_size-1) to peer IDs and coordinates
    the communication pattern over PeerTransport.

    Usage:
        group = CollectiveGroup(
            rank=0, world_size=3,
            transport=transport,
            rank_to_peer={1: "node-b", 2: "node-c"},
        )
        averaged = await group.allreduce(my_gradients, op="mean")
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        transport: PeerTransport,
        rank_to_peer: dict[int, str],
        config: Optional[CollectiveConfig] = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self._transport = transport
        self._rank_to_peer = rank_to_peer  # rank -> peer_id (excludes self)
        self._config = config or CollectiveConfig()

    def _peer_for_rank(self, rank: int) -> str:
        """Resolve a rank to a peer_id."""
        peer_id = self._rank_to_peer.get(rank)
        if peer_id is None:
            raise RuntimeError(f"No peer mapped to rank {rank}")
        return peer_id

    async def _send_array(self, rank: int, array: np.ndarray) -> None:
        """Send a numpy array to the peer at the given rank."""
        peer_id = self._peer_for_rank(rank)
        payload = pack_array(array)
        await self._transport.send(peer_id, payload, msg_type=MessageType.GRADIENT)

    async def _recv_array(self, rank: int) -> np.ndarray:
        """Receive a numpy array from the peer at the given rank."""
        peer_id = self._peer_for_rank(rank)
        payload = await self._transport.recv(peer_id)
        return unpack_array(payload)

    # ------------------------------------------------------------------ #
    # AllReduce                                                          #
    # ------------------------------------------------------------------ #

    async def allreduce(self, array: np.ndarray, op: str = "mean") -> np.ndarray:
        """AllReduce: average or sum arrays across all nodes.

        Args:
            array: Local numpy array (same shape on all nodes).
            op: "mean" or "sum".

        Returns:
            Reduced array (identical on all nodes).
        """
        if self.world_size == 1:
            return array

        if self.world_size == 2:
            return await self._direct_allreduce(array, op)

        return await self._ring_allreduce(array, op)

    async def _direct_allreduce(
        self, array: np.ndarray, op: str
    ) -> np.ndarray:
        """Direct exchange AllReduce for exactly 2 nodes.

        Both nodes send their arrays simultaneously, then reduce locally.
        """
        peer_rank = 1 - self.rank  # 0↔1

        async def _send() -> None:
            await self._send_array(peer_rank, array)

        async def _recv() -> np.ndarray:
            return await self._recv_array(peer_rank)

        _, remote = await asyncio.gather(_send(), _recv())

        if op == "mean":
            return (array + remote) / 2.0
        elif op == "sum":
            return array + remote
        raise ValueError(f"Unknown reduction op: {op}")

    async def _ring_allreduce(
        self, array: np.ndarray, op: str
    ) -> np.ndarray:
        """Ring AllReduce for N >= 3 nodes.

        Two phases:
            1. Scatter-reduce: each chunk accumulates as it passes around the ring
            2. Allgather: reduced chunks are broadcast around the ring
        """
        ws = self.world_size
        rank = self.rank

        # Flatten for chunking
        original_shape = array.shape
        original_dtype = array.dtype
        flat = array.flatten()
        numel = len(flat)

        # Pad to be evenly divisible
        pad_size = (ws - numel % ws) % ws
        if pad_size > 0:
            flat = np.concatenate([flat, np.zeros(pad_size, dtype=flat.dtype)])

        chunk_size = len(flat) // ws
        chunks = [flat[i * chunk_size : (i + 1) * chunk_size].copy() for i in range(ws)]

        left_rank = (rank - 1) % ws
        right_rank = (rank + 1) % ws

        # Phase 1: Scatter-reduce
        for step in range(ws - 1):
            send_idx = (rank - step) % ws
            recv_idx = (rank - step - 1) % ws

            send_chunk = chunks[send_idx]

            async def _send(chunk=send_chunk) -> None:
                await self._send_array(right_rank, chunk)

            async def _recv() -> np.ndarray:
                return await self._recv_array(left_rank)

            _, received = await asyncio.gather(_send(), _recv())
            chunks[recv_idx] = chunks[recv_idx] + received

        # Apply reduction
        if op == "mean":
            for i in range(ws):
                chunks[i] = chunks[i] / ws

        # Phase 2: Allgather
        for step in range(ws - 1):
            send_idx = (rank - step + 1) % ws
            recv_idx = (rank - step) % ws

            send_chunk = chunks[send_idx]

            async def _send(chunk=send_chunk) -> None:
                await self._send_array(right_rank, chunk)

            async def _recv() -> np.ndarray:
                return await self._recv_array(left_rank)

            _, received = await asyncio.gather(_send(), _recv())
            chunks[recv_idx] = received

        # Reassemble and restore shape
        result = np.concatenate(chunks)[:numel].reshape(original_shape)
        return result.astype(original_dtype)

    # ------------------------------------------------------------------ #
    # Broadcast                                                          #
    # ------------------------------------------------------------------ #

    async def broadcast(self, array: np.ndarray, src: int = 0) -> np.ndarray:
        """Broadcast array from src rank to all other ranks.

        Args:
            array: Array to broadcast (only meaningful on src).
            src: Source rank.

        Returns:
            The broadcast array (identical on all nodes).
        """
        if self.world_size == 1:
            return array

        if self.rank == src:
            sends = [
                self._send_array(r, array)
                for r in range(self.world_size)
                if r != src
            ]
            await asyncio.gather(*sends)
            return array
        else:
            return await self._recv_array(src)

    # ------------------------------------------------------------------ #
    # Scatter / Gather                                                   #
    # ------------------------------------------------------------------ #

    async def scatter(
        self, array: Optional[np.ndarray], src: int = 0
    ) -> np.ndarray:
        """Scatter: split array on src and send one chunk to each rank.

        Args:
            array: Array to scatter (only used on src rank).
            src: Source rank.

        Returns:
            This rank's chunk of the array.
        """
        if self.world_size == 1:
            return array

        if self.rank == src:
            chunks = np.array_split(array, self.world_size)
            sends = [
                self._send_array(r, chunks[r])
                for r in range(self.world_size)
                if r != src
            ]
            await asyncio.gather(*sends)
            return chunks[src]
        else:
            return await self._recv_array(src)

    async def gather(
        self, array: np.ndarray, dst: int = 0
    ) -> Optional[np.ndarray]:
        """Gather: collect arrays from all ranks to dst.

        Args:
            array: This rank's array.
            dst: Destination rank.

        Returns:
            Stacked array on dst, None on other ranks.
        """
        if self.world_size == 1:
            return np.expand_dims(array, 0)

        if self.rank == dst:
            chunks = [None] * self.world_size
            chunks[dst] = array

            recv_ranks = [r for r in range(self.world_size) if r != dst]
            results = await asyncio.gather(
                *(self._recv_array(r) for r in recv_ranks)
            )
            for r, received in zip(recv_ranks, results):
                chunks[r] = received

            return np.stack(chunks)
        else:
            await self._send_array(dst, array)
            return None


# --------------------------------------------------------------------------- #
# Convenience functions                                                       #
# --------------------------------------------------------------------------- #


async def allreduce(
    array: np.ndarray,
    group: CollectiveGroup,
    op: str = "mean",
) -> np.ndarray:
    """Convenience: AllReduce an array across the group."""
    return await group.allreduce(array, op)


async def broadcast(
    array: np.ndarray,
    group: CollectiveGroup,
    src: int = 0,
) -> np.ndarray:
    """Convenience: Broadcast an array from src."""
    return await group.broadcast(array, src)
