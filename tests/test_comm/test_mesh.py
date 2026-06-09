"""Tests for macfleet.training.mesh — SPMD rendezvous into a CollectiveGroup.

Framework-agnostic (numpy only). Each test runs every node's form_mesh()
concurrently in one event loop, exactly the way real nodes interleave at
their await points.
"""

from __future__ import annotations

import asyncio
import socket

import numpy as np
import pytest

from macfleet.comm.transport import PeerAuthError, TransportConfig
from macfleet.security.auth import SecurityConfig
from macfleet.training.mesh import (
    Mesh,
    MeshFormationError,
    NodeSpec,
    derive_ranks,
    form_mesh,
)

CONFIG = TransportConfig(connect_timeout_sec=5.0, recv_timeout_sec=10.0)


def _free_ports(n: int) -> list[int]:
    """Pre-pick n distinct ephemeral ports (bind all, then release all)."""
    socks = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _make_specs(n: int) -> list[NodeSpec]:
    ports = _free_ports(n)
    return [
        NodeSpec(node_id=f"node-{i}", ip_address="127.0.0.1", data_port=ports[i])
        for i in range(n)
    ]


async def _run_node(
    local_id: str,
    nodes: list[NodeSpec],
    array: np.ndarray,
    security: SecurityConfig | None = None,
    rendezvous_timeout_sec: float = 15.0,
) -> tuple[Mesh, np.ndarray]:
    mesh = await form_mesh(
        local_id,
        nodes,
        security=security,
        config=CONFIG,
        bind_host="127.0.0.1",
        rendezvous_timeout_sec=rendezvous_timeout_sec,
    )
    try:
        result = await mesh.group.allreduce(array, op="mean")
        return mesh, result
    finally:
        await mesh.close()


class TestDeriveRanks:
    def test_lexicographic_order(self):
        nodes = [
            NodeSpec("mac-c", "10.0.0.3", 1),
            NodeSpec("mac-a", "10.0.0.1", 2),
            NodeSpec("mac-b", "10.0.0.2", 3),
        ]
        assert derive_ranks(nodes) == {"mac-a": 0, "mac-b": 1, "mac-c": 2}

    def test_identical_on_any_input_order(self):
        nodes = [NodeSpec(f"n-{i}", "127.0.0.1", i) for i in range(5)]
        forward = derive_ranks(nodes)
        backward = derive_ranks(list(reversed(nodes)))
        assert forward == backward


class TestMeshValidation:
    @pytest.mark.asyncio
    async def test_local_id_not_in_spec(self):
        with pytest.raises(ValueError, match="not present"):
            await form_mesh("ghost", _make_specs(2))

    @pytest.mark.asyncio
    async def test_duplicate_node_ids(self):
        specs = [
            NodeSpec("same", "127.0.0.1", 1),
            NodeSpec("same", "127.0.0.1", 2),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            await form_mesh("same", specs)

    @pytest.mark.asyncio
    async def test_world_size_one_is_noop(self):
        spec = NodeSpec("solo", "127.0.0.1", 1)
        mesh = await form_mesh("solo", [spec])
        assert mesh.rank == 0
        assert mesh.world_size == 1
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = await mesh.group.allreduce(arr, op="mean")
        np.testing.assert_array_equal(result, arr)
        await mesh.close()


class TestMeshFormation:
    @pytest.mark.asyncio
    async def test_two_node_allreduce(self):
        specs = _make_specs(2)
        a0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a1 = np.array([3.0, 4.0, 5.0], dtype=np.float32)

        (m0, r0), (m1, r1) = await asyncio.gather(
            _run_node("node-0", specs, a0),
            _run_node("node-1", specs, a1),
        )
        expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(r0, expected)
        np.testing.assert_allclose(r1, expected)
        assert {m0.rank, m1.rank} == {0, 1}
        assert m0.world_size == m1.world_size == 2

    @pytest.mark.asyncio
    async def test_three_node_ring_allreduce(self):
        """N=3 exercises the ring-allreduce path through a formed mesh."""
        specs = _make_specs(3)
        arrays = [np.full(10, float(i + 1), dtype=np.float32) for i in range(3)]

        results = await asyncio.gather(
            *(_run_node(f"node-{i}", specs, arrays[i]) for i in range(3))
        )
        expected = np.full(10, 2.0, dtype=np.float32)  # mean(1, 2, 3)
        for mesh, reduced in results:
            np.testing.assert_allclose(reduced, expected, rtol=1e-6)
        assert sorted(m.rank for m, _ in results) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_staggered_start(self):
        """A late-starting peer is absorbed by the connect-retry loop."""
        specs = _make_specs(2)
        a0 = np.zeros(4, dtype=np.float32)
        a1 = np.full(4, 2.0, dtype=np.float32)

        async def _late_node1():
            await asyncio.sleep(1.0)
            return await _run_node("node-1", specs, a1)

        (_, r0), (_, r1) = await asyncio.gather(
            _run_node("node-0", specs, a0),
            _late_node1(),
        )
        np.testing.assert_allclose(r0, np.full(4, 1.0, dtype=np.float32))
        np.testing.assert_allclose(r1, np.full(4, 1.0, dtype=np.float32))


class TestSecureMesh:
    @pytest.mark.asyncio
    async def test_token_secured_allreduce(self):
        """Mesh over HMAC-authenticated, TLS-encrypted transports."""
        specs = _make_specs(2)
        sec = [SecurityConfig(token="mesh-test-token-123") for _ in range(2)]
        a0 = np.array([10.0, 20.0], dtype=np.float32)
        a1 = np.array([30.0, 40.0], dtype=np.float32)

        (_, r0), (_, r1) = await asyncio.gather(
            _run_node("node-0", specs, a0, security=sec[0]),
            _run_node("node-1", specs, a1, security=sec[1]),
        )
        expected = np.array([20.0, 30.0], dtype=np.float32)
        np.testing.assert_allclose(r0, expected)
        np.testing.assert_allclose(r1, expected)

    @pytest.mark.asyncio
    async def test_wrong_token_fails_fast(self):
        """A token mismatch is a PeerAuthError on the connecting side
        (no retry-until-deadline) and a rendezvous timeout on the
        accepting side."""
        specs = _make_specs(2)
        good = SecurityConfig(token="correct-token-abc")
        bad = SecurityConfig(token="wrong-token-xyz!")

        results = await asyncio.gather(
            _run_node("node-0", specs, np.zeros(2), security=good,
                      rendezvous_timeout_sec=5.0),
            _run_node("node-1", specs, np.zeros(2), security=bad,
                      rendezvous_timeout_sec=5.0),
            return_exceptions=True,
        )
        # node-0 connects outbound to node-1 → auth fails fast.
        assert isinstance(results[0], PeerAuthError)
        # node-1 waits for an inbound connection that never authenticates.
        assert isinstance(results[1], MeshFormationError)

    @pytest.mark.asyncio
    async def test_missing_peer_times_out_with_remediation(self):
        specs = _make_specs(2)
        with pytest.raises(MeshFormationError, match="node-1"):
            await form_mesh(
                "node-0",
                specs,
                config=CONFIG,
                bind_host="127.0.0.1",
                rendezvous_timeout_sec=1.0,
            )

    @pytest.mark.asyncio
    async def test_missing_inbound_peer_times_out(self):
        """The accepting side (higher node_id) also fails with the
        missing peer named."""
        specs = _make_specs(2)
        with pytest.raises(MeshFormationError, match="node-0"):
            await form_mesh(
                "node-1",
                specs,
                config=CONFIG,
                bind_host="127.0.0.1",
                rendezvous_timeout_sec=1.0,
            )
