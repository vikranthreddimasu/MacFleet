"""Training mesh formation: rendezvous N nodes into a CollectiveGroup.

Framework-agnostic (numpy/asyncio only — no torch, no mlx). This is the
glue between the pool layer (which knows WHO is in the fleet: node ids,
IPs, data ports) and the comm layer (which knows HOW to move arrays
between peers).

Every node runs the same code (SPMD). Ranks are derived from the sorted
node-id order, so any two nodes that agree on the member list agree on
the rank assignment without any extra coordination round:

    nodes = [NodeSpec("mac-b-1f2e", "192.168.1.7", 50052),
             NodeSpec("mac-a-9c41", "192.168.1.5", 50052)]
    mesh = await form_mesh("mac-a-9c41", nodes, security=sec)
    averaged = await mesh.group.allreduce(grads, op="mean")
    ...
    await mesh.close()

Connection convention: each node CONNECTS to every peer whose node_id
sorts after its own, and ACCEPTS connections from peers that sort
before it. One TCP connection per pair, used bidirectionally — exactly
what CollectiveGroup expects.

Rendezvous is retry-based: peers reach form_mesh() at different times
(the user starts the script on each Mac by hand), so outbound connects
retry until the deadline. Auth failures (wrong fleet token) fail fast —
retrying cannot fix a wrong token, and hammering the peer only trips
its rate limiter.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from macfleet.comm.collectives import CollectiveGroup
from macfleet.comm.transport import (
    HardwareExchange,
    PeerAuthError,
    PeerTransport,
    TransportConfig,
)
from macfleet.security.auth import SecurityConfig

# How long to wait between outbound connect attempts / inbound polls.
_CONNECT_RETRY_INTERVAL_SEC = 0.25
_INBOUND_POLL_INTERVAL_SEC = 0.1


class MeshFormationError(RuntimeError):
    """Raised when the training mesh cannot be formed before the deadline.

    The message names the peers that never showed up so the user can
    check those specific Macs instead of guessing.
    """


@dataclass(frozen=True)
class NodeSpec:
    """Identity + address of one training node.

    `data_port` is the training-transport port (PoolAgent.data_port /
    NodeRecord.data_port), NOT the heartbeat port.
    """

    node_id: str
    ip_address: str
    data_port: int


@dataclass
class Mesh:
    """A formed training mesh: live transport + collective group."""

    transport: PeerTransport
    group: CollectiveGroup
    rank: int
    world_size: int

    async def close(self) -> None:
        """Close all peer connections and stop the transport server."""
        await self.transport.disconnect_all()


def derive_ranks(nodes: Sequence[NodeSpec]) -> dict[str, int]:
    """Assign ranks by lexicographic node_id order.

    Deliberately NOT compute_score order (unlike ClusterRegistry.get_ranks):
    different nodes can transiently disagree about a peer's compute_score
    (e.g. one side still holds a zero-score mDNS placeholder while the
    other has the real HW from a gossip round). Disagreeing on ranks
    deadlocks the rendezvous. node_id is immutable and identical in every
    node's registry, so id-order is the only assignment every member is
    guaranteed to compute identically.
    """
    ordered = sorted(spec.node_id for spec in nodes)
    return {node_id: rank for rank, node_id in enumerate(ordered)}


async def form_mesh(
    local_id: str,
    nodes: Sequence[NodeSpec],
    *,
    security: Optional[SecurityConfig] = None,
    local_hw: Optional[HardwareExchange] = None,
    config: Optional[TransportConfig] = None,
    bind_host: str = "0.0.0.0",
    rendezvous_timeout_sec: float = 60.0,
) -> Mesh:
    """Form a full training mesh among `nodes` and return it.

    Args:
        local_id: This node's node_id. Must appear in `nodes`.
        nodes: Every member of the training group, including self.
            All nodes must pass the same member set (they snapshot the
            same registry); a disagreement surfaces as a rendezvous
            timeout naming the missing peers.
        security: Fleet SecurityConfig. With a fleet key set, every pair
            performs the HMAC challenge-response handshake over TLS.
        local_hw: Hardware profile advertised in the authenticated
            handshake (optional; affects election metadata only).
        config: TransportConfig override (timeouts, buffer sizes).
        bind_host: Address the local transport server binds to.
        rendezvous_timeout_sec: Total budget for all peers to show up.

    Returns:
        Mesh with a started PeerTransport and a ready CollectiveGroup.

    Raises:
        ValueError: local_id missing from nodes, or duplicate node_ids.
        PeerAuthError: a peer presented the wrong fleet token (fail fast).
        MeshFormationError: peers did not connect before the deadline.
    """
    by_id = {spec.node_id: spec for spec in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate node_ids in mesh spec")
    if local_id not in by_id:
        raise ValueError(f"local_id {local_id!r} not present in mesh spec")

    ranks = derive_ranks(nodes)
    rank = ranks[local_id]
    world_size = len(nodes)
    local_spec = by_id[local_id]

    transport = PeerTransport(
        local_id=local_id,
        config=config,
        security=security,
        local_hw=local_hw,
    )

    if world_size == 1:
        # Degenerate mesh: no server, no peers. CollectiveGroup handles
        # world_size=1 as a no-op for every collective.
        return Mesh(
            transport=transport,
            group=CollectiveGroup(
                rank=0, world_size=1, transport=transport, rank_to_peer={},
            ),
            rank=0,
            world_size=1,
        )

    await transport.start_server(bind_host, local_spec.data_port)
    deadline = time.monotonic() + rendezvous_timeout_sec

    outbound = [spec for spec in nodes if spec.node_id > local_id]
    inbound_ids = [spec.node_id for spec in nodes if spec.node_id < local_id]

    async def _connect_with_retry(spec: NodeSpec) -> None:
        while True:
            try:
                await transport.connect(spec.node_id, spec.ip_address, spec.data_port)
                return
            except PeerAuthError:
                # Wrong token can't be fixed by retrying — and each retry
                # counts against the peer's rate limiter.
                raise
            except (ConnectionError, OSError, asyncio.TimeoutError):
                if time.monotonic() >= deadline:
                    raise MeshFormationError(
                        f"Could not connect to peer {spec.node_id} at "
                        f"{spec.ip_address}:{spec.data_port} within "
                        f"{rendezvous_timeout_sec:.0f}s. Check that the peer is "
                        f"running the same training script, and that "
                        f"'macfleet status' shows it as alive."
                    ) from None
                await asyncio.sleep(_CONNECT_RETRY_INTERVAL_SEC)

    try:
        if outbound:
            await asyncio.gather(*(_connect_with_retry(s) for s in outbound))

        # Wait for inbound connections from lower-id peers. The transport
        # registers each connection under the peer_id it authenticated as.
        missing = [
            pid for pid in inbound_ids if transport.get_connection(pid) is None
        ]
        while missing:
            if time.monotonic() >= deadline:
                raise MeshFormationError(
                    f"Peers never connected within {rendezvous_timeout_sec:.0f}s: "
                    f"{', '.join(missing)}. Each Mac must run the same training "
                    f"script; check 'macfleet status' on the missing peers."
                )
            await asyncio.sleep(_INBOUND_POLL_INTERVAL_SEC)
            missing = [
                pid for pid in inbound_ids if transport.get_connection(pid) is None
            ]
    except BaseException:
        # Don't leak the half-formed mesh (server socket + partial conns).
        await transport.disconnect_all()
        raise

    rank_to_peer = {
        ranks[node_id]: node_id for node_id in by_id if node_id != local_id
    }
    group = CollectiveGroup(
        rank=rank,
        world_size=world_size,
        transport=transport,
        rank_to_peer=rank_to_peer,
    )
    return Mesh(transport=transport, group=group, rank=rank, world_size=world_size)
