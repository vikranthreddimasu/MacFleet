"""Gossip-based peer liveness detection.

Each node periodically pings a random subset of peers. If a node
misses consecutive rounds it's marked suspected, then failed.
Failure triggers scheduler re-computation and collective reconfiguration.
"""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from macfleet.security.auth import (
    SecurityConfig,
    create_client_ssl_context,
    sign_heartbeat,
    verify_heartbeat,
)

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Liveness status of a pool member."""
    ALIVE = "alive"
    SUSPECTED = "suspected"
    FAILED = "failed"
    LEFT = "left"  # graceful departure


@dataclass
class PeerState:
    """Tracked state for a single peer."""
    node_id: str
    ip_address: str
    port: int
    status: NodeStatus = NodeStatus.ALIVE
    last_seen: float = 0.0
    missed_rounds: int = 0
    compute_score: float = 0.0

    @property
    def is_alive(self) -> bool:
        return self.status == NodeStatus.ALIVE


@dataclass
class HeartbeatConfig:
    """Configuration for the gossip heartbeat protocol."""
    interval_sec: float = 1.0       # How often to ping
    suspicion_rounds: int = 3       # Missed rounds before suspected
    failure_timeout_sec: float = 10.0  # Suspected duration before failed
    fanout: int = 3                 # Number of peers to ping each round


class GossipHeartbeat:
    """Gossip-based failure detector for pool members.

    Each round, this node pings `fanout` random peers. If a peer
    doesn't respond for `suspicion_rounds`, it becomes suspected.
    If suspected for `failure_timeout_sec`, it's declared failed.
    """

    def __init__(
        self,
        node_id: str,
        config: Optional[HeartbeatConfig] = None,
        security: Optional[SecurityConfig] = None,
        on_suspected: Optional[Callable[[str], None]] = None,
        on_failed: Optional[Callable[[str], None]] = None,
        on_recovered: Optional[Callable[[str], None]] = None,
    ):
        self.node_id = node_id
        self.config = config or HeartbeatConfig()
        self._security = security or SecurityConfig()
        self._on_suspected = on_suspected
        self._on_failed = on_failed
        self._on_recovered = on_recovered

        self._peers: dict[str, PeerState] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ping_handler: Optional[Callable] = None

    @property
    def peers(self) -> dict[str, PeerState]:
        return dict(self._peers)

    @property
    def alive_peers(self) -> list[PeerState]:
        return [p for p in self._peers.values() if p.is_alive]

    @property
    def alive_count(self) -> int:
        return len(self.alive_peers)

    def add_peer(self, node_id: str, ip_address: str, port: int, compute_score: float = 0.0) -> None:
        """Add a peer to track."""
        if node_id == self.node_id:
            return  # Don't track self
        self._peers[node_id] = PeerState(
            node_id=node_id,
            ip_address=ip_address,
            port=port,
            status=NodeStatus.ALIVE,
            last_seen=time.monotonic(),
            compute_score=compute_score,
        )

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer (graceful departure)."""
        if node_id in self._peers:
            self._peers[node_id].status = NodeStatus.LEFT

    def record_heartbeat(self, node_id: str) -> None:
        """Record that we heard from a peer (they're alive)."""
        if node_id in self._peers:
            peer = self._peers[node_id]
            was_suspected = peer.status == NodeStatus.SUSPECTED
            peer.status = NodeStatus.ALIVE
            peer.last_seen = time.monotonic()
            peer.missed_rounds = 0
            if was_suspected and self._on_recovered:
                self._on_recovered(node_id)

    async def start(self) -> None:
        """Start the gossip heartbeat loop."""
        self._running = True
        self._task = asyncio.create_task(self._gossip_loop())

    async def stop(self) -> None:
        """Stop the gossip heartbeat loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _gossip_loop(self) -> None:
        """Main gossip loop: ping random peers, update statuses."""
        while self._running:
            await self._gossip_round()
            await asyncio.sleep(self.config.interval_sec)

    async def _gossip_round(self) -> None:
        """Single gossip round: select peers, ping, update status."""
        now = time.monotonic()
        candidates = [
            p for p in self._peers.values()
            if p.status in (NodeStatus.ALIVE, NodeStatus.SUSPECTED)
        ]

        if not candidates:
            return

        # Select random subset to ping
        targets = random.sample(candidates, min(self.config.fanout, len(candidates)))

        for peer in targets:
            alive = await self._ping_peer(peer)
            if alive:
                self.record_heartbeat(peer.node_id)
            else:
                peer.missed_rounds += 1

                if (
                    peer.status == NodeStatus.ALIVE
                    and peer.missed_rounds >= self.config.suspicion_rounds
                ):
                    peer.status = NodeStatus.SUSPECTED
                    if self._on_suspected:
                        self._on_suspected(peer.node_id)

                if (
                    peer.status == NodeStatus.SUSPECTED
                    and (now - peer.last_seen) > self.config.failure_timeout_sec
                ):
                    peer.status = NodeStatus.FAILED
                    if self._on_failed:
                        self._on_failed(peer.node_id)

    async def _ping_peer(self, peer: PeerState) -> bool:
        """Send a heartbeat ping to a peer. Returns True if peer responds."""
        fleet_key = self._security.fleet_key
        ssl_ctx = create_client_ssl_context() if self._security.tls else None
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.ip_address, peer.port, ssl=ssl_ctx),
                timeout=2.0,
            )

            if fleet_key:
                # Authenticated heartbeat: APING {node_id} {nonce_hex} {hmac_hex}\n
                nonce = secrets.token_bytes(16)
                sig = sign_heartbeat(fleet_key, self.node_id, nonce)
                msg = f"APING {self.node_id} {nonce.hex()} {sig.hex()}\n".encode()
                writer.write(msg)
                await writer.drain()

                response = await asyncio.wait_for(reader.readline(), timeout=2.0)
                writer.close()
                await writer.wait_closed()

                if not response.startswith(b"APONG"):
                    return False
                # Verify APONG signature
                parts = response.decode().strip().split(" ")
                if len(parts) != 4:
                    return False
                _, resp_node_id, resp_nonce_hex, resp_sig_hex = parts
                resp_nonce = bytes.fromhex(resp_nonce_hex)
                resp_sig = bytes.fromhex(resp_sig_hex)
                return verify_heartbeat(fleet_key, resp_node_id, resp_nonce, resp_sig)
            else:
                # Open heartbeat (backward compatible)
                msg = f"PING {self.node_id}\n".encode()
                writer.write(msg)
                await writer.drain()

                response = await asyncio.wait_for(reader.readline(), timeout=2.0)
                writer.close()
                await writer.wait_closed()
                return response.startswith(b"PONG")
        except (OSError, asyncio.TimeoutError, ConnectionRefusedError, ValueError):
            return False

    def get_status_summary(self) -> dict[str, int]:
        """Get count of peers by status."""
        summary: dict[str, int] = {}
        for peer in self._peers.values():
            key = peer.status.value
            summary[key] = summary.get(key, 0) + 1
        return summary
