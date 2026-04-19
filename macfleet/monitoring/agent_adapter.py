"""Bridge between a live PoolAgent and the Dashboard TUI.

v2.2 PR 11 (E2 from docs/designs/v3-cathedral.md): the Dashboard class
already exists and knows how to render NodeHealth objects. This module
converts the agent's ClusterRegistry records into NodeHealth snapshots
that can be fed to `Dashboard.update_nodes()`.

Why a separate adapter? The registry and the dashboard speak different
vocabularies on purpose:
    - Registry tracks persistent cluster state (who joined, who's alive,
      hardware specs, coordinator election)
    - NodeHealth captures a point-in-time health snapshot (thermal
      pressure NOW, memory pressure NOW, loss trend, warnings)

The adapter is the place that does the "compute current snapshot from
persistent state + ambient readings" work without either side caring
about the other.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from macfleet.monitoring.health import HealthStatus, MemoryInfo, NodeHealth
from macfleet.monitoring.thermal import get_thermal_state

if TYPE_CHECKING:
    from macfleet.pool.agent import PoolAgent


def build_node_health_for_self(
    agent: PoolAgent,
    loss_trend: str = "stable",
    throughput_samples_sec: float = 0.0,
    avg_sync_time_sec: float = 0.0,
    connection_failures: int = 0,
) -> NodeHealth:
    """Snapshot the LOCAL agent's health.

    Reads thermal + memory state directly from the OS. Training metrics
    come from the caller (training loop is the only place that knows
    the current loss / throughput / sync time).

    Args:
        agent: The live PoolAgent owned by this Pool.
        loss_trend: "decreasing", "stable", "increasing", or "diverging".
            Passed through by the training loop.
        throughput_samples_sec: Current samples/sec across all workers.
        avg_sync_time_sec: Rolling-average allreduce latency.
        connection_failures: Number of peer connections that have failed
            since startup (used by warnings + health score).

    Returns:
        NodeHealth ready to feed to Dashboard.update_nodes().
    """
    from macfleet.monitoring.health import get_memory_info

    memory: Optional[MemoryInfo]
    try:
        memory = get_memory_info()
    except Exception:
        memory = None

    try:
        thermal = get_thermal_state()
    except Exception:
        thermal = None

    health = NodeHealth(
        node_id=agent.node_id,
        timestamp=time.time(),
        status=HealthStatus.UNKNOWN,  # reclassified below based on score
        thermal=thermal,
        memory=memory,
        loss_trend=loss_trend,
        throughput_samples_sec=throughput_samples_sec,
        avg_sync_time_sec=avg_sync_time_sec,
        connection_failures=connection_failures,
    )
    health.status = classify_health(health.health_score)
    return health


def build_node_health_for_peers(agent: PoolAgent) -> list[NodeHealth]:
    """Snapshot peer nodes using only what's observable via the registry.

    v2.2 note: we don't yet have a gossip channel for peers to report
    their own thermal/memory state back to the coordinator, so the
    returned NodeHealth for remote peers is intentionally sparse — just
    what the registry already knows (node_id, thermal_pressure from
    heartbeat gossip via HardwareProfile.thermal_pressure).

    A later PR (Issue 12: richer heartbeat gossip) will add per-peer
    memory pressure, battery state, and throughput to the wire so this
    adapter can populate them here.
    """
    out: list[NodeHealth] = []
    if agent.registry is None:
        return out

    for record in agent.registry.alive_nodes:
        if record.node_id == agent.node_id:
            continue  # self is handled by build_node_health_for_self
        hw = record.hardware
        health = NodeHealth(
            node_id=record.node_id,
            timestamp=time.time(),
            status=HealthStatus.HEALTHY,  # alive in registry → assume healthy
            thermal=None,  # No gossip channel yet; see docstring
            memory=None,
            loss_trend="stable",
            throughput_samples_sec=0.0,
            avg_sync_time_sec=0.0,
        )
        # Stub thermal from the hardware profile (set at join time)
        if hw.thermal_pressure:
            from macfleet.monitoring.thermal import ThermalState
            health.thermal = ThermalState(pressure=hw.thermal_pressure)
        out.append(health)
    return out


def classify_health(score: float) -> HealthStatus:
    """Map a 0.0..1.0 health score to a HealthStatus bucket.

    Thresholds match the Dashboard panels' coloring:
        >= 0.8 → HEALTHY (green)
        >= 0.5 → DEGRADED (yellow)
        else   → UNHEALTHY (red)
    """
    if score >= 0.8:
        return HealthStatus.HEALTHY
    if score >= 0.5:
        return HealthStatus.DEGRADED
    return HealthStatus.UNHEALTHY


def snapshot_all(
    agent: PoolAgent,
    loss_trend: str = "stable",
    throughput_samples_sec: float = 0.0,
    avg_sync_time_sec: float = 0.0,
    connection_failures: int = 0,
) -> list[NodeHealth]:
    """One-shot: build the full [self, peer1, peer2, ...] NodeHealth list.

    This is the one call a Pool.dashboard() loop invokes every refresh
    tick. Self goes first so it's always top of the cluster table.
    """
    nodes = [
        build_node_health_for_self(
            agent,
            loss_trend=loss_trend,
            throughput_samples_sec=throughput_samples_sec,
            avg_sync_time_sec=avg_sync_time_sec,
            connection_failures=connection_failures,
        )
    ]
    nodes.extend(build_node_health_for_peers(agent))
    return nodes
