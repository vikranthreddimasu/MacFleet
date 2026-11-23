"""Node health monitoring for MacFleet.

Aggregates thermal state, memory pressure, network quality, and
training performance into a single health score per node.
"""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from macfleet.monitoring.thermal import ThermalState, get_thermal_state


class HealthStatus(Enum):
    """Overall node health classification."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"     # Still functional but performance reduced
    UNHEALTHY = "unhealthy"   # Should be excluded from training
    UNKNOWN = "unknown"


def classify_health(score: float) -> HealthStatus:
    """Map a 0.0..1.0 health score to a HealthStatus bucket.

    Single source of truth for score → status classification, used by both
    HealthMonitor.check() and the agent_adapter snapshot path so the same
    node is never classified two different ways.

        >= 0.8 → HEALTHY (green)
        >= 0.5 → DEGRADED (yellow)
        else   → UNHEALTHY (red)
    """
    if score >= 0.8:
        return HealthStatus.HEALTHY
    if score >= 0.5:
        return HealthStatus.DEGRADED
    return HealthStatus.UNHEALTHY


@dataclass
class MemoryInfo:
    """System memory information."""
    total_gb: float = 0.0
    used_gb: float = 0.0
    available_gb: float = 0.0

    @property
    def usage_pct(self) -> float:
        if (
            not math.isfinite(self.total_gb)
            or not math.isfinite(self.used_gb)
            or self.total_gb <= 0
        ):
            return 0.0
        return max(0.0, min(100.0, (self.used_gb / self.total_gb) * 100.0))


@dataclass
class NodeHealth:
    """Complete health snapshot for a single node."""
    node_id: str = ""
    timestamp: float = 0.0
    status: HealthStatus = HealthStatus.UNKNOWN

    # Components
    thermal: Optional[ThermalState] = None
    memory: Optional[MemoryInfo] = None

    # Training metrics
    loss_trend: str = "stable"        # "decreasing", "stable", "increasing", "diverging"
    throughput_samples_sec: float = 0.0
    avg_sync_time_sec: float = 0.0

    # Network
    peer_latency_ms: float = 0.0
    connection_failures: int = 0

    # System
    battery_pct: Optional[float] = None
    is_plugged_in: Optional[bool] = None
    uptime_sec: float = 0.0

    @property
    def health_score(self) -> float:
        """Composite health score 0.0 (dead) to 1.0 (perfect)."""
        score = 1.0

        # Thermal penalty
        if self.thermal:
            score *= self.thermal.workload_multiplier

        # Memory pressure penalty
        if self.memory and self.memory.usage_pct > 80:
            score *= max(0.3, 1.0 - (self.memory.usage_pct - 80) / 40)

        # Battery penalty
        if self.battery_pct is not None and not self.is_plugged_in:
            if self.battery_pct < 10:
                score *= 0.2
            elif self.battery_pct < 20:
                score *= 0.5

        # Connection failure penalty
        if self.connection_failures > 0:
            score *= max(0.3, 1.0 - self.connection_failures * 0.1)

        return max(0.0, min(1.0, score))

    @property
    def warnings(self) -> list[str]:
        """List of active warnings."""
        warns = []
        if self.thermal and self.thermal.is_throttling:
            warns.append(f"Thermal throttling: {self.thermal.pressure.value}")
        if self.memory and self.memory.usage_pct > 85:
            warns.append(f"High memory usage: {self.memory.usage_pct:.0f}%")
        if self.battery_pct is not None and not self.is_plugged_in:
            if self.battery_pct < 20:
                warns.append(f"Low battery: {self.battery_pct:.0f}%")
        if self.loss_trend == "diverging":
            warns.append("Loss is diverging")
        if self.connection_failures > 3:
            warns.append(f"Connection failures: {self.connection_failures}")
        return warns


def get_memory_info() -> MemoryInfo:
    """Estimate used memory as physical RAM less free and inactive pages."""
    try:
        physical = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True, timeout=2,
        )
        total = int(physical.stdout.strip()) / (1024**3)
        if total <= 0:
            return MemoryInfo()
    except (subprocess.SubprocessError, OSError, ValueError):
        return MemoryInfo()

    try:
        stats = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True, timeout=2)
        header, *rows = stats.stdout.splitlines()
        page_size = int(header.split("page size of ")[1].split()[0])
        counts = dict(row.split(":", 1) for row in rows if ":" in row)
        reclaimable = sum(int(counts[key].strip().rstrip(".")) for key in ("Pages free", "Pages inactive"))
        if page_size <= 0 or reclaimable < 0:
            return MemoryInfo(total_gb=total)
        available = min(total, reclaimable * page_size / (1024**3))
        return MemoryInfo(total_gb=total, used_gb=total - available, available_gb=available)
    except (subprocess.SubprocessError, OSError, ValueError, IndexError, KeyError):
        return MemoryInfo(total_gb=total)


def get_battery_info() -> tuple[Optional[float], Optional[bool]]:
    """Get battery percentage and charging status.

    Returns:
        (battery_pct, is_plugged_in) or (None, None) for desktops.
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            output = result.stdout
            is_plugged = "AC Power" in output

            for line in output.split("\n"):
                if "%" in line and "InternalBattery" in line:
                    try:
                        pct_str = line.split("%")[0].split()[-1]
                        return float(pct_str), is_plugged
                    except (ValueError, IndexError):
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, UnicodeDecodeError):
        pass

    return None, None


class HealthMonitor:
    """Monitors node health and produces periodic snapshots.

    Integrates thermal, memory, battery, and training metrics
    into a unified health assessment.
    """

    def __init__(self, node_id: str = ""):
        self.node_id = node_id
        self._history: list[NodeHealth] = []
        self._max_history = 100

    def check(
        self,
        throughput: float = 0.0,
        sync_time: float = 0.0,
        loss_history: Optional[list[float]] = None,
        connection_failures: int = 0,
        peer_latency_ms: float = 0.0,
    ) -> NodeHealth:
        """Perform a full health check.

        Args:
            throughput: Current training throughput (samples/sec).
            sync_time: Average gradient sync time (sec).
            loss_history: Recent loss values for trend analysis.
            connection_failures: Number of recent connection failures.
            peer_latency_ms: Average peer latency.

        Returns:
            NodeHealth snapshot.
        """
        thermal = get_thermal_state()
        memory = get_memory_info()
        battery_pct, is_plugged = get_battery_info()
        loss_trend = _analyze_loss_trend(loss_history) if loss_history else "stable"

        health = NodeHealth(
            node_id=self.node_id,
            timestamp=time.time(),
            thermal=thermal,
            memory=memory,
            loss_trend=loss_trend,
            throughput_samples_sec=throughput,
            avg_sync_time_sec=sync_time,
            peer_latency_ms=peer_latency_ms,
            connection_failures=connection_failures,
            battery_pct=battery_pct,
            is_plugged_in=is_plugged,
        )

        # Classify status (shared thresholds — see classify_health)
        health.status = classify_health(health.health_score)

        self._history.append(health)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return health

    @property
    def last_health(self) -> Optional[NodeHealth]:
        return self._history[-1] if self._history else None

    @property
    def health_history(self) -> list[NodeHealth]:
        return list(self._history)


def _analyze_loss_trend(losses: list[float], window: int = 5) -> str:
    """Analyze loss trend over recent history.

    Returns: "decreasing", "stable", "increasing", or "diverging".
    """
    if len(losses) < 2:
        return "stable"

    recent = losses[-window:]
    if len(recent) < 2:
        return "stable"

    # Check for NaN/inf
    if any(not (x == x) or abs(x) > 1e10 for x in recent):
        return "diverging"

    # Linear trend
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    avg_diff = sum(diffs) / len(diffs)

    if avg_diff < -0.01:
        return "decreasing"
    elif avg_diff > 0.1:
        return "diverging" if avg_diff > 1.0 else "increasing"
    return "stable"
