"""Tests for health monitoring."""

from __future__ import annotations

import pytest

from macfleet.engines.base import ThermalPressure
from macfleet.monitoring.health import (
    HealthMonitor,
    HealthStatus,
    MemoryInfo,
    NodeHealth,
    _analyze_loss_trend,
)
from macfleet.monitoring.thermal import ThermalState

# ------------------------------------------------------------------ #
# MemoryInfo                                                         #
# ------------------------------------------------------------------ #


class TestMemoryInfo:
    def test_usage_pct(self):
        mem = MemoryInfo(total_gb=16.0, used_gb=8.0, available_gb=8.0)
        assert mem.usage_pct == 50.0

    def test_zero_total(self):
        mem = MemoryInfo(total_gb=0.0)
        assert mem.usage_pct == 0.0


# ------------------------------------------------------------------ #
# NodeHealth                                                         #
# ------------------------------------------------------------------ #


class TestNodeHealth:
    def test_healthy_score(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.NOMINAL),
            memory=MemoryInfo(total_gb=16.0, used_gb=8.0, available_gb=8.0),
            is_plugged_in=True,
        )
        assert health.health_score == 1.0

    def test_thermal_throttling_penalty(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.SERIOUS),
            memory=MemoryInfo(total_gb=16.0, used_gb=4.0, available_gb=12.0),
        )
        assert health.health_score == pytest.approx(0.7, abs=0.01)

    def test_critical_thermal(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.CRITICAL),
        )
        assert health.health_score == pytest.approx(0.3, abs=0.01)

    def test_high_memory_penalty(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.NOMINAL),
            memory=MemoryInfo(total_gb=16.0, used_gb=15.0, available_gb=1.0),
        )
        # 93.75% usage → penalty
        assert health.health_score < 0.8

    def test_low_battery_penalty(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.NOMINAL),
            battery_pct=5.0,
            is_plugged_in=False,
        )
        assert health.health_score == pytest.approx(0.2, abs=0.01)

    def test_plugged_in_no_penalty(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.NOMINAL),
            battery_pct=5.0,
            is_plugged_in=True,
        )
        assert health.health_score == 1.0

    def test_connection_failures_penalty(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.NOMINAL),
            connection_failures=5,
        )
        assert health.health_score == pytest.approx(0.5, abs=0.01)

    def test_warnings_thermal(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.SERIOUS),
        )
        assert any("Thermal" in w for w in health.warnings)

    def test_warnings_memory(self):
        health = NodeHealth(
            memory=MemoryInfo(total_gb=16.0, used_gb=14.0, available_gb=2.0),
        )
        assert any("memory" in w.lower() for w in health.warnings)

    def test_warnings_diverging(self):
        health = NodeHealth(loss_trend="diverging")
        assert any("diverging" in w.lower() for w in health.warnings)

    def test_no_warnings_healthy(self):
        health = NodeHealth(
            thermal=ThermalState(pressure=ThermalPressure.NOMINAL),
            memory=MemoryInfo(total_gb=16.0, used_gb=4.0, available_gb=12.0),
        )
        assert len(health.warnings) == 0


# ------------------------------------------------------------------ #
# Loss trend analysis                                                #
# ------------------------------------------------------------------ #


class TestLossTrend:
    def test_decreasing(self):
        assert _analyze_loss_trend([1.0, 0.8, 0.6, 0.4, 0.2]) == "decreasing"

    def test_stable(self):
        assert _analyze_loss_trend([0.5, 0.5, 0.5, 0.5]) == "stable"

    def test_increasing(self):
        assert _analyze_loss_trend([0.1, 0.3, 0.5, 0.7, 0.9]) == "increasing"

    def test_diverging(self):
        assert _analyze_loss_trend([0.1, 0.5, 2.0, 10.0, 100.0]) == "diverging"

    def test_nan_diverging(self):
        assert _analyze_loss_trend([0.1, 0.2, float("nan")]) == "diverging"

    def test_too_short(self):
        assert _analyze_loss_trend([0.5]) == "stable"

    def test_empty(self):
        assert _analyze_loss_trend([]) == "stable"


# ------------------------------------------------------------------ #
# HealthMonitor integration                                          #
# ------------------------------------------------------------------ #


class TestHealthMonitor:
    def test_basic_check(self):
        monitor = HealthMonitor(node_id="test-node")
        health = monitor.check()

        assert health.node_id == "test-node"
        assert health.timestamp > 0
        assert health.status in HealthStatus

    def test_check_with_metrics(self):
        monitor = HealthMonitor(node_id="test-node")
        health = monitor.check(
            throughput=100.0,
            sync_time=0.05,
            loss_history=[1.0, 0.8, 0.6, 0.4, 0.2],
        )

        assert health.throughput_samples_sec == 100.0
        assert health.avg_sync_time_sec == 0.05
        assert health.loss_trend == "decreasing"

    def test_history_tracking(self):
        monitor = HealthMonitor(node_id="test-node")
        monitor.check()
        monitor.check()
        monitor.check()

        assert len(monitor.health_history) == 3
        assert monitor.last_health is not None
