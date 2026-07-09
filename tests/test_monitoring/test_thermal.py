"""Tests for basic ThermalMonitor configuration safety."""

from __future__ import annotations

import pytest

from macfleet.monitoring.thermal import ThermalMonitor


@pytest.mark.parametrize("interval", [0, -1, True, float("nan"), float("inf")])
def test_thermal_monitor_rejects_invalid_poll_interval(interval):
    with pytest.raises(ValueError, match="poll_interval_sec"):
        ThermalMonitor(poll_interval_sec=interval)
