"""Tests for basic ThermalMonitor configuration safety."""

from __future__ import annotations

import pytest

from macfleet.engines.base import ThermalPressure
from macfleet.monitoring.thermal import ThermalMonitor, ThermalState, estimate_safe_batch_size


@pytest.mark.parametrize("interval", [0, -1, True, float("nan"), float("inf")])
def test_thermal_monitor_rejects_invalid_poll_interval(interval):
    with pytest.raises(ValueError, match="poll_interval_sec"):
        ThermalMonitor(poll_interval_sec=interval)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_safe_batch_estimate_rejects_invalid_input(batch_size):
    with pytest.raises(ValueError, match="current_batch_size"):
        estimate_safe_batch_size(batch_size, ThermalState(pressure=ThermalPressure.NOMINAL))
