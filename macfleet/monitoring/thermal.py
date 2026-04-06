"""Thermal monitoring for macOS on Apple Silicon.

Ported from MacFleet v1. Monitors thermal state via pmset/ioreg/sysctl
to detect throttling and trigger workload rebalancing.
"""

import asyncio
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional

from macfleet.engines.base import ThermalPressure


@dataclass
class ThermalState:
    """Current thermal state of the system."""
    pressure: ThermalPressure
    cpu_temp_celsius: Optional[float] = None
    gpu_temp_celsius: Optional[float] = None
    fan_speed_rpm: Optional[int] = None
    timestamp: float = 0.0

    @property
    def is_throttling(self) -> bool:
        return self.pressure in (ThermalPressure.SERIOUS, ThermalPressure.CRITICAL)

    @property
    def workload_multiplier(self) -> float:
        return self.pressure.workload_multiplier


def get_thermal_state() -> ThermalState:
    """Get the current thermal state from macOS.

    Uses multiple detection methods:
    1. pmset -g therm (always available, no sudo)
    2. ioreg for battery temperature
    3. sysctl for CPU thermal level
    """
    pressure = ThermalPressure.NOMINAL
    cpu_temp = None
    gpu_temp = None
    fan_speed = None

    # Method 1: pmset (no sudo required)
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "cpu_speed_limit" in output:
                for line in output.split("\n"):
                    if "cpu_speed_limit" in line:
                        try:
                            limit = int(line.split()[-1])
                            if limit < 50:
                                pressure = ThermalPressure.CRITICAL
                            elif limit < 70:
                                pressure = ThermalPressure.SERIOUS
                            elif limit < 90:
                                pressure = ThermalPressure.FAIR
                        except (ValueError, IndexError):
                            pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Method 2: IOKit battery temperature (no sudo)
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AppleSmartBattery"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Temperature" in line:
                    try:
                        temp_val = int(line.split("=")[-1].strip())
                        cpu_temp = temp_val / 100.0
                    except (ValueError, IndexError):
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Method 3: sysctl CPU thermal level
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            try:
                level = int(result.stdout.strip())
                if level >= 100:
                    pressure = ThermalPressure.CRITICAL
                elif level >= 70:
                    pressure = ThermalPressure.SERIOUS
                elif level >= 30:
                    pressure = ThermalPressure.FAIR
            except ValueError:
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return ThermalState(
        pressure=pressure,
        cpu_temp_celsius=cpu_temp,
        gpu_temp_celsius=gpu_temp,
        fan_speed_rpm=fan_speed,
        timestamp=time.time(),
    )


def get_thermal_string() -> str:
    """Get thermal state as a simple string."""
    return get_thermal_state().pressure.value


class ThermalMonitor:
    """Monitor thermal state and trigger actions on throttling."""

    def __init__(
        self,
        poll_interval_sec: float = 5.0,
        on_throttle: Optional[Callable[[ThermalState], None]] = None,
        on_recover: Optional[Callable[[ThermalState], None]] = None,
    ):
        self._interval = poll_interval_sec
        self._on_throttle = on_throttle
        self._on_recover = on_recover
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_state: Optional[ThermalState] = None
        self._was_throttling = False

    @property
    def current_state(self) -> Optional[ThermalState]:
        return self._last_state

    @property
    def is_throttling(self) -> bool:
        if self._last_state is None:
            return False
        return self._last_state.is_throttling

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        while self._running:
            self._check_thermal()
            await asyncio.sleep(self._interval)

    def _check_thermal(self) -> None:
        state = get_thermal_state()
        self._last_state = state
        is_throttling = state.is_throttling

        if is_throttling and not self._was_throttling:
            if self._on_throttle:
                self._on_throttle(state)
        elif not is_throttling and self._was_throttling:
            if self._on_recover:
                self._on_recover(state)

        self._was_throttling = is_throttling

    def get_state(self) -> ThermalState:
        state = get_thermal_state()
        self._last_state = state
        return state


def estimate_safe_batch_size(current_batch_size: int, thermal_state: ThermalState) -> int:
    """Estimate a safe batch size given thermal state."""
    multiplier = thermal_state.workload_multiplier
    return max(1, int(current_batch_size * multiplier))


def thermal_state_to_string(state: ThermalState) -> str:
    """Format thermal state for display."""
    parts = [f"Pressure: {state.pressure.value}"]
    if state.cpu_temp_celsius is not None:
        parts.append(f"CPU: {state.cpu_temp_celsius:.1f}C")
    if state.gpu_temp_celsius is not None:
        parts.append(f"GPU: {state.gpu_temp_celsius:.1f}C")
    if state.fan_speed_rpm is not None:
        parts.append(f"Fan: {state.fan_speed_rpm} RPM")
    return " | ".join(parts)
