"""Thermal monitoring for macOS on Apple Silicon.

Monitors thermal state to detect throttling and trigger
workload rebalancing when a node gets too hot.
"""

import asyncio
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from rich.console import Console

console = Console()


class ThermalPressure(Enum):
    """Thermal pressure levels from macOS."""
    NOMINAL = "nominal"      # Normal operation
    FAIR = "fair"            # Slightly elevated
    SERIOUS = "serious"      # Throttling likely
    CRITICAL = "critical"    # Heavy throttling


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
        """Check if system is likely throttling."""
        return self.pressure in (ThermalPressure.SERIOUS, ThermalPressure.CRITICAL)

    @property
    def workload_multiplier(self) -> float:
        """Suggested workload multiplier based on thermal state.

        Returns a value < 1.0 if workload should be reduced.
        """
        if self.pressure == ThermalPressure.NOMINAL:
            return 1.0
        elif self.pressure == ThermalPressure.FAIR:
            return 0.9
        elif self.pressure == ThermalPressure.SERIOUS:
            return 0.7
        else:  # CRITICAL
            return 0.5


def get_thermal_state() -> ThermalState:
    """Get the current thermal state from macOS.

    Uses multiple methods to detect thermal pressure:
    1. pmset -g therm (always available, no sudo)
    2. powermetrics (more detailed, requires sudo)
    3. Fallback to nominal if detection fails

    Returns:
        Current ThermalState.
    """
    pressure = ThermalPressure.NOMINAL
    cpu_temp = None
    gpu_temp = None
    fan_speed = None

    # Try pmset first (no sudo required)
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "cpu_speed_limit" in output:
                # Parse speed limit percentage
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

    # Try to get more detailed info via IOKit (no sudo)
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AppleSmartBattery"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            # Parse temperature if available
            for line in result.stdout.split("\n"):
                if "Temperature" in line:
                    try:
                        # Temperature is in centidegrees
                        temp_val = int(line.split("=")[-1].strip())
                        cpu_temp = temp_val / 100.0
                    except (ValueError, IndexError):
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check thermal pressure via sysctl
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
            capture_output=True,
            text=True,
            timeout=2,
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
    """Get thermal state as a simple string.

    Returns one of: "nominal", "fair", "serious", "critical"
    """
    state = get_thermal_state()
    return state.pressure.value


class ThermalMonitor:
    """Monitor thermal state and trigger actions on throttling.

    Periodically checks thermal state and calls callbacks when
    the thermal pressure changes significantly.
    """

    def __init__(
        self,
        poll_interval_sec: float = 5.0,
        on_throttle: Optional[Callable[[ThermalState], None]] = None,
        on_recover: Optional[Callable[[ThermalState], None]] = None,
    ):
        """Initialize the thermal monitor.

        Args:
            poll_interval_sec: Interval between thermal checks.
            on_throttle: Callback when throttling is detected.
            on_recover: Callback when throttling ends.
        """
        self._interval = poll_interval_sec
        self._on_throttle = on_throttle
        self._on_recover = on_recover
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_state: Optional[ThermalState] = None
        self._was_throttling = False

    @property
    def current_state(self) -> Optional[ThermalState]:
        """Get the last recorded thermal state."""
        return self._last_state

    @property
    def is_throttling(self) -> bool:
        """Check if currently throttling."""
        if self._last_state is None:
            return False
        return self._last_state.is_throttling

    async def start(self) -> None:
        """Start thermal monitoring."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop thermal monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            self._check_thermal()
            await asyncio.sleep(self._interval)

    def _check_thermal(self) -> None:
        """Check thermal state and trigger callbacks."""
        state = get_thermal_state()
        self._last_state = state

        is_throttling = state.is_throttling

        if is_throttling and not self._was_throttling:
            console.print(
                f"[yellow]Thermal throttling detected: {state.pressure.value}[/yellow]"
            )
            if self._on_throttle:
                self._on_throttle(state)
        elif not is_throttling and self._was_throttling:
            console.print(
                f"[green]Thermal throttling ended: {state.pressure.value}[/green]"
            )
            if self._on_recover:
                self._on_recover(state)

        self._was_throttling = is_throttling

    def get_state(self) -> ThermalState:
        """Get current thermal state (synchronous poll)."""
        state = get_thermal_state()
        self._last_state = state
        return state


def estimate_safe_batch_size(
    current_batch_size: int,
    thermal_state: ThermalState,
) -> int:
    """Estimate a safe batch size given thermal state.

    Args:
        current_batch_size: Current batch size.
        thermal_state: Current thermal state.

    Returns:
        Suggested batch size (may be reduced).
    """
    multiplier = thermal_state.workload_multiplier
    new_size = int(current_batch_size * multiplier)
    return max(1, new_size)


def thermal_state_to_string(state: ThermalState) -> str:
    """Format thermal state for display.

    Args:
        state: Thermal state to format.

    Returns:
        Human-readable string.
    """
    parts = [f"Pressure: {state.pressure.value}"]

    if state.cpu_temp_celsius is not None:
        parts.append(f"CPU: {state.cpu_temp_celsius:.1f}°C")

    if state.gpu_temp_celsius is not None:
        parts.append(f"GPU: {state.gpu_temp_celsius:.1f}°C")

    if state.fan_speed_rpm is not None:
        parts.append(f"Fan: {state.fan_speed_rpm} RPM")

    return " | ".join(parts)
