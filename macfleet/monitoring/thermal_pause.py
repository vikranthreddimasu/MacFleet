"""Thermal-aware training pause controller (E4 + A16 + A17).

v2.2 PR 12: long training runs on Apple Silicon that aren't plugged into
a fan-cooled rig will throttle after 10-30 minutes of sustained GPU load.
Without thermal awareness, the loss curve shows a visible knee where
throughput halves. With sustained throttling, NaN gradients start
appearing and the model silently diverges.

This module provides a standalone `ThermalPauseController` that training
loops poll between steps. If local thermal state crosses a configured
threshold, the controller raises a pause. Downstream code skips the
forward/backward pass, waits a cool-down window, checks again, and
resumes only when thermal pressure drops back below the resume threshold.

Hysteresis is intentional (A16): without a separate resume threshold,
the system oscillates between PAUSED and RUNNING once per second as the
thermal state jitters at the boundary. With hysteresis:
    - Pause when pressure >= pause_at (default SERIOUS)
    - Resume when pressure <= resume_at (default FAIR)

Coordinator broadcast (A17) is deferred until Pool.train's distributed
wiring lands. Once it does, the coordinator can poll every peer's local
controller state and broadcast a fleet-wide pause when any one peer
hits thermal. This file ships the local-policy primitive that both
self-check loops AND future coordinator code can build on.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from macfleet.engines.base import ThermalPressure
from macfleet.monitoring.thermal import ThermalState, get_thermal_state

logger = logging.getLogger(__name__)


class PauseState(str, Enum):
    """Current state of the thermal pause FSM."""
    RUNNING = "running"
    PAUSED = "paused"


# Default hysteresis thresholds — these map to ThermalPressure values.
# Pause triggers at SERIOUS or CRITICAL (is_throttling == True by design).
# Resume waits until we're back to FAIR or NOMINAL. Crossing from SERIOUS
# directly back to NOMINAL is rare in practice; FAIR is the realistic
# resume target for M-series chips after a brief cool-down.
_DEFAULT_PAUSE_AT = ThermalPressure.SERIOUS
_DEFAULT_RESUME_AT = ThermalPressure.FAIR


@dataclass
class ThermalPauseConfig:
    """Tuning for the pause controller."""
    pause_at: ThermalPressure = _DEFAULT_PAUSE_AT
    resume_at: ThermalPressure = _DEFAULT_RESUME_AT
    # Minimum seconds to stay paused even if thermal recovers fast — avoids
    # bouncy resumes during a noisy reading window.
    min_pause_sec: float = 5.0
    # How often the controller re-reads the OS thermal state (via
    # `get_thermal_state()`). A tight poll is fine — pmset is O(ms).
    poll_interval_sec: float = 1.0


@dataclass
class ThermalPauseEvent:
    """What happened + why. Emitted to on_pause/on_resume callbacks."""
    state: PauseState
    thermal: ThermalState
    timestamp: float = field(default_factory=time.monotonic)


def _pressure_rank(p: ThermalPressure) -> int:
    """Ordering helper (NOMINAL < FAIR < SERIOUS < CRITICAL)."""
    order = {
        ThermalPressure.NOMINAL: 0,
        ThermalPressure.FAIR: 1,
        ThermalPressure.SERIOUS: 2,
        ThermalPressure.CRITICAL: 3,
    }
    return order[p]


class ThermalPauseController:
    """Polls local thermal state and maintains a RUNNING/PAUSED FSM.

    Usage pattern (training loop):

        controller = ThermalPauseController()
        for step, batch in enumerate(dataloader):
            if controller.should_pause():
                controller.wait_for_resume()  # blocks until thermal drops
                continue  # optional: skip this step's forward/backward
            loss = training_step(batch)

    For async loops:

        while controller.is_paused():
            await asyncio.sleep(controller.config.poll_interval_sec)
            controller.tick()

    The controller does NOT own a thread. Callers drive it by calling
    `tick()` (or letting `wait_for_resume()` drive) whenever they want
    the FSM to re-evaluate.
    """

    def __init__(
        self,
        config: Optional[ThermalPauseConfig] = None,
        read_thermal: Optional[Callable[[], ThermalState]] = None,
        on_pause: Optional[Callable[[ThermalPauseEvent], None]] = None,
        on_resume: Optional[Callable[[ThermalPauseEvent], None]] = None,
    ):
        self.config = config or ThermalPauseConfig()
        # Injectable for tests — production uses `get_thermal_state()`.
        self._read_thermal = read_thermal or get_thermal_state
        self._on_pause = on_pause
        self._on_resume = on_resume

        self._state: PauseState = PauseState.RUNNING
        self._last_poll: float = 0.0
        self._paused_at: Optional[float] = None
        self._last_thermal: Optional[ThermalState] = None

    @property
    def state(self) -> PauseState:
        return self._state

    def is_paused(self) -> bool:
        """True if the controller is currently in PAUSED state."""
        return self._state == PauseState.PAUSED

    def should_pause(self) -> bool:
        """Tick the FSM and return whether the caller should pause NOW.

        This is the primary entry point for training loops — call before
        each step. Returns True if the controller is (or just entered)
        PAUSED state.
        """
        self.tick()
        return self.is_paused()

    def tick(self) -> PauseState:
        """Re-evaluate thermal state and update FSM. Returns new state.

        Respects `poll_interval_sec` — calling tick() in a tight loop
        won't hammer `pmset`. Uses monotonic time so wall-clock jumps
        (NTP sync mid-run) don't confuse the min-pause logic.
        """
        now = time.monotonic()
        if (now - self._last_poll) < self.config.poll_interval_sec:
            return self._state
        self._last_poll = now

        try:
            thermal = self._read_thermal()
        except Exception as e:
            # Thermal reads can transiently fail (pmset hiccup, ioreg busy).
            # Stay in current state rather than flipping on a bad reading.
            logger.debug("Thermal read failed, holding state: %s", e)
            return self._state
        self._last_thermal = thermal

        if self._state == PauseState.RUNNING:
            if _pressure_rank(thermal.pressure) >= _pressure_rank(self.config.pause_at):
                self._enter_paused(thermal, now)
        else:  # PAUSED
            # Honor min_pause_sec even if thermal recovers instantly
            if (self._paused_at is not None
                and now - self._paused_at < self.config.min_pause_sec):
                return self._state
            if _pressure_rank(thermal.pressure) <= _pressure_rank(self.config.resume_at):
                self._enter_running(thermal)

        return self._state

    def wait_for_resume(self, timeout_sec: Optional[float] = None) -> bool:
        """Block until the FSM transitions back to RUNNING.

        Uses `time.sleep(poll_interval_sec)` between ticks — intentionally
        synchronous. Returns True if resumed within the timeout, False on
        timeout. With timeout=None, waits forever.

        For async training loops use `await async_wait_for_resume(...)`
        instead so the event loop isn't blocked.
        """
        if not self.is_paused():
            return True
        deadline = (time.monotonic() + timeout_sec) if timeout_sec else float("inf")
        while self.is_paused():
            time.sleep(self.config.poll_interval_sec)
            self.tick()
            if time.monotonic() > deadline:
                return False
        return True

    async def async_wait_for_resume(
        self, timeout_sec: Optional[float] = None,
    ) -> bool:
        """Async variant of wait_for_resume — yields the loop while polling."""
        import asyncio
        if not self.is_paused():
            return True
        deadline = (time.monotonic() + timeout_sec) if timeout_sec else float("inf")
        while self.is_paused():
            await asyncio.sleep(self.config.poll_interval_sec)
            self.tick()
            if time.monotonic() > deadline:
                return False
        return True

    def last_thermal(self) -> Optional[ThermalState]:
        """Most recent thermal reading (None if tick() hasn't fired yet)."""
        return self._last_thermal

    def _enter_paused(self, thermal: ThermalState, now: float) -> None:
        self._state = PauseState.PAUSED
        self._paused_at = now
        logger.warning(
            "Thermal pause ENGAGED: pressure=%s (>= %s)",
            thermal.pressure.value, self.config.pause_at.value,
        )
        if self._on_pause:
            try:
                self._on_pause(ThermalPauseEvent(state=self._state, thermal=thermal))
            except Exception as e:
                logger.debug("on_pause callback raised: %s", e)

    def _enter_running(self, thermal: ThermalState) -> None:
        self._state = PauseState.RUNNING
        self._paused_at = None
        logger.info(
            "Thermal pause RELEASED: pressure=%s (<= %s)",
            thermal.pressure.value, self.config.resume_at.value,
        )
        if self._on_resume:
            try:
                self._on_resume(ThermalPauseEvent(state=self._state, thermal=thermal))
            except Exception as e:
                logger.debug("on_resume callback raised: %s", e)
