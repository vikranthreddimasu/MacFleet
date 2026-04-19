"""Tests for ThermalPauseController (v2.2 PR 12)."""

from __future__ import annotations

import time

from macfleet.engines.base import ThermalPressure
from macfleet.monitoring.thermal import ThermalState
from macfleet.monitoring.thermal_pause import (
    PauseState,
    ThermalPauseConfig,
    ThermalPauseController,
    ThermalPauseEvent,
)


class FakeThermalReader:
    """Inject a controllable thermal source for tests."""

    def __init__(self, pressure: ThermalPressure = ThermalPressure.NOMINAL):
        self.pressure = pressure
        self.calls = 0

    def __call__(self) -> ThermalState:
        self.calls += 1
        return ThermalState(pressure=self.pressure)


class TestThermalPauseFSM:
    def test_starts_running(self):
        ctrl = ThermalPauseController(read_thermal=FakeThermalReader())
        assert ctrl.state == PauseState.RUNNING
        assert not ctrl.is_paused()

    def test_pauses_on_serious(self):
        reader = FakeThermalReader(ThermalPressure.SERIOUS)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=reader,
        )
        assert ctrl.should_pause() is True
        assert ctrl.is_paused()

    def test_pauses_on_critical(self):
        reader = FakeThermalReader(ThermalPressure.CRITICAL)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=reader,
        )
        assert ctrl.should_pause() is True

    def test_does_not_pause_on_fair(self):
        reader = FakeThermalReader(ThermalPressure.FAIR)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=reader,
        )
        assert ctrl.should_pause() is False

    def test_does_not_pause_on_nominal(self):
        reader = FakeThermalReader(ThermalPressure.NOMINAL)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=reader,
        )
        assert ctrl.should_pause() is False


class TestHysteresis:
    """Pause at SERIOUS, resume only at FAIR (not when still at SERIOUS)."""

    def test_does_not_resume_at_serious(self):
        reader = FakeThermalReader(ThermalPressure.SERIOUS)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.0, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert ctrl.is_paused()
        # Still SERIOUS → stays paused
        ctrl.tick()
        assert ctrl.is_paused()

    def test_resumes_at_fair(self):
        reader = FakeThermalReader(ThermalPressure.SERIOUS)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.0, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert ctrl.is_paused()

        reader.pressure = ThermalPressure.FAIR
        ctrl.tick()
        assert not ctrl.is_paused()

    def test_resumes_at_nominal(self):
        reader = FakeThermalReader(ThermalPressure.CRITICAL)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.0, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert ctrl.is_paused()

        reader.pressure = ThermalPressure.NOMINAL
        ctrl.tick()
        assert not ctrl.is_paused()


class TestMinPauseWindow:
    def test_min_pause_holds_state(self):
        """Even if thermal recovers instantly, min_pause_sec is honored."""
        reader = FakeThermalReader(ThermalPressure.SERIOUS)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.0, min_pause_sec=0.5,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert ctrl.is_paused()

        reader.pressure = ThermalPressure.NOMINAL
        ctrl.tick()
        # Min-pause hasn't elapsed → still paused
        assert ctrl.is_paused()

        time.sleep(0.55)
        ctrl.tick()
        # Past min-pause AND thermal is good → resume
        assert not ctrl.is_paused()


class TestPollInterval:
    def test_poll_interval_throttles_reads(self):
        """Back-to-back tick() calls don't spam the thermal read."""
        reader = FakeThermalReader(ThermalPressure.NOMINAL)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=10.0),
            read_thermal=reader,
        )
        ctrl.tick()
        ctrl.tick()
        ctrl.tick()
        ctrl.tick()
        # Only the first call should have triggered a read
        assert reader.calls == 1

    def test_zero_interval_always_reads(self):
        reader = FakeThermalReader(ThermalPressure.NOMINAL)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=reader,
        )
        for _ in range(5):
            ctrl.tick()
        assert reader.calls == 5


class TestCallbacks:
    def test_on_pause_fires(self):
        events: list[ThermalPauseEvent] = []
        reader = FakeThermalReader(ThermalPressure.SERIOUS)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0, min_pause_sec=0.0),
            read_thermal=reader,
            on_pause=events.append,
        )
        ctrl.tick()
        assert len(events) == 1
        assert events[0].state == PauseState.PAUSED
        assert events[0].thermal.pressure == ThermalPressure.SERIOUS

    def test_on_resume_fires(self):
        events: list[ThermalPauseEvent] = []
        reader = FakeThermalReader(ThermalPressure.SERIOUS)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0, min_pause_sec=0.0),
            read_thermal=reader,
            on_resume=events.append,
        )
        ctrl.tick()  # pause
        reader.pressure = ThermalPressure.NOMINAL
        ctrl.tick()  # resume
        assert len(events) == 1
        assert events[0].state == PauseState.RUNNING
        assert events[0].thermal.pressure == ThermalPressure.NOMINAL

    def test_callback_exception_swallowed(self):
        """A buggy callback can't crash the FSM."""
        reader = FakeThermalReader(ThermalPressure.SERIOUS)

        def bad(evt):
            raise RuntimeError("callback bug")

        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0, min_pause_sec=0.0),
            read_thermal=reader,
            on_pause=bad,
        )
        # Must not raise — FSM still transitions
        ctrl.tick()
        assert ctrl.is_paused()


class TestThermalReadFailureHandling:
    def test_exception_holds_state(self):
        """If get_thermal_state() fails, don't flip state on missing data."""
        def boom():
            raise RuntimeError("pmset died")

        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=boom,
        )
        # Starts RUNNING, stays RUNNING on error
        ctrl.tick()
        assert ctrl.state == PauseState.RUNNING


class TestWaitForResume:
    def test_returns_immediately_if_not_paused(self):
        reader = FakeThermalReader(ThermalPressure.NOMINAL)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.01),
            read_thermal=reader,
        )
        start = time.monotonic()
        assert ctrl.wait_for_resume(timeout_sec=5.0) is True
        # Immediate — no sleep
        assert (time.monotonic() - start) < 0.1

    def test_waits_for_pressure_drop(self):
        reader = FakeThermalReader(ThermalPressure.SERIOUS)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.01, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        assert ctrl.is_paused()

        # Schedule a thermal recovery on a background timer
        import threading

        def recover():
            time.sleep(0.1)
            reader.pressure = ThermalPressure.NOMINAL

        threading.Thread(target=recover, daemon=True).start()

        # Should resume within ~200ms
        assert ctrl.wait_for_resume(timeout_sec=2.0) is True
        assert not ctrl.is_paused()

    def test_timeout_returns_false(self):
        reader = FakeThermalReader(ThermalPressure.CRITICAL)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(
                poll_interval_sec=0.01, min_pause_sec=0.0,
            ),
            read_thermal=reader,
        )
        ctrl.tick()
        # Thermal never recovers; timeout must fire
        assert ctrl.wait_for_resume(timeout_sec=0.2) is False
        assert ctrl.is_paused()


class TestLastThermal:
    def test_reports_latest_reading(self):
        reader = FakeThermalReader(ThermalPressure.FAIR)
        ctrl = ThermalPauseController(
            config=ThermalPauseConfig(poll_interval_sec=0.0),
            read_thermal=reader,
        )
        assert ctrl.last_thermal() is None
        ctrl.tick()
        assert ctrl.last_thermal().pressure == ThermalPressure.FAIR
