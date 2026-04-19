"""Tests for throughput tracking."""

from __future__ import annotations

import time

import pytest

from macfleet.monitoring.throughput import (
    StepMetrics,
    ThroughputTracker,
)

# ------------------------------------------------------------------ #
# StepMetrics                                                        #
# ------------------------------------------------------------------ #


class TestStepMetrics:
    def test_throughput(self):
        m = StepMetrics(step=0, samples=32, compute_time_sec=0.08, sync_time_sec=0.02, step_time_sec=0.1)
        assert m.throughput == pytest.approx(320.0, abs=1.0)

    def test_compute_pct(self):
        m = StepMetrics(step=0, samples=32, compute_time_sec=0.08, sync_time_sec=0.02, step_time_sec=0.1)
        assert m.compute_pct == pytest.approx(0.8, abs=0.01)

    def test_comm_pct(self):
        m = StepMetrics(step=0, samples=32, compute_time_sec=0.08, sync_time_sec=0.02, step_time_sec=0.1)
        assert m.comm_pct == pytest.approx(0.2, abs=0.01)

    def test_zero_step_time(self):
        m = StepMetrics(step=0, samples=32, compute_time_sec=0.0, sync_time_sec=0.0, step_time_sec=0.0)
        assert m.throughput == 0.0
        assert m.compute_pct == 0.0


# ------------------------------------------------------------------ #
# ThroughputTracker manual recording                                 #
# ------------------------------------------------------------------ #


class TestThroughputTracker:
    def test_empty_tracker(self):
        tracker = ThroughputTracker()
        assert tracker.avg_throughput == 0.0
        assert tracker.total_steps == 0
        assert tracker.total_samples == 0

    def test_record_step(self):
        tracker = ThroughputTracker()
        m = StepMetrics(step=0, samples=32, compute_time_sec=0.08, sync_time_sec=0.02, step_time_sec=0.1)
        tracker.record(m)

        assert tracker.total_steps == 1
        assert tracker.total_samples == 32
        assert tracker.avg_throughput == pytest.approx(320.0, abs=1.0)

    def test_rolling_average(self):
        tracker = ThroughputTracker(window_size=3)

        for i in range(5):
            m = StepMetrics(
                step=i, samples=32,
                compute_time_sec=0.08, sync_time_sec=0.02,
                step_time_sec=0.1,
                loss=1.0 / (i + 1),
            )
            tracker.record(m)

        assert tracker.total_steps == 5
        assert tracker.total_samples == 160
        # Window only holds last 3
        assert len(tracker.loss_history) == 3

    def test_comm_compute_ratio(self):
        tracker = ThroughputTracker()
        m = StepMetrics(step=0, samples=32, compute_time_sec=0.1, sync_time_sec=0.3, step_time_sec=0.4)
        tracker.record(m)

        assert tracker.comm_compute_ratio == pytest.approx(3.0, abs=0.01)

    def test_summary(self):
        tracker = ThroughputTracker()
        m = StepMetrics(step=0, samples=64, compute_time_sec=0.05, sync_time_sec=0.01, step_time_sec=0.06)
        tracker.record(m)

        s = tracker.summary()
        assert s["total_steps"] == 1
        assert s["total_samples"] == 64
        assert s["avg_throughput"] > 0


# ------------------------------------------------------------------ #
# StepContext                                                        #
# ------------------------------------------------------------------ #


class TestStepContext:
    def test_step_context_basic(self):
        tracker = ThroughputTracker()

        with tracker.step(32) as s:
            time.sleep(0.01)  # simulate compute
            s.compute_done()
            time.sleep(0.005)  # simulate sync
            s.sync_done()
            s.record_loss(0.5)

        assert tracker.total_steps == 1
        assert tracker.total_samples == 32
        assert tracker.avg_step_time > 0.01
        assert tracker.avg_compute_time > 0
        assert tracker.avg_sync_time > 0
        assert tracker.loss_history == [0.5]

    def test_multiple_steps(self):
        tracker = ThroughputTracker()

        for i in range(3):
            with tracker.step(16) as s:
                s.compute_done()
                s.sync_done()
                s.record_loss(1.0 / (i + 1))

        assert tracker.total_steps == 3
        assert tracker.total_samples == 48
        assert len(tracker.throughput_history) == 3

    def test_no_compute_done_call(self):
        """If compute_done() is never called, full step time is compute."""
        tracker = ThroughputTracker()

        with tracker.step(32):
            time.sleep(0.005)

        assert tracker.total_steps == 1
        # compute_time should equal step_time
        assert tracker.avg_compute_time == pytest.approx(tracker.avg_step_time, abs=0.001)
        assert tracker.avg_sync_time == 0.0
