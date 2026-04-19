"""Tests for Rich TUI dashboard."""

from __future__ import annotations

from macfleet.engines.base import ThermalPressure
from macfleet.monitoring.dashboard import (
    Dashboard,
    build_cluster_table,
    build_network_panel,
    build_training_panel,
    build_warnings_panel,
    print_training_summary,
)
from macfleet.monitoring.health import (
    HealthStatus,
    MemoryInfo,
    NodeHealth,
)
from macfleet.monitoring.thermal import ThermalState


def _make_node(
    node_id: str = "test-node",
    status: HealthStatus = HealthStatus.HEALTHY,
    pressure: ThermalPressure = ThermalPressure.NOMINAL,
    mem_used: float = 4.0,
    mem_total: float = 16.0,
    throughput: float = 100.0,
) -> NodeHealth:
    return NodeHealth(
        node_id=node_id,
        status=status,
        thermal=ThermalState(pressure=pressure),
        memory=MemoryInfo(total_gb=mem_total, used_gb=mem_used, available_gb=mem_total - mem_used),
        throughput_samples_sec=throughput,
    )


class TestBuildClusterTable:
    def test_empty_nodes(self):
        table = build_cluster_table([])
        assert table.row_count == 1  # placeholder row

    def test_single_node(self):
        table = build_cluster_table([_make_node()])
        assert table.row_count == 1

    def test_multiple_nodes(self):
        nodes = [
            _make_node("node-1", HealthStatus.HEALTHY),
            _make_node("node-2", HealthStatus.DEGRADED, ThermalPressure.FAIR),
            _make_node("node-3", HealthStatus.UNHEALTHY, ThermalPressure.CRITICAL),
        ]
        table = build_cluster_table(nodes)
        assert table.row_count == 3


class TestBuildTrainingPanel:
    def test_basic(self):
        panel = build_training_panel(epoch=5, total_epochs=10, loss=0.5, throughput=200.0)
        # Panel should render without error
        assert panel.title == "Training Progress"

    def test_empty(self):
        panel = build_training_panel()
        assert panel.title == "Training Progress"


class TestBuildWarningsPanel:
    def test_no_warnings(self):
        panel = build_warnings_panel([_make_node()])
        assert panel.title == "Warnings"

    def test_with_warnings(self):
        node = _make_node(pressure=ThermalPressure.SERIOUS)
        panel = build_warnings_panel([node])
        assert panel.title == "Warnings"


class TestBuildNetworkPanel:
    def test_basic(self):
        panel = build_network_panel(bytes_sent=1024 * 1024, bytes_saved=512 * 1024)
        assert panel.title == "Network"

    def test_empty(self):
        panel = build_network_panel()
        assert panel.title == "Network"


class TestDashboard:
    def test_create(self):
        dash = Dashboard()
        assert dash._nodes == []

    def test_update_training(self):
        dash = Dashboard()
        dash.update_training(epoch=5, total_epochs=10, loss=0.3)
        assert dash._epoch == 5
        assert dash._total_epochs == 10
        assert dash._loss == 0.3

    def test_update_nodes(self):
        dash = Dashboard()
        nodes = [_make_node("n1"), _make_node("n2")]
        dash.update_nodes(nodes)
        assert len(dash._nodes) == 2

    def test_update_network(self):
        dash = Dashboard()
        dash.update_network(bytes_sent=1000, bytes_saved=500)
        assert dash._bytes_sent == 1000
        assert dash._bytes_saved == 500

    def test_render_no_crash(self):
        """Dashboard should render without errors even with no data."""
        dash = Dashboard()
        layout = dash._render()
        assert layout is not None

    def test_render_with_data(self):
        """Dashboard should render with data."""
        dash = Dashboard()
        dash.update_training(epoch=3, total_epochs=10, loss=0.5, throughput=150.0)
        dash.update_nodes([_make_node("test-1")])
        layout = dash._render()
        assert layout is not None


class TestPrintTrainingSummary:
    def test_prints_without_error(self, capsys):
        from rich.console import Console

        console = Console(file=open("/dev/null", "w"))
        print_training_summary(
            {"loss": 0.1, "epochs": 10, "steps": 100, "time_sec": 30.0,
             "loss_history": [1.0, 0.5, 0.3, 0.1]},
            console=console,
        )
