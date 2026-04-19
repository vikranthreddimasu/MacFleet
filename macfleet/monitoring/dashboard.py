"""Rich TUI dashboard for MacFleet cluster monitoring.

Displays real-time cluster topology, training progress,
per-node health, and throughput metrics.
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from macfleet.engines.base import ThermalPressure
from macfleet.monitoring.health import HealthStatus, NodeHealth

# ------------------------------------------------------------------ #
# Building blocks: tables and panels                                 #
# ------------------------------------------------------------------ #


def build_cluster_table(nodes: list[NodeHealth]) -> Table:
    """Build a table showing all cluster nodes."""
    table = Table(title="Cluster Nodes", expand=True)
    table.add_column("Node", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Thermal", justify="center")
    table.add_column("Memory", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Sync", justify="right")
    table.add_column("Score", justify="right")

    for node in nodes:
        # Status indicator
        status_style = {
            HealthStatus.HEALTHY: "[green]OK[/green]",
            HealthStatus.DEGRADED: "[yellow]WARN[/yellow]",
            HealthStatus.UNHEALTHY: "[red]FAIL[/red]",
            HealthStatus.UNKNOWN: "[dim]?[/dim]",
        }
        status = status_style.get(node.status, "[dim]?[/dim]")

        # Thermal indicator
        thermal_str = "[dim]--[/dim]"
        if node.thermal:
            thermal_style = {
                ThermalPressure.NOMINAL: "[green]",
                ThermalPressure.FAIR: "[yellow]",
                ThermalPressure.SERIOUS: "[red]",
                ThermalPressure.CRITICAL: "[bold red]",
            }
            style = thermal_style.get(node.thermal.pressure, "")
            thermal_str = f"{style}{node.thermal.pressure.value}[/]"

        # Memory
        mem_str = "[dim]--[/dim]"
        if node.memory and node.memory.total_gb > 0:
            mem_str = f"{node.memory.used_gb:.1f}/{node.memory.total_gb:.0f}GB"

        # Throughput
        tp_str = f"{node.throughput_samples_sec:.0f} s/s" if node.throughput_samples_sec > 0 else "[dim]--[/dim]"

        # Sync time
        sync_str = f"{node.avg_sync_time_sec * 1000:.0f}ms" if node.avg_sync_time_sec > 0 else "[dim]--[/dim]"

        # Score
        score = node.health_score
        if score >= 0.7:
            score_str = f"[green]{score:.2f}[/green]"
        elif score >= 0.4:
            score_str = f"[yellow]{score:.2f}[/yellow]"
        else:
            score_str = f"[red]{score:.2f}[/red]"

        table.add_row(
            node.node_id or "unknown",
            status,
            thermal_str,
            mem_str,
            tp_str,
            sync_str,
            score_str,
        )

    if not nodes:
        table.add_row("[dim]No nodes connected[/dim]", "", "", "", "", "", "")

    return table


def build_training_panel(
    epoch: int = 0,
    total_epochs: int = 0,
    step: int = 0,
    loss: float = 0.0,
    throughput: float = 0.0,
    elapsed_sec: float = 0.0,
    compression_ratio: float = 1.0,
) -> Panel:
    """Build a panel showing training progress."""
    lines = []

    if total_epochs > 0:
        pct = (epoch / total_epochs) * 100
        bar_len = 30
        filled = int(bar_len * epoch / total_epochs)
        bar = "[green]" + "=" * filled + "[/green]" + "[dim]" + "-" * (bar_len - filled) + "[/dim]"
        lines.append(f"Epoch: {epoch}/{total_epochs} [{bar}] {pct:.0f}%")
    else:
        lines.append("Epoch: --")

    lines.append(f"Step:  {step}")
    lines.append(f"Loss:  {loss:.4f}" if loss > 0 else "Loss:  --")
    lines.append(f"Speed: {throughput:.1f} samples/sec" if throughput > 0 else "Speed: --")

    if elapsed_sec > 0:
        mins = int(elapsed_sec // 60)
        secs = int(elapsed_sec % 60)
        lines.append(f"Time:  {mins}m {secs}s")

    if compression_ratio < 1.0:
        lines.append(f"Compression: {compression_ratio:.1%}")

    return Panel("\n".join(lines), title="Training Progress", border_style="blue")


def build_warnings_panel(nodes: list[NodeHealth]) -> Panel:
    """Build a panel showing active warnings."""
    warnings = []
    for node in nodes:
        for w in node.warnings:
            warnings.append(f"[yellow]{node.node_id}[/yellow]: {w}")

    if not warnings:
        content = "[green]No warnings[/green]"
    else:
        content = "\n".join(warnings[:10])  # limit to 10

    return Panel(content, title="Warnings", border_style="yellow")


def build_network_panel(
    bytes_sent: int = 0,
    bytes_saved: int = 0,
    avg_latency_ms: float = 0.0,
) -> Panel:
    """Build a panel showing network statistics."""
    lines = []

    if bytes_sent > 0:
        sent_mb = bytes_sent / (1024 * 1024)
        lines.append(f"Sent:  {sent_mb:.1f} MB")
    if bytes_saved > 0:
        saved_mb = bytes_saved / (1024 * 1024)
        lines.append(f"Saved: {saved_mb:.1f} MB (compression)")
    if avg_latency_ms > 0:
        lines.append(f"Latency: {avg_latency_ms:.1f} ms")

    if not lines:
        lines.append("[dim]No network activity[/dim]")

    return Panel("\n".join(lines), title="Network", border_style="cyan")


# ------------------------------------------------------------------ #
# Dashboard class                                                    #
# ------------------------------------------------------------------ #


class Dashboard:
    """Rich TUI dashboard for monitoring MacFleet training.

    Usage:
        dashboard = Dashboard()
        dashboard.start()
        # ... during training ...
        dashboard.update_training(epoch=1, loss=0.5, throughput=100.0)
        dashboard.update_nodes([node_health1, node_health2])
        # ... after training ...
        dashboard.stop()

    Or as a context manager:
        with Dashboard() as dash:
            dash.update_training(epoch=1, loss=0.5)
    """

    def __init__(self, refresh_rate: float = 2.0):
        self._console = Console()
        self._live: Optional[Live] = None
        self._refresh_rate = refresh_rate

        # State
        self._nodes: list[NodeHealth] = []
        self._epoch = 0
        self._total_epochs = 0
        self._step = 0
        self._loss = 0.0
        self._throughput = 0.0
        self._elapsed_sec = 0.0
        self._compression_ratio = 1.0
        self._bytes_sent = 0
        self._bytes_saved = 0
        self._avg_latency_ms = 0.0

    def __enter__(self) -> Dashboard:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def start(self) -> None:
        """Start the live dashboard."""
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=1.0 / self._refresh_rate,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live dashboard."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_training(
        self,
        epoch: int = 0,
        total_epochs: int = 0,
        step: int = 0,
        loss: float = 0.0,
        throughput: float = 0.0,
        elapsed_sec: float = 0.0,
        compression_ratio: float = 1.0,
    ) -> None:
        """Update training progress."""
        if epoch: self._epoch = epoch
        if total_epochs: self._total_epochs = total_epochs
        if step: self._step = step
        if loss: self._loss = loss
        if throughput: self._throughput = throughput
        if elapsed_sec: self._elapsed_sec = elapsed_sec
        if compression_ratio < 1.0: self._compression_ratio = compression_ratio
        self._refresh()

    def update_nodes(self, nodes: list[NodeHealth]) -> None:
        """Update cluster node health."""
        self._nodes = nodes
        self._refresh()

    def update_network(
        self,
        bytes_sent: int = 0,
        bytes_saved: int = 0,
        avg_latency_ms: float = 0.0,
    ) -> None:
        """Update network statistics."""
        self._bytes_sent = bytes_sent
        self._bytes_saved = bytes_saved
        self._avg_latency_ms = avg_latency_ms
        self._refresh()

    def _refresh(self) -> None:
        """Re-render the dashboard."""
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Layout:
        """Render the full dashboard layout."""
        layout = Layout()

        # Header
        header = Layout(
            Panel(
                Text("MacFleet Cluster Dashboard", style="bold white", justify="center"),
                border_style="bright_blue",
            ),
            size=3,
        )

        # Main content
        body = Layout()

        # Left: cluster table
        left = Layout(build_cluster_table(self._nodes), ratio=3)

        # Right: stacked panels
        right = Layout(ratio=2)
        right.split_column(
            Layout(build_training_panel(
                epoch=self._epoch,
                total_epochs=self._total_epochs,
                step=self._step,
                loss=self._loss,
                throughput=self._throughput,
                elapsed_sec=self._elapsed_sec,
                compression_ratio=self._compression_ratio,
            )),
            Layout(build_network_panel(
                bytes_sent=self._bytes_sent,
                bytes_saved=self._bytes_saved,
                avg_latency_ms=self._avg_latency_ms,
            )),
            Layout(build_warnings_panel(self._nodes)),
        )

        body.split_row(left, right)

        layout.split_column(header, body)
        return layout


# ------------------------------------------------------------------ #
# Standalone display helpers for CLI                                 #
# ------------------------------------------------------------------ #


def print_cluster_status(nodes: list[NodeHealth], console: Optional[Console] = None) -> None:
    """Print cluster status table to console (non-live)."""
    c = console or Console()
    c.print(build_cluster_table(nodes))

    warnings = []
    for node in nodes:
        warnings.extend(node.warnings)
    if warnings:
        c.print()
        for w in warnings:
            c.print(f"  [yellow]![/yellow] {w}")


def print_training_summary(
    results: dict,
    console: Optional[Console] = None,
) -> None:
    """Print a training summary after completion."""
    c = console or Console()

    table = Table(title="Training Complete", expand=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Final Loss", f"{results.get('loss', 0):.4f}")
    table.add_row("Epochs", str(results.get("epochs", 0)))
    table.add_row("Total Steps", str(results.get("steps", 0)))
    table.add_row("Time", f"{results.get('time_sec', 0):.1f}s")

    if "loss_history" in results and results["loss_history"]:
        history = results["loss_history"]
        table.add_row("Start Loss", f"{history[0]:.4f}")
        table.add_row("Min Loss", f"{min(history):.4f}")

    c.print(table)
