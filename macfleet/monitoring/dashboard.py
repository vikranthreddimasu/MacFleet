"""Rich terminal dashboard for MacFleet training monitoring.

Displays real-time training metrics including loss, throughput,
communication time, compression ratio, and node health.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from macfleet.monitoring.health import NodeHealth, NodeStatus

console = Console()


@dataclass
class TrainingMetrics:
    """Metrics for the training dashboard."""
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0

    # Timing
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Throughput
    samples_per_sec: float = 0.0
    samples_processed: int = 0

    # Compression
    compression_ratio: float = 1.0
    bytes_sent: int = 0
    bytes_received: int = 0

    # History for charts
    loss_history: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)


@dataclass
class ClusterMetrics:
    """Cluster-wide metrics."""
    world_size: int = 1
    nodes: Dict[int, NodeHealth] = field(default_factory=dict)
    training_active: bool = False
    start_time: float = 0.0

    @property
    def elapsed_time(self) -> float:
        """Time since training started."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def alive_nodes(self) -> int:
        """Number of alive nodes."""
        return sum(1 for n in self.nodes.values() if n.is_alive)


class Dashboard:
    """Rich terminal dashboard for training monitoring.

    Displays a live-updating dashboard with:
    - Training progress (epoch, step, loss, accuracy)
    - Performance metrics (throughput, compute/comm time)
    - Cluster status (node health, thermal state)
    - Compression stats
    """

    def __init__(
        self,
        title: str = "MacFleet Training Dashboard",
        refresh_rate: float = 4.0,
    ):
        """Initialize the dashboard.

        Args:
            title: Dashboard title.
            refresh_rate: Refresh rate in Hz.
        """
        self.title = title
        self.refresh_rate = refresh_rate

        self.training_metrics = TrainingMetrics()
        self.cluster_metrics = ClusterMetrics()

        self._live: Optional[Live] = None
        self._running = False

    def start(self) -> None:
        """Start the live dashboard."""
        self._running = True
        self._live = Live(
            self._generate_layout(),
            console=console,
            refresh_per_second=self.refresh_rate,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live dashboard."""
        self._running = False
        if self._live:
            self._live.stop()
            self._live = None

    def update(self) -> None:
        """Update the dashboard display."""
        if self._live:
            self._live.update(self._generate_layout())

    def _generate_layout(self) -> Panel:
        """Generate the dashboard layout."""
        # Training progress section
        training_table = self._create_training_table()

        # Performance section
        perf_table = self._create_performance_table()

        # Cluster section
        cluster_table = self._create_cluster_table()

        # Combine into layout
        grid = Table.grid(padding=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        grid.add_row(
            Panel(training_table, title="Training Progress", border_style="blue"),
            Panel(perf_table, title="Performance", border_style="green"),
        )
        grid.add_row(
            Panel(cluster_table, title="Cluster Status", border_style="yellow"),
            Panel(self._create_compression_table(), title="Compression", border_style="magenta"),
        )

        return Panel(
            grid,
            title=f"[bold]{self.title}[/bold]",
            border_style="cyan",
        )

    def _create_training_table(self) -> Table:
        """Create the training metrics table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        m = self.training_metrics

        table.add_row("Epoch", f"{m.epoch + 1}")
        table.add_row("Step", f"{m.step}/{m.total_steps}" if m.total_steps > 0 else str(m.step))
        table.add_row("Loss", f"{m.loss:.4f}")
        table.add_row("Accuracy", f"{m.accuracy:.2%}")
        table.add_row("Learning Rate", f"{m.learning_rate:.6f}")
        table.add_row("Samples", f"{m.samples_processed:,}")

        # Progress bar
        if m.total_steps > 0:
            progress = m.step / m.total_steps
            bar_width = 20
            filled = int(progress * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            table.add_row("Progress", f"[{bar}] {progress:.1%}")

        return table

    def _create_performance_table(self) -> Table:
        """Create the performance metrics table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="green")
        table.add_column("Value", justify="right")

        m = self.training_metrics

        table.add_row("Throughput", f"{m.samples_per_sec:.1f} samples/s")
        table.add_row("Compute Time", f"{m.compute_time_ms:.1f} ms")
        table.add_row("Comm Time", f"{m.comm_time_ms:.1f} ms")
        table.add_row("Total Time", f"{m.total_time_ms:.1f} ms")

        # Comm overhead percentage
        if m.total_time_ms > 0:
            comm_pct = (m.comm_time_ms / m.total_time_ms) * 100
            color = "green" if comm_pct < 20 else "yellow" if comm_pct < 50 else "red"
            table.add_row("Comm Overhead", f"[{color}]{comm_pct:.1f}%[/{color}]")

        # Elapsed time
        elapsed = self.cluster_metrics.elapsed_time
        if elapsed > 0:
            hours, rem = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(rem, 60)
            table.add_row("Elapsed", f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        return table

    def _create_cluster_table(self) -> Table:
        """Create the cluster status table."""
        table = Table(box=None, padding=(0, 1))
        table.add_column("Rank", style="cyan", justify="center")
        table.add_column("Host", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Throughput", justify="right")
        table.add_column("Thermal", justify="center")

        for rank, node in sorted(self.cluster_metrics.nodes.items()):
            # Status with color
            status = node.status.value
            if node.status == NodeStatus.TRAINING:
                status_text = f"[green]{status}[/green]"
            elif node.status == NodeStatus.CONNECTED:
                status_text = f"[blue]{status}[/blue]"
            elif node.status == NodeStatus.DISCONNECTED:
                status_text = f"[red]{status}[/red]"
            else:
                status_text = f"[yellow]{status}[/yellow]"

            # Thermal with color
            thermal = node.thermal_state
            if thermal == "nominal":
                thermal_text = f"[green]{thermal}[/green]"
            elif thermal == "fair":
                thermal_text = f"[yellow]{thermal}[/yellow]"
            else:
                thermal_text = f"[red]{thermal}[/red]"

            table.add_row(
                str(rank),
                node.hostname[:15],
                status_text,
                f"{node.throughput:.1f}/s",
                thermal_text,
            )

        if not self.cluster_metrics.nodes:
            table.add_row("-", "No nodes", "-", "-", "-")

        return table

    def _create_compression_table(self) -> Table:
        """Create the compression stats table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="magenta")
        table.add_column("Value", justify="right")

        m = self.training_metrics

        table.add_row("Compression Ratio", f"{m.compression_ratio:.1%}")
        table.add_row("Bytes Sent", format_bytes(m.bytes_sent))
        table.add_row("Bytes Received", format_bytes(m.bytes_received))

        # Bandwidth estimate
        if m.comm_time_ms > 0:
            bandwidth = (m.bytes_sent + m.bytes_received) / (m.comm_time_ms / 1000)
            table.add_row("Bandwidth", f"{format_bytes(int(bandwidth))}/s")

        return table

    def update_training(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        samples_processed: Optional[int] = None,
    ) -> None:
        """Update training metrics."""
        m = self.training_metrics

        if epoch is not None:
            m.epoch = epoch
        if step is not None:
            m.step = step
        if total_steps is not None:
            m.total_steps = total_steps
        if loss is not None:
            m.loss = loss
            m.loss_history.append(loss)
            if len(m.loss_history) > 100:
                m.loss_history.pop(0)
        if accuracy is not None:
            m.accuracy = accuracy
        if learning_rate is not None:
            m.learning_rate = learning_rate
        if samples_processed is not None:
            m.samples_processed = samples_processed

        self.update()

    def update_performance(
        self,
        compute_time_ms: Optional[float] = None,
        comm_time_ms: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
        compression_ratio: Optional[float] = None,
        bytes_sent: Optional[int] = None,
        bytes_received: Optional[int] = None,
    ) -> None:
        """Update performance metrics."""
        m = self.training_metrics

        if compute_time_ms is not None:
            m.compute_time_ms = compute_time_ms
        if comm_time_ms is not None:
            m.comm_time_ms = comm_time_ms
        if samples_per_sec is not None:
            m.samples_per_sec = samples_per_sec
            m.throughput_history.append(samples_per_sec)
            if len(m.throughput_history) > 100:
                m.throughput_history.pop(0)
        if compression_ratio is not None:
            m.compression_ratio = compression_ratio
        if bytes_sent is not None:
            m.bytes_sent = bytes_sent
        if bytes_received is not None:
            m.bytes_received = bytes_received

        m.total_time_ms = m.compute_time_ms + m.comm_time_ms

        self.update()

    def update_node(self, health: NodeHealth) -> None:
        """Update node health information."""
        self.cluster_metrics.nodes[health.rank] = health
        self.update()

    def set_training_active(self, active: bool) -> None:
        """Set training active state."""
        self.cluster_metrics.training_active = active
        if active and self.cluster_metrics.start_time == 0:
            self.cluster_metrics.start_time = time.time()
        self.update()


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def create_simple_progress(
    description: str,
    total: int,
) -> Progress:
    """Create a simple progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def print_training_summary(
    epochs: int,
    final_loss: float,
    final_accuracy: float,
    total_time: float,
    samples_trained: int,
) -> None:
    """Print a training summary."""
    table = Table(title="Training Summary", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Epochs", str(epochs))
    table.add_row("Final Loss", f"{final_loss:.4f}")
    table.add_row("Final Accuracy", f"{final_accuracy:.2%}")
    table.add_row("Total Time", f"{total_time:.1f}s")
    table.add_row("Samples Trained", f"{samples_trained:,}")
    table.add_row("Avg Throughput", f"{samples_trained/total_time:.1f} samples/s")

    console.print(table)
