"""MacFleet CLI: zero-config compute pool for Apple Silicon Macs.

Commands:
    macfleet join       Join the compute pool
    macfleet leave      Leave the pool gracefully
    macfleet status     Show pool members and network info
    macfleet info       Show local hardware info
    macfleet train      Submit a training job
    macfleet bench      Benchmark network + compute
    macfleet diagnose   System health check
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

import macfleet

console = Console()


@click.group()
@click.version_option(version=macfleet.__version__, prog_name="macfleet")
def cli():
    """MacFleet: Pool Apple Silicon Macs for distributed ML training."""
    pass


@cli.command()
@click.option("--name", default=None, help="Custom node name")
@click.option("--port", default=50051, help="Communication port")
@click.option("--token", default=None, envvar="MACFLEET_TOKEN", help="Pool token (or set MACFLEET_TOKEN env var)")
@click.option("--fleet-id", default=None, help="Fleet identifier (isolates pool on network)")
@click.option("--tls", "use_tls", is_flag=True, default=False, help="Enable TLS encryption")
def join(name: str | None, port: int, token: str | None, fleet_id: str | None, use_tls: bool):
    """Join the compute pool. Auto-discovers peers on the network."""
    from macfleet.pool.agent import PoolAgent

    agent = PoolAgent(name=name, port=port, token=token, fleet_id=fleet_id, tls=use_tls)

    async def run():
        await agent.start()
        console.print("\n[dim]Press Ctrl+C to leave the pool[/dim]\n")

        # Wait for interrupt
        stop_event = asyncio.Event()
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        await stop_event.wait()
        await agent.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


@cli.command()
def info():
    """Show local hardware information."""
    from macfleet.pool.agent import profile_hardware
    from macfleet.pool.network import get_network_topology
    from macfleet.monitoring.thermal import get_thermal_state, thermal_state_to_string

    hw = profile_hardware()
    topo = get_network_topology()
    thermal = get_thermal_state()

    table = Table(title="MacFleet Node Info", show_header=False)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Hostname", hw.hostname)
    table.add_row("Chip", hw.chip_name)
    table.add_row("GPU Cores", str(hw.gpu_cores))
    table.add_row("RAM", f"{hw.ram_gb:.0f} GB")
    table.add_row("Memory Bandwidth", f"~{hw.memory_bandwidth_gbps:.0f} GB/s")
    table.add_row("Compute Score", f"{hw.compute_score:.0f}")
    table.add_row("MPS Available", "yes" if hw.mps_available else "no")
    table.add_row("MLX Available", "yes" if hw.mlx_available else "no")
    table.add_row("Thermal", thermal_state_to_string(thermal))

    # Network
    table.add_row("", "")
    for link in topo.links:
        table.add_row(f"Network ({link.interface})", f"{link.link_type.value} — {link.ip_address}")

    console.print(table)


@cli.command()
@click.option("--token", default=None, envvar="MACFLEET_TOKEN", help="Pool token (scopes discovery to fleet)")
@click.option("--fleet-id", default=None, help="Fleet identifier")
def status(token: str | None, fleet_id: str | None):
    """Show pool status (discovers peers for 3 seconds)."""
    from macfleet.pool.discovery import ServiceRegistry
    from macfleet.security.auth import SecurityConfig

    sec = SecurityConfig(token=token, fleet_id=fleet_id) if token else None
    if sec and sec.is_secure:
        fleet_label = fleet_id or "default"
        console.print(f"[bold]Scanning fleet '{fleet_label}' for members...[/bold]")
    else:
        console.print("[bold]Scanning for pool members...[/bold]")

    registry = ServiceRegistry(security=sec)
    try:
        peers = registry.find_peers(timeout=3.0)
    finally:
        registry.stop()

    if not peers:
        console.print("[yellow]No pool members found on the network.[/yellow]")
        console.print("[dim]Run 'macfleet join' on this and other Macs to form a pool.[/dim]")
        return

    table = Table(title=f"MacFleet Pool ({len(peers)} nodes)")
    table.add_column("Hostname", style="bold")
    table.add_column("Chip")
    table.add_column("GPU Cores", justify="right")
    table.add_column("RAM (GB)", justify="right")
    table.add_column("IP Address")
    table.add_column("Score", justify="right")

    for node in sorted(peers, key=lambda n: -n.compute_score):
        table.add_row(
            node.hostname,
            node.chip_name,
            str(node.gpu_cores),
            str(node.ram_gb),
            f"{node.ip_address}:{node.port}",
            f"{node.compute_score:.0f}",
        )

    console.print(table)


@cli.command()
def diagnose():
    """Run system health checks."""
    from macfleet.pool.agent import profile_hardware, _check_mps_available, _check_mlx_available
    from macfleet.monitoring.thermal import get_thermal_state
    from macfleet.pool.network import detect_interfaces

    console.print("[bold]Running diagnostics...[/bold]\n")

    checks_passed = 0
    checks_total = 0

    def check(name: str, passed: bool, detail: str = ""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if passed:
            checks_passed += 1
            console.print(f"  [green]PASS[/green] {name}" + (f" — {detail}" if detail else ""))
        else:
            console.print(f"  [red]FAIL[/red] {name}" + (f" — {detail}" if detail else ""))

    # Hardware
    console.print("[bold]Hardware[/bold]")
    hw = profile_hardware()
    check("Apple Silicon detected", "apple" in hw.chip_name.lower() or "m" in hw.chip_name.lower(), hw.chip_name)
    check("GPU cores detected", hw.gpu_cores > 0, f"{hw.gpu_cores} cores")
    check("RAM detected", hw.ram_gb > 0, f"{hw.ram_gb:.0f} GB")
    check("RAM >= 8 GB", hw.ram_gb >= 8, f"{hw.ram_gb:.0f} GB")

    # Frameworks
    console.print("\n[bold]ML Frameworks[/bold]")
    check("MPS available", _check_mps_available())
    check("MLX available", _check_mlx_available())

    # Thermal
    console.print("\n[bold]Thermal[/bold]")
    thermal = get_thermal_state()
    check("Not throttling", not thermal.is_throttling, thermal.pressure.value)

    # Network
    console.print("\n[bold]Network[/bold]")
    links = detect_interfaces()
    check("Network interfaces found", len(links) > 0, f"{len(links)} interfaces")
    has_non_loopback = any(l.link_type.value != "loopback" for l in links)
    check("Non-loopback interface", has_non_loopback)

    # Summary
    console.print(f"\n[bold]{checks_passed}/{checks_total} checks passed[/bold]")
    if checks_passed == checks_total:
        console.print("[green]System is ready for MacFleet![/green]")
    else:
        console.print("[yellow]Some checks failed. See above for details.[/yellow]")


@cli.command()
@click.argument("script", required=False)
@click.option("--engine", type=click.Choice(["torch", "mlx"]), default="torch")
@click.option("--epochs", default=10, help="Number of training epochs")
@click.option("--batch-size", default=128, help="Global batch size")
@click.option("--lr", default=0.001, help="Learning rate")
@click.option("--compression", default="none", help="Compression: none, topk, fp16, topk_fp16")
@click.option("--config", "config_path", default=None, help="YAML config file")
def train(
    script: str | None,
    engine: str,
    epochs: int,
    batch_size: int,
    lr: float,
    compression: str,
    config_path: str | None,
):
    """Submit a training job to the pool.

    If SCRIPT is provided, it is executed as a Python file that defines
    `model` and `dataset` variables. Otherwise, runs a built-in demo
    (small MLP on synthetic data) useful for testing the pipeline.
    """
    if script:
        _train_from_script(script, engine, epochs, batch_size, lr, compression)
    else:
        _train_demo(engine, epochs, batch_size, lr)


def _train_demo(engine_type: str, epochs: int, batch_size: int, lr: float):
    """Run a built-in demo training on synthetic data (single-node)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from macfleet.engines.torch_engine import TorchEngine
    from macfleet.training.data_parallel import DataParallel
    from macfleet.comm.collectives import CollectiveGroup
    from macfleet.comm.transport import PeerTransport

    console.print("[bold blue]MacFleet Demo Training[/bold blue]")
    console.print("[dim]Single-node training on synthetic data (no peers needed)[/dim]\n")

    # Synthetic classification: 4 features, 2 classes
    torch.manual_seed(42)
    n_samples = 1000
    X = torch.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Simple MLP
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    eng = TorchEngine(device="cpu")
    eng.load_model(model, optimizer)

    console.print(f"  Model params: {eng.param_count():,}")
    console.print(f"  Dataset size: {n_samples}")
    console.print(f"  Batch size:   {batch_size}")
    console.print(f"  Epochs:       {epochs}")
    console.print(f"  Device:       {eng.device}\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_x, batch_y in dataloader:
            eng.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            eng.backward(loss)
            eng.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(1) == batch_y).sum().item()
            total += len(batch_y)

        elapsed = time.time() - t0
        acc = correct / total * 100
        avg_loss = epoch_loss / max(len(dataloader), 1)
        console.print(
            f"  Epoch {epoch + 1:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  acc={acc:.1f}%  "
            f"time={elapsed:.2f}s"
        )

    console.print("\n[green]Training complete![/green]")
    console.print("[dim]To train across multiple Macs, use the Python SDK:[/dim]")
    console.print("[dim]  macfleet.Pool().train(model, dataset, epochs=10)[/dim]")


def _train_from_script(
    script: str,
    engine_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    compression: str,
):
    """Run a user-provided training script."""
    import importlib.util
    import os

    if not os.path.isfile(script):
        console.print(f"[red]Error: Script not found: {script}[/red]")
        sys.exit(1)

    console.print(f"[bold blue]MacFleet Training[/bold blue] — {script}")

    # Load user script
    spec = importlib.util.spec_from_file_location("user_train", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Expect the script to define a `main()` function or `model`/`dataset`
    if hasattr(module, "main"):
        module.main()
    else:
        console.print("[red]Error: Script must define a main() function.[/red]")
        console.print("[dim]Example:[/dim]")
        console.print("[dim]  def main():[/dim]")
        console.print("[dim]      model = MyModel()[/dim]")
        console.print("[dim]      macfleet.train(model, dataset, epochs=10)[/dim]")
        sys.exit(1)


@cli.command()
@click.option("--type", "bench_type", type=click.Choice(["network", "compute", "allreduce"]), default="network")
@click.option("--size-mb", default=10, help="Payload size in MB for network tests")
@click.option("--iterations", default=5, help="Number of iterations")
def bench(bench_type: str, size_mb: int, iterations: int):
    """Benchmark network and compute performance."""
    if bench_type == "compute":
        _bench_compute(iterations)
    elif bench_type == "network":
        _bench_network(size_mb, iterations)
    elif bench_type == "allreduce":
        _bench_allreduce(size_mb, iterations)


def _bench_compute(iterations: int):
    """Benchmark local compute throughput."""
    import torch
    import torch.nn as nn

    console.print("[bold blue]MacFleet Compute Benchmark[/bold blue]\n")

    from macfleet.engines.torch_engine import TorchEngine

    eng = TorchEngine(device="cpu")
    model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    eng.load_model(model, optimizer)

    console.print(f"  Model: 2-layer MLP ({eng.param_count():,} params)")
    console.print(f"  Device: {eng.device}")
    console.print(f"  Iterations: {iterations}\n")

    # Warmup
    x = torch.randn(64, 512)
    for _ in range(3):
        eng.zero_grad()
        loss = model(x).sum()
        eng.backward(loss)
        eng.step()

    # Benchmark
    times = []
    for i in range(iterations):
        x = torch.randn(64, 512)
        t0 = time.perf_counter()
        eng.zero_grad()
        loss = model(x).sum()
        eng.backward(loss)
        eng.step()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        console.print(f"  Step {i + 1}: {elapsed * 1000:.1f} ms")

    import numpy as np

    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    console.print(f"\n  [bold]Average: {avg:.1f} ms/step (std={std:.1f} ms)[/bold]")
    console.print(f"  Throughput: {64 / np.mean(times):.0f} samples/sec")


def _bench_network(size_mb: int, iterations: int):
    """Benchmark loopback network throughput."""
    import numpy as np

    console.print("[bold blue]MacFleet Network Benchmark[/bold blue]\n")
    console.print(f"  Payload: {size_mb} MB, loopback")
    console.print(f"  Iterations: {iterations}\n")

    from macfleet.comm.transport import PeerTransport, TransportConfig

    config = TransportConfig(recv_timeout_sec=30.0, connect_timeout_sec=10.0)

    async def run():
        server = PeerTransport(local_id="bench-server", config=config)
        client = PeerTransport(local_id="bench-client", config=config)

        await server.start_server("127.0.0.1", 0)
        port = server._server.sockets[0].getsockname()[1]
        await client.connect("bench-server", "127.0.0.1", port)
        await asyncio.sleep(0.1)

        payload = bytes(range(256)) * (size_mb * 1024 * 1024 // 256)
        times = []

        for i in range(iterations):
            t0 = time.perf_counter()
            await client.send("bench-server", payload)
            await server.recv("bench-client")
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            throughput = size_mb / elapsed
            console.print(f"  Transfer {i + 1}: {elapsed * 1000:.1f} ms ({throughput:.0f} MB/s)")

        await client.disconnect_all()
        await server.disconnect_all()

        avg_time = np.mean(times)
        avg_throughput = size_mb / avg_time
        console.print(f"\n  [bold]Average: {avg_throughput:.0f} MB/s[/bold]")

    asyncio.run(run())


def _bench_allreduce(size_mb: int, iterations: int):
    """Benchmark AllReduce over loopback (simulates 2-node)."""
    import numpy as np

    console.print("[bold blue]MacFleet AllReduce Benchmark (2-node loopback)[/bold blue]\n")
    console.print(f"  Array size: {size_mb} MB")
    console.print(f"  Iterations: {iterations}\n")

    from macfleet.comm.collectives import CollectiveGroup
    from macfleet.comm.transport import PeerTransport, TransportConfig

    config = TransportConfig(recv_timeout_sec=30.0, connect_timeout_sec=10.0)

    async def run():
        # Setup 2-node mesh
        t0_transport = PeerTransport(local_id="node-0", config=config)
        t1_transport = PeerTransport(local_id="node-1", config=config)

        await t1_transport.start_server("127.0.0.1", 0)
        port = t1_transport._server.sockets[0].getsockname()[1]
        await t0_transport.connect("node-1", "127.0.0.1", port)
        await asyncio.sleep(0.1)

        group0 = CollectiveGroup(rank=0, world_size=2, transport=t0_transport, rank_to_peer={1: "node-1"})
        group1 = CollectiveGroup(rank=1, world_size=2, transport=t1_transport, rank_to_peer={0: "node-0"})

        # Create arrays
        n_floats = size_mb * 1024 * 1024 // 4
        arr0 = np.random.randn(n_floats).astype(np.float32)
        arr1 = np.random.randn(n_floats).astype(np.float32)

        times = []
        for i in range(iterations):
            t0 = time.perf_counter()
            await asyncio.gather(
                group0.allreduce(arr0, op="mean"),
                group1.allreduce(arr1, op="mean"),
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            console.print(f"  AllReduce {i + 1}: {elapsed * 1000:.1f} ms")

        await t0_transport.disconnect_all()
        await t1_transport.disconnect_all()

        avg = np.mean(times) * 1000
        console.print(f"\n  [bold]Average AllReduce: {avg:.1f} ms[/bold]")
        console.print(f"  Effective bandwidth: {size_mb * 2 / np.mean(times):.0f} MB/s")

    asyncio.run(run())


if __name__ == "__main__":
    cli()
