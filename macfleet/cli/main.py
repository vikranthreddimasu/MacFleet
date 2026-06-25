"""MacFleet CLI: zero-config compute pool for Apple Silicon Macs.

Commands:
    macfleet join        Join the compute pool
    macfleet leave       Leave the pool gracefully
    macfleet status      Show pool members and network info
    macfleet info        Show local hardware info
    macfleet train       Submit a training job
    macfleet bench       Benchmark network + compute
    macfleet doctor      System health check (alias: diagnose)
    macfleet quickstart  Write a starter training script
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time

import click
from rich.console import Console
from rich.table import Table

import macfleet

console = Console()


@click.group()
@click.version_option(version=macfleet.__version__, prog_name="macfleet")
def cli():
    """MacFleet: Pool Apple Silicon Macs for distributed ML training."""
    pass


def _best_pairing_host() -> str:
    """Return a reachable LAN address for enrollment instructions."""
    try:
        from macfleet.pool.network import LinkType, get_network_topology

        topology = get_network_topology()
        best = topology.best_link
        if best and best.link_type != LinkType.LOOPBACK:
            return best.ip_address
    except Exception:
        pass
    return "127.0.0.1"


@cli.command()
@click.option("--name", default=None, help="Custom node name")
@click.option("--port", default=50051, help="Heartbeat / discovery port")
@click.option("--data-port", default=50052, help="Data transport port (training)")
@click.option("--token", default=None, envvar="MACFLEET_TOKEN", help="Pool token (or set MACFLEET_TOKEN env var)")
@click.option("--fleet-id", default=None, help="Fleet identifier (isolates pool on network)")
@click.option(
    "--tls", "use_tls", is_flag=True, default=False,
    help="Enable TLS encryption (no-op when --token is set: TLS is "
         "already mandatory in secure mode)",
)
@click.option(
    "--open", "open_fleet", is_flag=True, default=False,
    help="Disable security (open fleet, no authentication)",
)
@click.option(
    "--allow-insecure-open",
    is_flag=True,
    default=False,
    help="Required with --open. Makes unauthenticated LAN exposure explicit.",
)
@click.option("--peer", "peers", multiple=True, help="Peer address (IP:PORT). Use when mDNS is blocked. Repeatable.")
@click.option(
    "--bootstrap", is_flag=True, default=False,
    help="Start a short-lived one-time enrollment endpoint for pairing.",
)
@click.option(
    "--enroll-ttl",
    default=300.0,
    show_default=True,
    help="Seconds that the one-time enrollment code remains valid.",
)
@click.option(
    "--enroll-uses",
    default=1,
    show_default=True,
    help="Number of Macs that may pair with this enrollment code.",
)
@click.option(
    "--show-token",
    is_flag=True,
    default=False,
    help="Reveal the permanent fleet token in the terminal (dangerous).",
)
def join(
    name: str | None, port: int, data_port: int, token: str | None, fleet_id: str | None,
    use_tls: bool, open_fleet: bool, allow_insecure_open: bool, peers: tuple,
    bootstrap: bool, enroll_ttl: float, enroll_uses: int, show_token: bool,
):
    """Join the compute pool. Auto-discovers peers on the network.

    Security is enabled by default. A fleet token is auto-generated on first
    run and saved to ~/.macfleet/fleet-token. Pairing uses a short-lived
    one-time code so the permanent token does not need to leave this Mac.

    Use --open --allow-insecure-open to disable security (not recommended).

    \b
    If mDNS discovery doesn't work (e.g. enterprise WiFi), use --peer:
        Mac A: macfleet join
        Mac B: macfleet join --token <token> --peer <Mac-A-IP>:50051

    \b
    For 5-second cross-Mac pairing, use --bootstrap:
        Mac A: macfleet join --bootstrap
        Mac B: macfleet pair --host <Mac-A-IP>:<port> --code <code>
        Mac B: macfleet join
    """
    import os

    from macfleet.pool.agent import PoolAgent
    from macfleet.security.audit import audit_event
    from macfleet.security.auth import TOKEN_ENV_VAR, TOKEN_FILE, resolve_token_with_file
    from macfleet.security.enrollment import EnrollmentServer, print_enrollment_info

    start_enrollment = False
    if open_fleet:
        if token:
            console.print("[red]Error: --open and --token are mutually exclusive.[/red]")
            sys.exit(1)
        if bootstrap:
            console.print("[red]Error: --bootstrap requires a token (can't pair an open fleet).[/red]")
            sys.exit(1)
        if not allow_insecure_open:
            console.print(
                "[red]Error: --open disables authentication. Re-run with "
                "--open --allow-insecure-open if you really want an open LAN fleet.[/red]"
            )
            sys.exit(1)
        audit_event("fleet.open_mode_enabled", port=port, data_port=data_port)
        resolved_token = None
    else:
        # First run on this Mac? (No explicit token, no env var, no saved
        # file → resolve_token_with_file is about to mint a fresh token.)
        # Show the pairing block automatically so pairing a second Mac
        # needs zero extra flags.
        first_run = (
            token is None
            and os.environ.get(TOKEN_ENV_VAR) is None
            and not os.path.exists(TOKEN_FILE)
        )
        resolved_token = resolve_token_with_file(token, auto_generate=True)
        if use_tls:
            console.print(
                "[dim]--tls is implicit when a token is set (already enforced "
                "by SecurityConfig).[/dim]"
            )
        if token is None:
            console.print("\n[bold green]Fleet token configured[/bold green]")
            console.print(f"[dim]Saved to {TOKEN_FILE}[/dim]")
            if show_token:
                console.print(f"[bold yellow]Permanent fleet token:[/bold yellow] {resolved_token}")
                audit_event("token.revealed", source="join_show_token")
            else:
                console.print(
                    "[dim]Token hidden. Use `macfleet rotate-token` if it was exposed.[/dim]"
                )

        if bootstrap or first_run:
            if resolved_token is None:
                console.print("[red]Cannot show pairing info: no fleet token available.[/red]")
                raise SystemExit(1)
            start_enrollment = True
        elif token is None:
            console.print(
                "[dim]Pair another Mac: run `macfleet join --bootstrap` here, "
                "then the printed `macfleet pair --host ... --code ...` command there.[/dim]\n"
            )

    agent = PoolAgent(
        name=name, port=port, data_port=data_port,
        token=resolved_token, fleet_id=fleet_id, tls=use_tls,
        peers=list(peers),
    )

    async def run():
        enrollment_server = None
        await agent.start()
        if start_enrollment and resolved_token is not None:
            enrollment_server = EnrollmentServer(
                token=resolved_token,
                fleet_id=fleet_id,
                node_id=agent.node_id,
                ttl_sec=enroll_ttl,
                max_uses=enroll_uses,
            )
            await enrollment_server.start()
            print_enrollment_info(
                _best_pairing_host(),
                enrollment_server.bound_port,
                enrollment_server.code,
                enrollment_server.expires_at_epoch,
                out=sys.stdout,
            )
        console.print("\n[dim]Press Ctrl+C to leave the pool[/dim]\n")

        # Wait for interrupt
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        try:
            await stop_event.wait()
        finally:
            if enrollment_server is not None:
                await enrollment_server.stop()
            await agent.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


@cli.command()
def info():
    """Show local hardware information."""
    from macfleet.monitoring.thermal import get_thermal_state, thermal_state_to_string
    from macfleet.pool.agent import profile_hardware
    from macfleet.pool.network import get_network_topology

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
@click.option("--open", "open_fleet", is_flag=True, default=False, help="Scan open fleet (ignore saved token)")
def status(token: str | None, fleet_id: str | None, open_fleet: bool):
    """Show pool status (discovers peers for 3 seconds)."""
    from macfleet.pool.discovery import ServiceRegistry
    from macfleet.security.auth import SecurityConfig, resolve_token_with_file

    if open_fleet:
        resolved = None
    else:
        resolved = resolve_token_with_file(token)

    sec = SecurityConfig(token=resolved, fleet_id=fleet_id) if resolved else None
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
    table.add_column("IP / heartbeat / data")
    table.add_column("Score", justify="right")

    for node in sorted(peers, key=lambda n: -n.compute_score):
        table.add_row(
            node.hostname,
            node.chip_name,
            str(node.gpu_cores),
            str(node.ram_gb),
            f"{node.ip_address} :{node.port} :{node.data_port}",
            f"{node.compute_score:.0f}",
        )

    console.print(table)


@cli.command()
def diagnose():
    """Run system health checks."""
    from macfleet.monitoring.thermal import get_thermal_state
    from macfleet.pool.agent import _check_mlx_available, _check_mps_available, profile_hardware
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

    # Security
    console.print("\n[bold]Security[/bold]")
    import os
    import stat as stat_mod

    from macfleet.security.auth import (
        RECOMMENDED_TOKEN_LENGTH,
        TOKEN_FILE,
        resolve_token_with_file,
    )

    tok = resolve_token_with_file(None)
    check(
        "Fleet token configured",
        tok is not None,
        "" if tok else "run 'macfleet join' once to auto-generate",
    )
    if tok is not None:
        check(
            f"Token length >= {RECOMMENDED_TOKEN_LENGTH}",
            len(tok) >= RECOMMENDED_TOKEN_LENGTH,
            f"{len(tok)} chars" + (
                "" if len(tok) >= RECOMMENDED_TOKEN_LENGTH
                else " — short tokens are dictionary-attackable"
            ),
        )
    if os.path.exists(TOKEN_FILE):
        mode = stat_mod.S_IMODE(os.stat(TOKEN_FILE).st_mode)
        check(
            "Token file private (0600)",
            (mode & 0o077) == 0,
            f"mode {oct(mode)}" + (
                "" if (mode & 0o077) == 0 else f" — fix: chmod 600 {TOKEN_FILE}"
            ),
        )

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
    if spec is None or spec.loader is None:
        console.print(f"[red]Error: Cannot load script '{script}' as a Python module.[/red]")
        sys.exit(1)
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


@cli.command(name="run")
@click.argument("script")
@click.option("--fn", "fn_name", default="main", help="Function to execute (default: main)")
@click.option("--token", default=None, envvar="MACFLEET_TOKEN", help="Pool token")
@click.option("--open", "open_fleet", is_flag=True, default=False, help="Disable security")
def run_command(script: str, fn_name: str, token: str | None, open_fleet: bool):
    """Run a Python script on the pool.

    The script must define the named function (default: main).
    The function is executed across the pool's compute resources.

    \b
    Examples:
        macfleet run process.py
        macfleet run analysis.py --fn analyze
    """
    import importlib.util
    import os

    if not os.path.isfile(script):
        console.print(f"[red]Error: Script not found: {script}[/red]")
        sys.exit(1)

    # Load user script
    spec = importlib.util.spec_from_file_location("user_script", script)
    if spec is None or spec.loader is None:
        console.print(f"[red]Error: Cannot load script '{script}' as a Python module.[/red]")
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        console.print(f"[red]Error: Function '{fn_name}' not found in {script}[/red]")
        console.print(f"[dim]The script must define a callable named '{fn_name}'.[/dim]")
        sys.exit(1)

    console.print(f"[bold blue]MacFleet Run[/bold blue] — {script}:{fn_name}()")

    from macfleet.sdk.pool import Pool

    with Pool(token=token, open=open_fleet) as pool:
        t0 = time.time()
        result = pool.run(fn)
        elapsed = time.time() - t0

    console.print(f"\n[green]Completed in {elapsed:.2f}s[/green]")
    if result is not None:
        console.print(f"Result: {result}")


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


@cli.command()
@click.option(
    "--stdin", "from_stdin", is_flag=True, default=False,
    help="Read a legacy token-bearing pairing URL from stdin.",
)
@click.option(
    "--pasteboard",
    "from_pasteboard",
    is_flag=True,
    default=False,
    help="Read a legacy token-bearing pairing URL from the macOS pasteboard.",
)
@click.option(
    "--host",
    "enroll_host",
    default=None,
    help="Enrollment server in HOST:PORT form from `macfleet join --bootstrap`.",
)
@click.option(
    "--code",
    "enroll_code",
    default=None,
    help="One-time enrollment code from `macfleet join --bootstrap`.",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Do not prompt before replacing an existing saved fleet token.",
)
def pair(
    from_stdin: bool,
    from_pasteboard: bool,
    enroll_host: str | None,
    enroll_code: str | None,
    yes: bool,
):
    """Pair this Mac with an existing fleet.

    Preferred flow: use the short-lived host/code printed by
    `macfleet join --bootstrap`. Legacy `macfleet://pair?token=...` URLs
    are still accepted from explicitly requested pasteboard or stdin input
    so existing setups can migrate.

    \b
    Typical flow:
        Mac #1: macfleet join --bootstrap
        Mac #2: macfleet pair --host <Mac-A-IP>:<port> --code <code>
        Mac #2: macfleet join
    """
    import os

    from macfleet.security.audit import audit_event
    from macfleet.security.auth import TOKEN_FILE, _write_token_file
    from macfleet.security.bootstrap import (
        PairingError,
        parse_pairing_url,
        read_from_pasteboard,
    )
    from macfleet.security.enrollment import (
        EnrollmentError,
        parse_host_port,
        request_enrollment,
    )

    def confirm_token_replacement(*, can_prompt: bool = True) -> None:
        if yes or not os.path.lexists(TOKEN_FILE):
            return
        if not can_prompt:
            console.print(
                "[red]Error: refusing to replace an existing saved fleet token from --stdin.[/red]\n"
                "[dim]Re-run with --yes if this migration should overwrite the saved token.[/dim]"
            )
            raise click.exceptions.Exit(1)
        confirmed = click.confirm(
            "Replace the saved fleet token on this Mac? "
            "This Mac will need to rejoin peers from the new fleet.",
            default=False,
        )
        if not confirmed:
            console.print("[yellow]Pairing cancelled; existing fleet token left unchanged.[/yellow]")
            raise click.exceptions.Exit(1)

    def write_pairing_token(new_token: str) -> None:
        try:
            _write_token_file(new_token)
        except OSError as e:
            console.print(f"[red]Error: couldn't write fleet token to {TOKEN_FILE}: {e}[/red]")
            sys.exit(1)

    if enroll_host or enroll_code:
        if from_stdin or from_pasteboard:
            console.print(
                "[red]Error: legacy URL input cannot be combined with --host/--code.[/red]"
            )
            sys.exit(1)
        if not enroll_host or not enroll_code:
            console.print("[red]Error: --host and --code must be provided together.[/red]")
            sys.exit(1)
        confirm_token_replacement()
        try:
            host, port = parse_host_port(enroll_host)
            result = asyncio.run(request_enrollment(host, port, enroll_code))
        except (EnrollmentError, OSError, asyncio.TimeoutError) as e:
            console.print(f"[red]Error: enrollment failed: {e}[/red]")
            sys.exit(1)
        write_pairing_token(result.token)
        audit_event(
            "pairing.completed",
            mode="enrollment",
            fleet_id=result.fleet_id,
            server_node=result.server_node,
        )
        console.print(
            f"[green]Paired.[/green] Token written to {TOKEN_FILE}"
            + (f" (fleet: [bold]{result.fleet_id}[/bold])" if result.fleet_id else "")
            + f"\n[dim]Server: {result.server_node}. Next: macfleet join[/dim]"
        )
        return

    if from_stdin and from_pasteboard:
        console.print("[red]Error: choose only one legacy URL input source.[/red]")
        sys.exit(1)

    if from_stdin:
        url = sys.stdin.read().strip()
        if not url:
            console.print("[red]Error: no URL on stdin.[/red]")
            sys.exit(1)
    elif from_pasteboard:
        url = read_from_pasteboard()
        if not url:
            console.print(
                "[red]Error: couldn't read pairing URL from pasteboard.[/red]\n"
                "[dim]Preferred: run `macfleet join --bootstrap` on the first Mac, "
                "then paste the printed `macfleet pair --host ... --code ...` command here.[/dim]\n"
                "[dim]Legacy URL fallback: echo 'macfleet://pair?token=...' | macfleet pair --stdin[/dim]"
            )
            sys.exit(1)
        url = url.strip()
    else:
        console.print(
            "[red]Error: pairing requires an explicit input source.[/red]\n"
            "[dim]Preferred: run `macfleet join --bootstrap` on the first Mac, "
            "then run the printed `macfleet pair --host ... --code ...` command here.[/dim]\n"
            "[dim]Legacy URL migration: use --stdin or --pasteboard explicitly.[/dim]"
        )
        sys.exit(1)

    try:
        token, fleet_id = parse_pairing_url(url)
    except PairingError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(
            "[dim]Expected format: macfleet://pair?token=<token>&fleet=<id>[/dim]"
        )
        sys.exit(1)

    confirm_token_replacement(can_prompt=not from_stdin)
    write_pairing_token(token)
    audit_event("pairing.completed", mode="legacy_url", fleet_id=fleet_id)
    console.print(
        f"[green]Paired from legacy token URL.[/green] Token written to {TOKEN_FILE}"
        + (f" (fleet: [bold]{fleet_id}[/bold])" if fleet_id else "")
        + "\n[dim]Next: macfleet join[/dim]"
    )


@cli.command("rotate-token")
@click.option("--yes", is_flag=True, default=False, help="Do not prompt before replacing an existing token.")
@click.option("--show-token", is_flag=True, default=False, help="Reveal the new permanent token in the terminal.")
def rotate_token(yes: bool, show_token: bool):
    """Rotate the saved fleet token on this Mac.

    Rotation invalidates future joins that use the old token. Restart running
    MacFleet agents and re-pair other Macs with `macfleet join --bootstrap`.
    """
    import os

    from macfleet.security.audit import audit_event
    from macfleet.security.auth import TOKEN_FILE, rotate_saved_fleet_token

    if os.path.exists(TOKEN_FILE) and not yes:
        confirmed = click.confirm(
            "Replace the saved fleet token on this Mac? "
            "Running agents and other Macs must be restarted/re-paired.",
            default=False,
        )
        if not confirmed:
            console.print("[yellow]Token rotation cancelled.[/yellow]")
            raise click.exceptions.Exit(1)

    token = rotate_saved_fleet_token()
    console.print(f"[green]Rotated fleet token.[/green] Saved to {TOKEN_FILE}")
    if show_token:
        console.print(f"[bold yellow]New permanent fleet token:[/bold yellow] {token}")
        audit_event("token.revealed", source="rotate_token_show_token")
    else:
        console.print("[dim]Token hidden. Run `macfleet join --bootstrap` to pair other Macs safely.[/dim]")
    console.print("[dim]Restart MacFleet on every Mac and re-pair peers with the new token.[/dim]")


# v2.2 PR 16 (D10): `macfleet doctor` is a friendlier alias for `diagnose`.
# Users trained by `brew doctor` / `rustup doctor` look for this name first.
@cli.command()
@click.pass_context
def doctor(ctx):
    """System health check (alias for `diagnose`)."""
    # Delegate through Click's context so diagnose runs with a proper context
    # (obj, exception handling) instead of a bare callback() call.
    ctx.invoke(diagnose)


# v2.2 PR 16 (D10): `macfleet quickstart` scaffolds a demo training script.
# Goal: 5 seconds from `pip install macfleet` to `python my_macfleet_demo.py`
# to first loss going down. First-run success is the north star.
QUICKSTART_TEMPLATE = '''"""MacFleet quickstart: a 30-line distributed-training demo.

Written by `macfleet quickstart`. Run it:

    python {filename}

If you've paired a second Mac via `macfleet join --bootstrap` + `macfleet pair`,
set `enable_pool_distributed=True` below and run THIS SAME SCRIPT on both Macs
to spread training across them (each Mac becomes one rank; gradients are
averaged every step).
"""

import macfleet
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """A 2-layer MLP — intentionally small so this demo finishes fast."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    torch.manual_seed(42)
    X = torch.randn(1000, 4)
    y = (X.sum(dim=1) > 0).long()  # linearly separable toy task

    with macfleet.Pool(
        engine="torch",
        # Flip to True after pairing a second Mac (see docs/getting-started/pairing.md)
        enable_pool_distributed=False,
    ) as pool:
        print(f"Pool world size: {{pool.world_size}}")
        result = pool.train(
            model=TinyMLP(),
            dataset=(X, y),
            epochs=10,
            batch_size=64,
            lr=0.01,
            loss_fn=nn.CrossEntropyLoss(),
        )
        print("Training done:", result)


if __name__ == "__main__":
    main()
'''


@cli.command()
@click.option(
    "--output", "-o",
    default="my_macfleet_demo.py",
    help="Target filename for the generated demo.",
)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Overwrite the target file if it already exists.",
)
def quickstart(output: str, force: bool):
    """Write a starter training script to get you running in <1 minute."""
    from pathlib import Path

    target = Path(output)
    if target.exists() and not force:
        console.print(
            f"[yellow]{target} already exists.[/yellow] "
            f"Pass --force to overwrite, or pick another filename with --output."
        )
        raise click.exceptions.Exit(1)

    content = QUICKSTART_TEMPLATE.format(filename=target.name)
    target.write_text(content)
    console.print(
        f"[green]Wrote {target} ({len(content)} bytes)[/green]\n"
        f"\nNext steps:\n"
        f"  1. Install torch: [bold]pip install 'macfleet\\[torch]'[/bold]\n"
        f"  2. Run it: [bold]python {target}[/bold]\n"
        f"  3. Pair a second Mac: [bold]macfleet join --bootstrap[/bold]\n"
        f"     (see docs/getting-started/pairing.md)\n"
    )


if __name__ == "__main__":
    cli()
