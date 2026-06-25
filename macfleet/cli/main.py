"""Command-line interface for MacFleet.

Provides the `macfleet` CLI with commands for:
- launch: Start a coordinator or worker node
- status: Check cluster status
- benchmark: Run performance benchmarks
"""

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from macfleet import __version__
from macfleet.core.config import (
    DEFAULT_AUTH_TOKEN_ENV,
    ClusterConfig,
    NodeRole,
    resolve_auth_token,
)
from macfleet.utils.network import (
    is_port_available,
    is_reachable,
    parse_endpoint,
    validate_ip_address,
    validate_port,
)

console = Console()


def _parse_endpoint_option(value: str, default_port: int, option_name: str) -> tuple[str, int]:
    """Parse a CLI endpoint and surface validation as a Click error."""
    try:
        return parse_endpoint(value, default_port, option_name)
    except ValueError as exc:
        raise click.ClickException(f"Invalid {option_name}: {exc}") from exc


def _parse_size_list(value: str) -> list[int]:
    """Parse comma-separated positive integer sizes."""
    sizes: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            raise click.ClickException("Invalid --sizes: entries must not be empty")
        try:
            size = int(item)
        except ValueError as exc:
            raise click.ClickException(
                f"Invalid --sizes: {item!r} is not an integer"
            ) from exc
        if size <= 0:
            raise click.ClickException("Invalid --sizes: entries must be positive")
        sizes.append(size)
    return sizes


@click.group()
@click.version_option(version=__version__, prog_name="macfleet")
def cli():
    """MacFleet: Distributed ML training across Apple Silicon Macs.

    Use 'macfleet launch' to start a node, 'macfleet status' to check
    cluster status, and 'macfleet benchmark' to run performance tests.
    """
    pass


@cli.command()
@click.option(
    "--role",
    type=click.Choice(["master", "worker"]),
    required=True,
    help="Role of this node in the cluster.",
)
@click.option(
    "--port",
    type=int,
    default=50051,
    help="gRPC port for control messages (default: 50051).",
)
@click.option(
    "--tensor-port",
    type=int,
    default=50052,
    help="Port for tensor transfers (default: 50052).",
)
@click.option(
    "--master",
    type=str,
    default=None,
    help="Master address for workers (e.g., 10.0.0.1 or 10.0.0.1:50051).",
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="IP address to bind to (e.g., 169.254.83.200 for Thunderbolt bridge).",
)
@click.option(
    "--no-discovery",
    is_flag=True,
    help="Disable Bonjour/zeroconf discovery.",
)
@click.option(
    "--auth-token-env",
    default=DEFAULT_AUTH_TOKEN_ENV,
    show_default=True,
    help="Environment variable containing the shared control-plane token.",
)
def launch(
    role: str,
    port: int,
    tensor_port: int,
    master: str,
    host: str,
    no_discovery: bool,
    auth_token_env: str,
):
    """Launch a MacFleet node (coordinator or worker).

    Examples:

        # On MacBook Pro (master), bind to Thunderbolt IP:
        macfleet launch --role master --host 169.254.83.200 --port 50051

        # On MacBook Air (worker):
        macfleet launch --role worker --master 169.254.83.200 --port 50051

        # Worker with specific ports:
        macfleet launch --role worker --master 10.0.0.1:50051 --tensor-port 50053
    """
    # Parse master address
    master_addr = "10.0.0.1"
    master_port = port

    if master:
        master_addr, master_port = _parse_endpoint_option(master, port, "--master")

    # Create cluster config
    cluster_config = ClusterConfig(
        role=NodeRole.MASTER if role == "master" else NodeRole.WORKER,
        master_addr=master_addr,
        master_port=master_port,
        tensor_port=tensor_port,
        discovery_enabled=not no_discovery,
        host=host,
        auth_token_env=auth_token_env,
    )

    # Print banner
    console.print()
    console.print("[bold blue]╔══════════════════════════════════════╗[/bold blue]")
    console.print(
        "[bold blue]║[/bold blue]     "
        "[bold white]MacFleet[/bold white] - Distributed Training   "
        "[bold blue]║[/bold blue]"
    )
    console.print(
        "[bold blue]║[/bold blue]     "
        "[dim]Apple Silicon over Thunderbolt[/dim]    "
        "[bold blue]║[/bold blue]"
    )
    console.print("[bold blue]╚══════════════════════════════════════╝[/bold blue]")
    console.print()

    try:
        if role == "master":
            _run_coordinator(cluster_config)
        else:
            _run_worker(cluster_config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        sys.exit(1)


def _run_coordinator(config: ClusterConfig):
    """Run the coordinator node."""
    from macfleet.core.coordinator import Coordinator

    coordinator = Coordinator(cluster_config=config)
    asyncio.run(coordinator.run())


def _run_worker(config: ClusterConfig):
    """Run the worker node."""
    from macfleet.core.worker import Worker

    worker = Worker(cluster_config=config)
    asyncio.run(worker.run())


@cli.command()
@click.option(
    "--master",
    type=str,
    default="10.0.0.1:50051",
    help="Master address (default: 10.0.0.1:50051).",
)
@click.option(
    "--auth-token-env",
    default=DEFAULT_AUTH_TOKEN_ENV,
    show_default=True,
    help="Environment variable containing the shared control-plane token.",
)
def status(master: str, auth_token_env: str):
    """Check the status of a MacFleet cluster.

    Connects to the coordinator and displays cluster information.

    Example:

        macfleet status --master 10.0.0.1:50051
    """
    from macfleet.comm.grpc_service import ClusterControlClient

    # Parse master address
    master_addr, master_port = _parse_endpoint_option(master, 50051, "--master")

    console.print(f"Connecting to {master_addr}:{master_port}...")

    try:
        client = ClusterControlClient(
            master_addr,
            master_port,
            auth_token_env=auth_token_env,
        )
        client.connect()
        state = client.get_cluster_state()
        client.disconnect()

        # Display cluster state
        console.print()
        console.print("[bold green]Cluster Status[/bold green]")
        console.print(f"  World Size: {state['world_size']}")
        console.print(f"  Training: {state['training_status']}")

        if state['training_active']:
            console.print(f"  Epoch: {state['current_epoch']}")
            console.print(f"  Step: {state['current_step']}")

        console.print()

        # Node table
        table = Table(title="Nodes")
        table.add_column("Rank", style="cyan")
        table.add_column("Hostname", style="green")
        table.add_column("IP Address")
        table.add_column("GPU Cores", justify="right")
        table.add_column("RAM (GB)", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("Status", style="yellow")

        for node in state['nodes']:
            table.add_row(
                str(node['rank']),
                node['hostname'],
                node['ip_address'],
                str(node['gpu_cores']),
                str(node['ram_gb']),
                f"{node['workload_weight']:.1%}",
                node['status'],
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        console.print("[yellow]Make sure the coordinator is running.[/yellow]")
        sys.exit(1)


@cli.command()
@click.option(
    "--type",
    "bench_type",
    type=click.Choice(["bandwidth", "allreduce", "latency"]),
    default="bandwidth",
    help="Type of benchmark to run.",
)
@click.option(
    "--master",
    type=str,
    default=None,
    help="Master address for distributed benchmarks.",
)
@click.option(
    "--sizes",
    type=str,
    default="1,10,50,100,500",
    help="Comma-separated tensor sizes in MB (default: 1,10,50,100,500).",
)
def benchmark(bench_type: str, master: str, sizes: str):
    """Run MacFleet performance benchmarks.

    Examples:

        # Test local bandwidth:
        macfleet benchmark --type bandwidth

        # Test bandwidth with specific sizes:
        macfleet benchmark --type bandwidth --sizes 10,50,100

        # Test AllReduce with master:
        macfleet benchmark --type allreduce --master 10.0.0.1
    """
    console.print(f"[bold blue]Running {bench_type} benchmark...[/bold blue]")
    console.print()

    # Parse sizes
    size_list = _parse_size_list(sizes)

    if bench_type == "bandwidth":
        _run_bandwidth_benchmark(size_list)
    elif bench_type == "latency":
        _run_latency_benchmark()
    elif bench_type == "allreduce":
        _run_allreduce_benchmark(size_list)


def _run_bandwidth_benchmark(sizes_mb: list[int]):
    """Run local bandwidth benchmark."""
    import time

    import torch

    console.print("Testing tensor serialization bandwidth...")
    console.print()

    from macfleet.utils.tensor_utils import bytes_to_tensor, tensor_to_bytes

    table = Table(title="Bandwidth Results")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Serialize (ms)", justify="right")
    table.add_column("Deserialize (ms)", justify="right")
    table.add_column("Throughput (GB/s)", justify="right")

    for size_mb in sizes_mb:
        # Create tensor
        numel = (size_mb * 1024 * 1024) // 4  # FP32 = 4 bytes
        tensor = torch.randn(numel)

        # Benchmark serialization
        start = time.perf_counter()
        data = tensor_to_bytes(tensor)
        serialize_time = (time.perf_counter() - start) * 1000

        # Benchmark deserialization
        start = time.perf_counter()
        tensor2, _ = bytes_to_tensor(data)
        deserialize_time = (time.perf_counter() - start) * 1000

        # Calculate throughput
        total_time_sec = (serialize_time + deserialize_time) / 1000
        throughput_gbps = (size_mb * 2) / (total_time_sec * 1024) if total_time_sec > 0 else 0

        table.add_row(
            str(size_mb),
            f"{serialize_time:.2f}",
            f"{deserialize_time:.2f}",
            f"{throughput_gbps:.2f}",
        )

    console.print(table)


def _run_latency_benchmark():
    """Run loopback latency benchmark."""
    import time

    console.print("Testing loopback latency...")
    console.print()

    async def run():
        import torch

        from macfleet.comm.transport import TensorTransport

        # Start server
        server = TensorTransport("127.0.0.1", 50099)
        received = []

        async def on_recv(tensor, msg_type, addr):
            received.append(time.perf_counter())

        await server.start_server(on_recv)

        # Connect and send
        client = TensorTransport()
        conn_key = await client.connect("127.0.0.1", 50099)

        latencies = []
        for _ in range(100):
            tensor = torch.randn(1000)  # Small tensor
            start = time.perf_counter()
            await client.send_tensor(tensor, conn_key)
            await asyncio.sleep(0.01)  # Wait for receipt
            if received:
                latency = (received[-1] - start) * 1000
                latencies.append(latency)

        await client.disconnect(conn_key)
        await server.stop_server()

        if latencies:
            avg = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            console.print(f"  Average latency: {avg:.2f} ms")
            console.print(f"  Min latency: {min_lat:.2f} ms")
            console.print(f"  Max latency: {max_lat:.2f} ms")
        else:
            console.print("[yellow]No latency measurements collected[/yellow]")

    asyncio.run(run())


def _run_allreduce_benchmark(sizes_mb: list[int]):
    """Run AllReduce benchmark using loopback simulation."""
    import time

    import torch

    console.print("Running loopback AllReduce benchmark...")
    console.print("Testing with no compression and TopK+FP16...")
    console.print()

    async def run():
        from macfleet.comm.collectives import AllReduce, CollectiveGroup
        from macfleet.comm.transport import TensorTransport

        port0, port1 = 50100, 50101
        t0 = TensorTransport("127.0.0.1", port0)
        t1 = TensorTransport("127.0.0.1", port1)
        await t0.start_server()
        await t1.start_server()

        g0 = CollectiveGroup(rank=0, world_size=2, transport=t0)
        g1 = CollectiveGroup(rank=1, world_size=2, transport=t1)
        await g0.connect_to_peer(1, "127.0.0.1", port1)
        await g1.connect_to_peer(0, "127.0.0.1", port0)

        table = Table(title="AllReduce Benchmark Results")
        table.add_column("Size (MB)", justify="right", style="cyan")
        table.add_column("Compression", style="yellow")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Throughput (Gbps)", justify="right", style="green")

        for comp in ["none", "topk_fp16"]:
            ar0 = AllReduce(g0)
            ar1 = AllReduce(g1)

            for size_mb in sizes_mb:
                numel = int((size_mb * 1024 * 1024) / 4)
                tensor0 = torch.randn(numel)
                tensor1 = torch.randn(numel)
                actual_mb = (numel * 4) / (1024 * 1024)

                latencies = []
                for _ in range(10):
                    start = time.perf_counter()
                    await asyncio.gather(
                        ar0(tensor0.clone()), ar1(tensor1.clone())
                    )
                    latencies.append((time.perf_counter() - start) * 1000)

                avg_lat = sum(latencies) / len(latencies)
                throughput = (actual_mb * 1024 * 1024 * 2 * 8) / (avg_lat / 1000 * 1e9)

                table.add_row(
                    f"{actual_mb:.1f}", comp,
                    f"{avg_lat:.2f}", f"{throughput:.2f}",
                )

        console.print(table)

        await g0.disconnect_all()
        await g1.disconnect_all()
        await t0.stop_server()
        await t1.stop_server()

    asyncio.run(run())


@cli.command()
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Local IP address to check for binding.",
)
@click.option(
    "--port",
    type=int,
    default=50051,
    show_default=True,
    help="gRPC control port to check.",
)
@click.option(
    "--tensor-port",
    type=int,
    default=50052,
    show_default=True,
    help="Tensor transfer port to check.",
)
@click.option(
    "--master",
    type=str,
    default=None,
    help="Optional coordinator endpoint to test (host or host:port).",
)
@click.option(
    "--auth-token-env",
    default=DEFAULT_AUTH_TOKEN_ENV,
    show_default=True,
    help="Environment variable containing the shared control-plane token.",
)
def diagnose(
    host: str,
    port: int,
    tensor_port: int,
    master: str,
    auth_token_env: str,
):
    """Run local readiness checks for a MacFleet node."""
    rows: list[tuple[str, str, str]] = []
    blocking_errors = 0

    def add_check(name: str, status: str, detail: str, blocking: bool = False) -> None:
        nonlocal blocking_errors
        rows.append((name, status, detail))
        if blocking:
            blocking_errors += 1

    try:
        bind_host = validate_ip_address(host, "--host")
        control_port = validate_port(port, "--port")
        tensor_port = validate_port(tensor_port, "--tensor-port")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if sys.version_info >= (3, 11):
        add_check("Python", "OK", sys.version.split()[0])
    else:
        add_check("Python", "FAIL", "Python 3.11+ is required", blocking=True)

    if sys.platform == "darwin":
        add_check("Platform", "OK", "macOS detected")
    else:
        add_check("Platform", "WARN", "MacFleet is intended for macOS Apple Silicon")

    try:
        import torch

        add_check("PyTorch", "OK", torch.__version__)
        if torch.backends.mps.is_available():
            add_check("MPS", "OK", "Apple GPU acceleration available")
        else:
            add_check("MPS", "WARN", "MPS unavailable; CPU training may be slow")
    except ImportError:
        add_check("PyTorch", "FAIL", "Install torch or macfleet[torch]", blocking=True)

    if is_port_available(control_port, bind_host):
        add_check("Control port", "OK", f"{bind_host}:{control_port} is available")
    else:
        add_check("Control port", "WARN", f"{bind_host}:{control_port} is already in use")

    if is_port_available(tensor_port, bind_host):
        add_check("Tensor port", "OK", f"{bind_host}:{tensor_port} is available")
    else:
        add_check("Tensor port", "WARN", f"{bind_host}:{tensor_port} is already in use")

    try:
        token = resolve_auth_token(auth_token_env=auth_token_env)
        if token:
            add_check("Auth token", "OK", f"{auth_token_env} is set")
        else:
            add_check("Auth token", "WARN", f"{auth_token_env} is not set")
    except ValueError as exc:
        add_check("Auth token", "FAIL", str(exc), blocking=True)

    if master:
        try:
            master_host, master_port = parse_endpoint(master, port, "--master")
        except ValueError as exc:
            raise click.ClickException(f"Invalid --master: {exc}") from exc

        if is_reachable(master_host, master_port, timeout=1.0):
            add_check("Coordinator", "OK", f"{master_host}:{master_port} is reachable")
        else:
            add_check(
                "Coordinator",
                "WARN",
                f"{master_host}:{master_port} is not reachable",
            )

    table = Table(title="MacFleet Diagnostics")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Detail")

    for name, status, detail in rows:
        style = "green" if status == "OK" else "yellow" if status == "WARN" else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]", detail)

    console.print(table)

    if blocking_errors:
        raise click.ClickException("Diagnostics found blocking issues.")


@cli.command()
def info():
    """Display system information for MacFleet.

    Shows GPU, memory, and network configuration.
    """
    from macfleet.utils.network import (
        get_gpu_info,
        get_hostname,
        get_local_ip,
        get_memory_bandwidth,
        get_memory_info,
        get_thunderbolt_bridge_ip,
    )

    console.print("[bold blue]System Information[/bold blue]")
    console.print()

    console.print(f"  Hostname: {get_hostname()}")
    console.print(f"  Local IP: {get_local_ip()}")

    tb_ip = get_thunderbolt_bridge_ip()
    if tb_ip:
        console.print(f"  Thunderbolt IP: [green]{tb_ip}[/green]")
    else:
        console.print("  Thunderbolt IP: [yellow]Not detected[/yellow]")

    console.print()

    gpu_info = get_gpu_info()
    console.print(f"  GPU: {gpu_info.get('gpu_name', 'Unknown')}")
    console.print(f"  GPU Cores: {gpu_info.get('gpu_cores', 0)}")

    memory_info = get_memory_info()
    console.print(f"  RAM: {memory_info.get('total_gb', 0)} GB")
    console.print(f"  Memory Bandwidth: ~{get_memory_bandwidth():.0f} GB/s")

    console.print()

    # Check PyTorch/MPS
    try:
        import torch
        console.print(f"  PyTorch: {torch.__version__}")
        console.print(f"  MPS Available: {torch.backends.mps.is_available()}")
        console.print(f"  MPS Built: {torch.backends.mps.is_built()}")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")


if __name__ == "__main__":
    cli()
