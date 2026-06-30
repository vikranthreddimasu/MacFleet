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
import errno
import signal
import sys
import time
from contextlib import contextmanager

import click
from rich.console import Console
from rich.table import Table

import macfleet

console = Console()
TRAINING_COMPRESSION_CHOICES = ("none", "light", "moderate", "aggressive", "adaptive")


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


@contextmanager
def _script_import_context(script: str):
    """Temporarily import sibling modules like `python path/to/script.py`."""
    import os

    script_dir = os.path.abspath(os.path.dirname(script) or ".")
    inserted = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass


@cli.command()
@click.option("--name", default=None, help="Custom node name")
@click.option(
    "--port",
    default=50051,
    type=click.IntRange(0, 65535),
    help="Heartbeat / discovery port",
)
@click.option(
    "--data-port",
    default=50052,
    type=click.IntRange(0, 65535),
    help="Data transport port (training)",
)
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
    type=click.FloatRange(0.0, min_open=True),
    show_default=True,
    help="Seconds that the one-time enrollment code remains valid.",
)
@click.option(
    "--enroll-uses",
    default=1,
    type=click.IntRange(1),
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
        try:
            resolved_token = resolve_token_with_file(token, auto_generate=True)
        except OSError as e:
            console.print(f"[red]Error: couldn't configure fleet token at {TOKEN_FILE}: {e}[/red]")
            console.print(
                "[dim]Fix the token path, or remove it and run `macfleet join` again "
                "to generate a fresh token.[/dim]"
            )
            sys.exit(1)
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
        agent_started = False

        async def cleanup_failed_start() -> None:
            try:
                await agent.stop()
            except Exception as cleanup_error:
                console.print(f"[dim]Cleanup after failed startup also failed: {cleanup_error}[/dim]")

        try:
            await agent.start()
            agent_started = True
        except OSError as e:
            await cleanup_failed_start()
            console.print(f"[red]Error: couldn't start MacFleet agent: {e}[/red]")
            if e.errno == errno.EADDRINUSE:
                console.print(
                    f"[dim]Port conflict detected. Stop the other MacFleet process or "
                    f"retry with --port {port + 10} --data-port {data_port + 10}.[/dim]"
                )
            elif e.errno == errno.EACCES:
                console.print("[dim]Permission denied while binding a local port. Try ports above 1024.[/dim]")
            else:
                console.print("[dim]Run `macfleet doctor` to check local networking.[/dim]")
            raise click.exceptions.Exit(1) from e
        except Exception as e:
            await cleanup_failed_start()
            console.print(f"[red]Error: couldn't start MacFleet agent: {e}[/red]")
            console.print("[dim]Run `macfleet doctor` to check local networking and token state.[/dim]")
            raise click.exceptions.Exit(1) from e

        try:
            if start_enrollment and resolved_token is not None:
                try:
                    enrollment_server = EnrollmentServer(
                        token=resolved_token,
                        fleet_id=fleet_id,
                        node_id=agent.node_id,
                        ttl_sec=enroll_ttl,
                        max_uses=enroll_uses,
                    )
                    await enrollment_server.start()
                except Exception as e:
                    console.print(f"[red]Error: couldn't start enrollment server: {e}[/red]")
                    console.print("[dim]Retry without --bootstrap, or run `macfleet doctor` for local checks.[/dim]")
                    raise click.exceptions.Exit(1) from e
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

            await stop_event.wait()
        finally:
            if enrollment_server is not None:
                await enrollment_server.stop()
            if agent_started:
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
@click.option("--json", "json_output", is_flag=True, default=False, help="Print machine-readable JSON")
def status(token: str | None, fleet_id: str | None, open_fleet: bool, json_output: bool):
    """Show pool status (discovers peers for 3 seconds)."""
    import json

    from macfleet.pool.discovery import ServiceRegistry
    from macfleet.security.auth import SecurityConfig, resolve_token_with_file

    if open_fleet:
        resolved = None
    else:
        try:
            resolved = resolve_token_with_file(token)
        except OSError as e:
            console.print(f"[red]Error: couldn't read fleet token at {e.filename or 'configured path'}: {e}[/red]")
            console.print("[dim]Use `macfleet status --open` to scan unauthenticated open fleets.[/dim]")
            sys.exit(1)

    sec = SecurityConfig(token=resolved, fleet_id=fleet_id) if resolved else None
    secure = bool(sec and sec.is_secure)
    if not json_output and secure:
        fleet_label = fleet_id or "default"
        console.print(f"[bold]Scanning fleet '{fleet_label}' for members...[/bold]")
    elif not json_output:
        console.print("[bold]Scanning for pool members...[/bold]")

    registry = ServiceRegistry(security=sec)
    try:
        peers = registry.find_peers(timeout=3.0)
    finally:
        registry.stop()

    sorted_peers = sorted(peers, key=lambda n: -n.compute_score)
    if json_output:
        payload = {
            "secure": secure,
            "fleet_id": fleet_id or ("default" if secure else None),
            "count": len(sorted_peers),
            "nodes": [_status_node_to_dict(node) for node in sorted_peers],
        }
        click.echo(json.dumps(payload, sort_keys=True))
        return

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

    for node in sorted_peers:
        table.add_row(
            node.hostname,
            node.chip_name,
            str(node.gpu_cores),
            str(node.ram_gb),
            f"{node.ip_address} :{node.port} :{node.data_port}",
            f"{node.compute_score:.0f}",
        )

    console.print(table)


def _status_node_to_dict(node) -> dict[str, object]:
    """Return stable JSON for `macfleet status --json`."""
    return {
        "hostname": node.hostname,
        "node_id": node.node_id,
        "ip_address": node.ip_address,
        "heartbeat_port": node.port,
        "data_port": node.data_port,
        "gpu_cores": node.gpu_cores,
        "ram_gb": node.ram_gb,
        "chip_name": node.chip_name,
        "link_types": node.link_type_list,
        "pool_version": node.pool_version,
        "compute_score": node.compute_score,
    }


@cli.command()
@click.option("--json", "json_output", is_flag=True, default=False, help="Print machine-readable JSON")
def diagnose(json_output: bool):
    """Run system health checks."""
    import json
    import os
    import platform
    import stat as stat_mod

    from macfleet.monitoring.thermal import get_thermal_state
    from macfleet.pool.agent import _check_mlx_available, _check_mps_available, profile_hardware
    from macfleet.pool.network import detect_interfaces
    from macfleet.security.auth import (
        RECOMMENDED_TOKEN_LENGTH,
        TOKEN_FILE,
        resolve_token_with_file,
    )

    checks: list[dict[str, object]] = []

    def add_check(section: str, name: str, passed: bool, detail: str = ""):
        checks.append(
            {
                "id": _diagnostic_check_id(section, name),
                "section": section,
                "name": name,
                "status": "ok" if passed else "fail",
                "passed": bool(passed),
                "detail": detail,
            }
        )

    # Runtime
    add_check(
        "Runtime",
        "Python >= 3.11",
        sys.version_info >= (3, 11),
        ".".join(str(part) for part in sys.version_info[:3]),
    )
    system = platform.system() or "unknown"
    mac_version = platform.mac_ver()[0] or platform.release()
    add_check("Runtime", "macOS detected", system == "Darwin", f"{system} {mac_version}".strip())
    machine = platform.machine() or "unknown"
    add_check("Runtime", "Apple Silicon architecture", machine == "arm64", machine)

    # Hardware
    hw = profile_hardware()
    add_check(
        "Hardware",
        "Apple Silicon detected",
        "apple" in hw.chip_name.lower() or "m" in hw.chip_name.lower(),
        hw.chip_name,
    )
    add_check("Hardware", "GPU cores detected", hw.gpu_cores > 0, f"{hw.gpu_cores} cores")
    add_check("Hardware", "RAM detected", hw.ram_gb > 0, f"{hw.ram_gb:.0f} GB")
    add_check("Hardware", "RAM >= 8 GB", hw.ram_gb >= 8, f"{hw.ram_gb:.0f} GB")

    # Frameworks
    add_check("ML Frameworks", "MPS available", _check_mps_available())
    add_check("ML Frameworks", "MLX available", _check_mlx_available())

    # Thermal
    thermal = get_thermal_state()
    add_check("Thermal", "Not throttling", not thermal.is_throttling, thermal.pressure.value)

    # Network
    links = detect_interfaces()
    add_check("Network", "Network interfaces found", len(links) > 0, f"{len(links)} interfaces")
    has_non_loopback = any(l.link_type.value != "loopback" for l in links)
    add_check("Network", "Non-loopback interface", has_non_loopback)

    # Security
    token_read_error = None
    try:
        tok = resolve_token_with_file(None)
    except OSError as e:
        tok = None
        token_read_error = e
    if token_read_error is not None:
        add_check("Security", "Fleet token readable", False, str(token_read_error))
    else:
        add_check(
            "Security",
            "Fleet token configured",
            tok is not None,
            "" if tok else "run 'macfleet join' once to auto-generate",
        )
    if tok is not None:
        add_check(
            "Security",
            f"Token length >= {RECOMMENDED_TOKEN_LENGTH}",
            len(tok) >= RECOMMENDED_TOKEN_LENGTH,
            f"{len(tok)} chars" + (
                "" if len(tok) >= RECOMMENDED_TOKEN_LENGTH
                else " — short tokens are dictionary-attackable"
            ),
        )
    if os.path.lexists(TOKEN_FILE):
        try:
            token_stat = os.lstat(TOKEN_FILE)
        except OSError as e:
            add_check("Security", "Token file metadata readable", False, str(e))
        else:
            is_regular = stat_mod.S_ISREG(token_stat.st_mode)
            add_check(
                "Security",
                "Token file is regular file",
                is_regular,
                "" if is_regular else f"must not be a symlink or directory: {TOKEN_FILE}",
            )
            if is_regular:
                mode = stat_mod.S_IMODE(token_stat.st_mode)
                add_check(
                    "Security",
                    "Token file private (0600)",
                    (mode & 0o077) == 0,
                    f"mode {oct(mode)}" + (
                        "" if (mode & 0o077) == 0 else f" — fix: chmod 600 {TOKEN_FILE}"
                    ),
                )

    passed_count = sum(1 for check in checks if check["passed"])
    total_count = len(checks)
    report = {
        "status": "ok" if passed_count == total_count else "fail",
        "ready": passed_count == total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "total": total_count,
        "checks": checks,
    }

    if json_output:
        click.echo(json.dumps(report, sort_keys=True))
        return

    _print_diagnostics_report(report)


def _diagnostic_check_id(section: str, name: str) -> str:
    raw = f"{section}.{name}".lower()
    return "".join(char if char.isalnum() else "_" for char in raw).strip("_")


def _print_diagnostics_report(report: dict[str, object]):
    console.print("[bold]Running diagnostics...[/bold]\n")
    current_section = None
    checks = report["checks"]
    assert isinstance(checks, list)
    for check in checks:
        assert isinstance(check, dict)
        section = check["section"]
        if section != current_section:
            if current_section is not None:
                console.print()
            console.print(f"[bold]{section}[/bold]")
            current_section = section

        passed = bool(check["passed"])
        label = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        detail = check["detail"]
        console.print(f"  {label} {check['name']}" + (f" — {detail}" if detail else ""))

    console.print(f"\n[bold]{report['passed']}/{report['total']} checks passed[/bold]")
    if report["ready"]:
        console.print("[green]System is ready for MacFleet![/green]")
    else:
        console.print("[yellow]Some checks failed. See above for details.[/yellow]")


@cli.command()
@click.argument("script", required=False)
@click.option("--engine", type=click.Choice(["torch", "mlx"]), default="torch")
@click.option("--epochs", default=10, type=click.IntRange(1), help="Number of training epochs")
@click.option("--batch-size", default=128, type=click.IntRange(1), help="Global batch size")
@click.option("--lr", default=0.001, type=click.FloatRange(0.0, min_open=True), help="Learning rate")
@click.option(
    "--compression",
    default="none",
    type=click.Choice(TRAINING_COMPRESSION_CHOICES),
    help="Distributed compression mode",
)
@click.option("--config", "config_path", default=None, help="JSON/YAML config file")
@click.pass_context
def train(
    ctx: click.Context,
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
    a `main()` function. If that function accepts parameters named
    `engine`, `epochs`, `batch_size`, `lr`, `compression`, or `config_path`,
    the matching CLI values are passed in. Otherwise, runs a built-in demo
    (small MLP on synthetic data) useful for testing the pipeline.
    """
    if config_path is not None:
        engine, epochs, batch_size, lr, compression = _apply_train_config(
            ctx,
            config_path,
            engine=engine,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            compression=compression,
        )
    if script:
        _train_from_script(
            script, engine, epochs, batch_size, lr, compression, config_path
        )
    else:
        _train_demo(engine, epochs, batch_size, lr)


def _apply_train_config(
    ctx: click.Context,
    config_path: str,
    *,
    engine: str,
    epochs: int,
    batch_size: int,
    lr: float,
    compression: str,
) -> tuple[str, int, int, float, str]:
    """Load training config values, letting explicit CLI flags win."""
    config = _load_train_config(config_path)
    values: dict[str, object] = {
        "engine": engine,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "compression": compression,
    }
    for name in values:
        if (
            name in config
            and ctx.get_parameter_source(name) == click.core.ParameterSource.DEFAULT
        ):
            values[name] = config[name]

    return (
        _coerce_train_config_value("engine", values["engine"]),
        _coerce_train_config_value("epochs", values["epochs"]),
        _coerce_train_config_value("batch_size", values["batch_size"]),
        _coerce_train_config_value("lr", values["lr"]),
        _coerce_train_config_value("compression", values["compression"]),
    )


def _load_train_config(config_path: str) -> dict[str, object]:
    """Load and validate a JSON/YAML training config file."""
    import json
    from pathlib import Path

    path = Path(config_path)
    if not path.is_file():
        raise click.ClickException(f"Training config not found: {config_path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            loaded = json.loads(path.read_text())
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as e:
                raise click.ClickException(
                    "Reading YAML configs requires PyYAML. Install with "
                    "`pip install 'macfleet[yaml]'`, or use a .json config."
                ) from e
            try:
                loaded = yaml.safe_load(path.read_text())
            except yaml.YAMLError as e:
                raise click.ClickException(
                    f"Couldn't parse training config {config_path}: {e}"
                ) from e
        else:
            raise click.ClickException(
                "Training config must be a .json, .yaml, or .yml file."
            )
    except OSError as e:
        raise click.ClickException(f"Couldn't read training config {config_path}: {e}") from e
    except (json.JSONDecodeError, ValueError) as e:
        raise click.ClickException(f"Couldn't parse training config {config_path}: {e}") from e

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise click.ClickException("Training config must contain a mapping/object at the top level.")

    normalized: dict[str, object] = {}
    aliases = {
        "batch-size": "batch_size",
        "learning-rate": "lr",
        "learning_rate": "lr",
    }
    allowed = {"engine", "epochs", "batch_size", "lr", "compression"}
    for raw_key, value in loaded.items():
        key = aliases.get(str(raw_key), str(raw_key).replace("-", "_"))
        if key not in allowed:
            allowed_list = ", ".join(sorted(allowed))
            raise click.ClickException(
                f"Unknown training config key {raw_key!r}. Supported keys: {allowed_list}."
            )
        normalized[key] = value
    return normalized


def _coerce_train_config_value(name: str, value: object):
    """Coerce config-file values to the same shape Click returns."""
    import math

    if name == "engine":
        if not isinstance(value, str) or value not in ("torch", "mlx"):
            raise click.ClickException("Training config 'engine' must be 'torch' or 'mlx'.")
        return value
    if name == "compression":
        if not isinstance(value, str) or value not in TRAINING_COMPRESSION_CHOICES:
            choices = ", ".join(TRAINING_COMPRESSION_CHOICES)
            raise click.ClickException(
                f"Training config 'compression' must be one of: {choices}."
            )
        return value
    if name in {"epochs", "batch_size"}:
        if isinstance(value, bool):
            raise click.ClickException(f"Training config '{name}' must be a positive integer.")
        try:
            coerced = int(value)
        except (TypeError, ValueError) as e:
            raise click.ClickException(
                f"Training config '{name}' must be a positive integer."
            ) from e
        if coerced < 1 or str(value).strip() != str(coerced):
            raise click.ClickException(f"Training config '{name}' must be a positive integer.")
        return coerced
    if name == "lr":
        if isinstance(value, bool):
            raise click.ClickException("Training config 'lr' must be a positive finite number.")
        try:
            coerced_lr = float(value)
        except (TypeError, ValueError) as e:
            raise click.ClickException(
                "Training config 'lr' must be a positive finite number."
            ) from e
        if not math.isfinite(coerced_lr) or coerced_lr <= 0:
            raise click.ClickException("Training config 'lr' must be a positive finite number.")
        return coerced_lr
    return value


def _train_demo(engine_type: str, epochs: int, batch_size: int, lr: float):
    """Run a built-in demo training on synthetic data (single-node)."""
    if engine_type == "mlx":
        _train_demo_mlx(epochs, batch_size, lr)
        return
    _train_demo_torch(epochs, batch_size, lr)


def _train_demo_torch(epochs: int, batch_size: int, lr: float) -> None:
    """Run the built-in PyTorch demo training on synthetic data."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise click.ClickException(
            "PyTorch demo training requires PyTorch. Install with "
            "`pip install 'macfleet[torch]'`, or run `macfleet train --engine mlx`."
        ) from e

    from macfleet.engines.torch_engine import TorchEngine

    console.print("[bold blue]MacFleet Demo Training[/bold blue] — torch")
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


def _train_demo_mlx(epochs: int, batch_size: int, lr: float) -> None:
    """Run the built-in MLX demo training on synthetic data."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
    except ImportError as e:
        raise click.ClickException(
            "MLX demo training requires MLX. Install with "
            "`pip install 'macfleet[mlx]'`, or run `macfleet train --engine torch`."
        ) from e

    from macfleet.engines.mlx_engine import MLXEngine

    class TinyMLXModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(4, 32)
            self.linear2 = nn.Linear(32, 2)

        def __call__(self, x):
            return self.linear2(nn.relu(self.linear1(x)))

    def loss_fn(model, x, y):
        return nn.losses.cross_entropy(model(x), y, reduction="mean")

    console.print("[bold blue]MacFleet Demo Training[/bold blue] — mlx")
    console.print("[dim]Single-node training on synthetic data (no peers needed)[/dim]\n")

    rng = np.random.default_rng(42)
    n_samples = 1000
    x_np = rng.normal(size=(n_samples, 4)).astype(np.float32)
    y_np = (x_np[:, 0] + x_np[:, 1] > 0).astype(np.int32)

    model = TinyMLXModel()
    optimizer = optim.Adam(learning_rate=lr)
    eng = MLXEngine()
    eng.load_model(model, optimizer, loss_fn=loss_fn)

    console.print(f"  Model params: {eng.param_count():,}")
    console.print(f"  Dataset size: {n_samples}")
    console.print(f"  Batch size:   {batch_size}")
    console.print(f"  Epochs:       {epochs}")
    console.print("  Device:       mlx\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        order = rng.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = order[start:start + batch_size]
            batch_x = mx.array(x_np[batch_idx])
            batch_y = mx.array(y_np[batch_idx])

            eng.zero_grad()
            loss = eng.forward((batch_x, batch_y))
            eng.backward(loss)
            logits = model(batch_x)
            eng.step()

            epoch_loss += float(loss)
            correct += int((np.array(logits).argmax(axis=1) == y_np[batch_idx]).sum())
            total += len(batch_idx)

        elapsed = time.time() - t0
        acc = correct / total * 100
        avg_loss = epoch_loss / max((n_samples + batch_size - 1) // batch_size, 1)
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
    config_path: str | None,
):
    """Run a user-provided training script."""
    import importlib.util
    import inspect
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
    with _script_import_context(script):
        spec.loader.exec_module(module)

    # Expect the script to define a `main()` function.
    if hasattr(module, "main"):
        main_fn = module.main
        if not callable(main_fn):
            console.print(f"[red]Error: Script attribute 'main' in {script} is not callable.[/red]", soft_wrap=True)
            console.print("[dim]Define a function named main(), then retry.[/dim]")
            sys.exit(1)
        options = {
            "engine": engine_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "compression": compression,
            "config_path": config_path,
        }
        try:
            signature = inspect.signature(main_fn)
        except (TypeError, ValueError):
            with _script_import_context(script):
                main_fn()
            return

        params = signature.parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in params.values()
        )
        kwargs = {}
        for name, value in options.items():
            param = params.get(name)
            if accepts_kwargs or (
                param is not None
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ):
                kwargs[name] = value
        if "config" in params and "config_path" not in params:
            param = params["config"]
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                kwargs["config"] = config_path

        missing_required = [
            name
            for name, param in params.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and name not in kwargs
        ]
        if missing_required:
            console.print(
                "[red]Error: Script main() has required parameter(s) MacFleet "
                f"cannot provide: {', '.join(missing_required)}[/red]"
            )
            console.print(
                "[dim]Supported injected parameters: engine, epochs, "
                "batch_size, lr, compression, config_path.[/dim]"
            )
            sys.exit(1)

        with _script_import_context(script):
            main_fn(**kwargs)
    else:
        console.print("[red]Error: Script must define a main() function.[/red]")
        console.print("[dim]Example:[/dim]")
        console.print("[dim]  def main(epochs=10, batch_size=128, lr=0.001):[/dim]")
        console.print("[dim]      model = MyModel()[/dim]")
        console.print(
            "[dim]      macfleet.train(model, dataset, epochs=epochs, "
            "batch_size=batch_size, lr=lr)[/dim]"
        )
        sys.exit(1)


@cli.command(name="run")
@click.argument("script")
@click.option("--fn", "fn_name", default="main", help="Function to execute (default: main)")
@click.option("--token", default=None, envvar="MACFLEET_TOKEN", help="Pool token")
@click.option("--open", "open_fleet", is_flag=True, default=False, help="Disable security")
@click.option(
    "--allow-insecure-open",
    is_flag=True,
    default=False,
    help="Required with --open. Makes unauthenticated execution explicit.",
)
@click.option(
    "--allow-legacy-pickle",
    is_flag=True,
    default=False,
    help="Allow trusted local-only execution of undecorated functions.",
)
def run_command(
    script: str,
    fn_name: str,
    token: str | None,
    open_fleet: bool,
    allow_insecure_open: bool,
    allow_legacy_pickle: bool,
):
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

    if open_fleet:
        if token:
            console.print("[red]Error: --open and --token are mutually exclusive.[/red]")
            sys.exit(1)
        if not allow_insecure_open:
            console.print(
                "[red]Error: --open disables authentication for execution. Re-run with "
                "--open --allow-insecure-open if you really want an open LAN pool.[/red]"
            )
            sys.exit(1)

    if not os.path.isfile(script):
        console.print(f"[red]Error: Script not found: {script}[/red]")
        sys.exit(1)

    # Load user script
    spec = importlib.util.spec_from_file_location("user_script", script)
    if spec is None or spec.loader is None:
        console.print(f"[red]Error: Cannot load script '{script}' as a Python module.[/red]")
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    with _script_import_context(script):
        spec.loader.exec_module(module)

    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        console.print(f"[red]Error: Function '{fn_name}' not found in {script}[/red]")
        console.print(f"[dim]The script must define a callable named '{fn_name}'.[/dim]")
        sys.exit(1)
    if not allow_legacy_pickle and not hasattr(fn, "task_name"):
        console.print(
            "[red]Error: macfleet run requires a function decorated with "
            "@macfleet.task by default.[/red]"
        )
        console.print(
            "[dim]Decorate the function, or re-run with --allow-legacy-pickle "
            "for trusted local-only migration code.[/dim]"
        )
        sys.exit(1)

    console.print(f"[bold blue]MacFleet Run[/bold blue] — {script}:{fn_name}()")

    from macfleet.sdk.pool import Pool

    with _script_import_context(script):
        with Pool(
            token=token,
            open=open_fleet,
            allow_legacy_pickle=allow_legacy_pickle,
        ) as pool:
            t0 = time.time()
            result = pool.run(fn)
            elapsed = time.time() - t0

    console.print(f"\n[green]Completed in {elapsed:.2f}s[/green]")
    if result is not None:
        console.print(f"Result: {result}")


@cli.command()
@click.option("--type", "bench_type", type=click.Choice(["network", "compute", "allreduce"]), default="network")
@click.option(
    "--size-mb",
    default=10,
    type=click.IntRange(1, 4096),
    help="Payload size in MB for network tests",
)
@click.option("--iterations", default=5, type=click.IntRange(1), help="Number of iterations")
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

    if os.path.lexists(TOKEN_FILE) and not yes:
        confirmed = click.confirm(
            "Replace the saved fleet token on this Mac? "
            "Running agents and other Macs must be restarted/re-paired.",
            default=False,
        )
        if not confirmed:
            console.print("[yellow]Token rotation cancelled.[/yellow]")
            raise click.exceptions.Exit(1)

    try:
        token = rotate_saved_fleet_token()
    except OSError as e:
        console.print(f"[red]Error: couldn't rotate fleet token at {TOKEN_FILE}: {e}[/red]")
        sys.exit(1)
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
@click.option("--json", "json_output", is_flag=True, default=False, help="Print machine-readable JSON")
@click.pass_context
def doctor(ctx, json_output: bool):
    """System health check (alias for `diagnose`)."""
    # Delegate through Click's context so diagnose runs with a proper context
    # (obj, exception handling) instead of a bare callback() call.
    ctx.invoke(diagnose, json_output=json_output)


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
