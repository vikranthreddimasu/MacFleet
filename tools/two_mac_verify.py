"""MacFleet two-Mac verification harness.

Runs a battery of structured PASS/FAIL checks that confirm MacFleet is
installed correctly and behaving as intended. Two modes:

    # Single machine, no peer needed. Run this FIRST on each Mac.
    python tools/two_mac_verify.py --self-check

    # Distributed. Run on BOTH paired Macs within ~45s of each other.
    python tools/two_mac_verify.py

Self-check exercises the local plumbing that does not need a second Mac:
import + version, interface detection, deterministic topology address
selection, link serialization, and a single-node ``Pool.train`` that must
actually reduce loss.

Distributed mode exercises the real cross-Mac path: quorum formation,
registry sanity (two distinct nodes with data ports and chips), topology
peer-address selection, and a distributed ``Pool.train`` whose final
parameter hash must be identical on both Macs with no degraded steps.

Exit code is the number of failed checks (0 == everything passed), so the
harness is usable as a gate in scripts and CI.

In distributed mode each Mac also prints a single ``VERIFY-RESULT`` line and
writes ``~/.macfleet/verify-<hostname>.json``. The two Macs PASS the
cross-machine check only when their ``params_sha256`` values match.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Tiny check framework
# ---------------------------------------------------------------------------
@dataclass
class Check:
    check_id: str
    passed: bool
    detail: str


@dataclass
class Report:
    mode: str
    hostname: str
    checks: list[Check] = field(default_factory=list)
    info: dict[str, object] = field(default_factory=dict)

    def add(self, check_id: str, passed: bool, detail: str) -> bool:
        self.checks.append(Check(check_id, passed, detail))
        return passed

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


def _guard(report: Report, check_id: str, fn: Callable[[], tuple[bool, str]]) -> bool:
    """Run a check body, turning any exception into a clean FAIL."""
    try:
        passed, detail = fn()
    except Exception as exc:  # noqa: BLE001 - report any failure, never crash
        detail = f"raised {type(exc).__name__}: {exc}"
        if os.environ.get("VERIFY_TRACEBACK"):
            traceback.print_exc()
        passed = False
    return report.add(check_id, passed, detail)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _peers_from_env() -> list[str]:
    raw = os.environ.get("MACFLEET_PEERS", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# Shared training fixtures (deterministic, linearly separable -> loss drops)
# ---------------------------------------------------------------------------
def _make_dataset(n_samples: int = 512):
    import torch
    from torch.utils.data import TensorDataset

    torch.manual_seed(1234)
    x = torch.randn(n_samples, 8)
    y = (x[:, 0] + 0.7 * x[:, 1] - 0.4 * x[:, 2] > 0).long()
    return TensorDataset(x, y)


def _make_model():
    import torch
    import torch.nn as nn

    # Per-host seed proves rank 0's setup broadcast works in distributed mode.
    host_seed = sum(socket.gethostname().encode("utf-8")) % 10_000
    torch.manual_seed(9000 + host_seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))


# ---------------------------------------------------------------------------
# Self-check: everything that needs only one machine
# ---------------------------------------------------------------------------
def run_self_check() -> Report:
    report = Report(mode="self-check", hostname=socket.gethostname().split(".")[0])

    # C1: import + version
    def _import() -> tuple[bool, str]:
        import macfleet

        version = getattr(macfleet, "__version__", "")
        report.info["version"] = version
        ok = isinstance(version, str) and bool(version)
        return ok, f"macfleet {version!r}"

    _guard(report, "import_and_version", _import)

    # C2: interface detection finds at least one dialable address
    def _interfaces() -> tuple[bool, str]:
        from macfleet.pool.network import detect_interfaces
        from macfleet.pool.topology import _is_dialable_address

        links = detect_interfaces()
        dialable = [link for link in links if _is_dialable_address(link.ip_address)]
        report.info["interfaces"] = [
            {"i": link.interface, "t": link.link_type.value, "ip": link.ip_address}
            for link in links
        ]
        return bool(dialable), (
            f"{len(links)} interface(s), {len(dialable)} dialable: "
            + ", ".join(f"{link.interface}/{link.link_type.value}" for link in links)
        )

    _guard(report, "interface_detection", _interfaces)

    # C3: topology address selection is correct AND deterministic
    def _topology_selection() -> tuple[bool, str]:
        from macfleet.pool.network import LinkType, NetworkLink
        from macfleet.pool.topology import NodeTopology, best_peer_address

        local = NodeTopology(
            node_id="local",
            default_ip="192.168.1.10",
            links=(
                NetworkLink("en0", LinkType.WIFI, "192.168.1.10"),
                NetworkLink("bridge0", LinkType.THUNDERBOLT, "10.0.0.10"),
            ),
        )
        peer = NodeTopology(
            node_id="peer",
            default_ip="192.168.1.20",
            links=(
                NetworkLink("en0", LinkType.WIFI, "192.168.1.20"),
                NetworkLink("bridge0", LinkType.THUNDERBOLT, "10.0.0.20"),
            ),
        )
        first = best_peer_address(local, peer)
        again = best_peer_address(local, peer)
        # A shared Thunderbolt subnet must beat shared Wi-Fi, and be stable.
        prefers_thunderbolt = first == "10.0.0.20"
        deterministic = first == again
        ok = prefers_thunderbolt and deterministic
        return ok, (
            f"chose {first!r} (thunderbolt-preferred={prefers_thunderbolt}, "
            f"deterministic={deterministic})"
        )

    _guard(report, "topology_selection", _topology_selection)

    # C4: link serialization round-trips and drops loopback / non-dialable
    def _serialization() -> tuple[bool, str]:
        from macfleet.pool.network import LinkType, NetworkLink
        from macfleet.pool.topology import (
            deserialize_network_links,
            serialize_network_links,
        )

        links = [
            NetworkLink("lo0", LinkType.LOOPBACK, "127.0.0.1"),
            NetworkLink("en0", LinkType.WIFI, "192.168.1.10", bandwidth_mbps=300.0),
            NetworkLink("bridge0", LinkType.THUNDERBOLT, "169.254.10.1"),
        ]
        payload = serialize_network_links(links)
        restored = deserialize_network_links(payload)
        ips = {link.ip_address for link in restored}
        loopback_dropped = "127.0.0.1" not in ips
        kept_real = {"192.168.1.10", "169.254.10.1"} <= ips
        ok = loopback_dropped and kept_real
        return ok, f"serialized {len(restored)} link(s) ips={sorted(ips)}"

    _guard(report, "link_serialization", _serialization)

    # C5: single-node Pool.train actually trains (loss decreases, steps > 0)
    def _single_node_train() -> tuple[bool, str]:
        import torch.nn as nn

        import macfleet

        with macfleet.Pool(name="verify-self", enable_pool_distributed=False) as pool:
            world = pool.world_size
            result = pool.train(
                model=_make_model(),
                dataset=_make_dataset(256),
                epochs=_env_int("EPOCHS", 4),
                batch_size=_env_int("BATCH_SIZE", 32),
                lr=_env_float("LR", 0.05),
                loss_fn=nn.CrossEntropyLoss(),
                device=os.environ.get("DEVICE", "cpu"),
                distributed=False,
            )
        history = result.get("loss_history") or []
        steps = result.get("steps", 0)
        report.info["self_train"] = {
            "world_size": world,
            "steps": steps,
            "loss_first": history[0] if history else None,
            "loss_last": history[-1] if history else None,
        }
        learned = bool(history) and history[-1] < history[0]
        ok = world == 1 and steps > 0 and learned
        first = history[0] if history else float("nan")
        last = history[-1] if history else float("nan")
        return ok, (
            f"world_size={world} steps={steps} "
            f"loss {first:.4f} -> {last:.4f} (decreased={learned})"
        )

    _guard(report, "single_node_train", _single_node_train)

    return report


# ---------------------------------------------------------------------------
# Distributed check: needs the second Mac
# ---------------------------------------------------------------------------
def run_distributed_check() -> Report:
    import torch.nn as nn

    import macfleet

    report = Report(mode="distributed", hostname=socket.gethostname().split(".")[0])
    expected = _env_int("QUORUM_SIZE", 2)

    name = os.environ.get("MACFLEET_NAME", report.hostname)
    pool = macfleet.Pool(
        name=name,
        enable_pool_distributed=True,
        quorum_size=expected,
        quorum_timeout_sec=_env_float("QUORUM_TIMEOUT", 45.0),
        rendezvous_timeout_sec=_env_float("RENDEZVOUS_TIMEOUT", 90.0),
        peers=_peers_from_env(),
    )

    try:
        pool.__enter__()
    except Exception as exc:  # noqa: BLE001
        report.add(
            "quorum_formation",
            False,
            f"failed to join pool: {type(exc).__name__}: {exc}",
        )
        # Nothing else can run without a pool.
        return report

    try:
        nodes = list(pool.nodes)
        report.info["nodes"] = nodes

        # D1: quorum reached the expected world size
        report.add(
            "quorum_formation",
            pool.world_size == expected,
            f"world_size={pool.world_size} (expected {expected})",
        )

        # D2: registry sanity - distinct nodes, ports, chips, self present
        def _registry() -> tuple[bool, str]:
            node_ids = {str(n["node_id"]) for n in nodes}
            distinct = len(node_ids) == len(nodes) == expected
            ports_ok = all(int(n.get("data_port") or 0) > 0 for n in nodes)
            chips_ok = all(str(n.get("chip_name") or "").strip() for n in nodes)
            self_present = any(
                str(n.get("hostname", "")).split(".")[0] == report.hostname
                for n in nodes
            )
            ok = distinct and ports_ok and chips_ok and self_present
            chips = ", ".join(sorted(str(n.get("chip_name")) for n in nodes))
            return ok, (
                f"{len(nodes)} node(s) distinct={distinct} ports_ok={ports_ok} "
                f"chips_ok={chips_ok} self_present={self_present} [{chips}]"
            )

        _guard(report, "registry_sanity", _registry)

        # D3: topology selects a dialable peer address from real interfaces
        def _peer_address() -> tuple[bool, str]:
            from macfleet.pool.network import detect_interfaces
            from macfleet.pool.topology import (
                NodeTopology,
                _is_dialable_address,
                best_peer_address,
            )

            peers = [
                n
                for n in nodes
                if str(n.get("hostname", "")).split(".")[0] != report.hostname
            ]
            if not peers:
                return False, "no peer node found in registry"
            peer = peers[0]
            local_topo = NodeTopology(
                node_id="local",
                default_ip=detect_interfaces()[0].ip_address
                if detect_interfaces()
                else "0.0.0.0",
                links=tuple(detect_interfaces()),
            )
            peer_topo = NodeTopology(
                node_id=str(peer["node_id"]),
                default_ip=str(peer["ip_address"]),
            )
            chosen = best_peer_address(local_topo, peer_topo)
            ok = _is_dialable_address(chosen)
            return ok, f"peer {peer['ip_address']} -> dial {chosen!r}"

        _guard(report, "topology_peer_address", _peer_address)

        # D4: distributed Pool.train converges to identical synced params
        sha = None

        def _distributed_train() -> tuple[bool, str]:
            nonlocal sha
            result = pool.train(
                model=_make_model(),
                dataset=_make_dataset(),
                epochs=_env_int("EPOCHS", 3),
                batch_size=_env_int("BATCH_SIZE", 64),
                lr=_env_float("LR", 0.03),
                loss_fn=nn.CrossEntropyLoss(),
                device=os.environ.get("DEVICE", "cpu"),
                compression=os.environ.get("COMPRESSION", "none"),
                distributed=True,
            )
            sha = result.get("params_sha256")
            report.info["train"] = {
                "rank": result.get("rank"),
                "world_size": result.get("world_size"),
                "steps": result.get("steps"),
                "degraded": result.get("degraded"),
                "unsynced_steps": result.get("unsynced_steps"),
                "last_sync_error": result.get("last_sync_error"),
                "avg_sync_time_sec": result.get("avg_sync_time_sec"),
                "params_sha256": sha,
            }
            world_ok = result.get("world_size") == expected
            not_degraded = result.get("degraded") is False
            synced = (result.get("unsynced_steps") or 0) == 0
            sha_ok = isinstance(sha, str) and bool(_SHA256_RE.match(sha))
            ok = world_ok and not_degraded and synced and sha_ok
            return ok, (
                f"world={result.get('world_size')} degraded={result.get('degraded')} "
                f"unsynced={result.get('unsynced_steps')} "
                f"params_sha256={(sha or '')[:12]}..."
            )

        _guard(report, "distributed_train", _distributed_train)
    finally:
        try:
            pool.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_report(report: Report) -> None:
    print()
    print(f"MacFleet verify [{report.mode}] on {report.hostname}")
    print("-" * 64)
    for check in report.checks:
        tag = "PASS" if check.passed else "FAIL"
        print(f"  [{tag}] {check.check_id}: {check.detail}")
    print("-" * 64)
    total = len(report.checks)
    passed = total - report.failures
    print(f"  {passed}/{total} checks passed")

    if report.mode == "distributed":
        train = report.info.get("train")
        train = train if isinstance(train, dict) else {}
        sha = str(train.get("params_sha256") or "")
        world_size = train.get("world_size")
        degraded = train.get("degraded")
        # Single machine-readable line for cross-Mac comparison.
        print(
            f"VERIFY-RESULT host={report.hostname} world_size={world_size} "
            f"degraded={degraded} params_sha256={sha or 'none'}"
        )
        artifact = Path.home() / ".macfleet" / f"verify-{report.hostname}.json"
        try:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps(
                    {
                        "mode": report.mode,
                        "hostname": report.hostname,
                        "checks": [asdict(c) for c in report.checks],
                        "info": report.info,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            print(f"  wrote {artifact}")
        except OSError as exc:
            print(f"  (could not write artifact: {exc})")
        print()
        print("CROSS-MAC PASS when BOTH Macs show the SAME params_sha256,")
        print("world_size=2, and degraded=False above.")

    if os.environ.get("VERIFY_JSON"):
        print(
            json.dumps(
                {
                    "mode": report.mode,
                    "hostname": report.hostname,
                    "failures": report.failures,
                    "checks": [asdict(c) for c in report.checks],
                },
                indent=2,
                sort_keys=True,
            )
        )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    self_check = "--self-check" in argv or os.environ.get("VERIFY_SELF_CHECK") == "1"

    report = run_self_check() if self_check else run_distributed_check()
    _print_report(report)
    return report.failures


if __name__ == "__main__":
    raise SystemExit(main())
