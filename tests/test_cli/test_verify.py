from click.testing import CliRunner

from macfleet.cli.main import cli


def test_verify_passes_fleet_id_to_harness(monkeypatch):
    calls = []
    monkeypatch.setattr("macfleet.cli.verify.main", lambda args: calls.append(args) or 0)
    result = CliRunner().invoke(cli, ["verify", "--fleet-id", "named"])
    assert result.exit_code == 0
    assert calls == [["--fleet-id", "named"]]


def test_verifier_passes_named_fleet_to_pool(monkeypatch):
    import macfleet
    from macfleet.cli.verify import run_distributed_check

    calls = []

    class UnavailablePool:
        def __enter__(self):
            raise RuntimeError("no peers")

    def make_pool(**kwargs):
        calls.append(kwargs)
        return UnavailablePool()

    monkeypatch.setattr(macfleet, "Pool", make_pool)
    report = run_distributed_check(fleet_id="named")
    assert calls[0]["fleet_id"] == "named"
    assert report.failures == 1


def test_verifier_uses_configured_name_for_self_and_peer(monkeypatch):
    import macfleet
    from macfleet.cli import verify
    from macfleet.pool.network import LinkType, NetworkLink

    class Pool:
        world_size = 2
        nodes = [
            {"node_id": "self", "hostname": "custom-name", "ip_address": "192.168.1.10",
             "data_port": 50052, "chip_name": "Apple M5"},
            {"node_id": "peer", "hostname": "other-mac", "ip_address": "192.168.1.20",
             "data_port": 50052, "chip_name": "Apple M4"},
        ]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def train(self, **kwargs):
            return {"rank": 0, "world_size": 2, "steps": 24, "degraded": False,
                    "unsynced_steps": 0, "params_sha256": "a" * 64}

    monkeypatch.setenv("MACFLEET_NAME", "custom-name")
    monkeypatch.setattr(macfleet, "Pool", lambda **kwargs: Pool())
    monkeypatch.setattr(verify.socket, "gethostname", lambda: "real-host.local")
    monkeypatch.setattr(
        "macfleet.pool.network.detect_interfaces",
        lambda: [NetworkLink("utun0", LinkType.UNKNOWN, "fe80::1"),
                 NetworkLink("lo0", LinkType.LOOPBACK, "127.0.0.1"),
                 NetworkLink("en0", LinkType.WIFI, "192.168.1.10")],
    )
    monkeypatch.setattr(verify, "_make_dataset", lambda: None)
    monkeypatch.setattr(verify, "_make_model", lambda: None)
    report = verify.run_distributed_check()
    checks = {check.check_id: check for check in report.checks}
    assert checks["registry_sanity"].passed
    assert checks["topology_peer_address"].passed
    assert "peer 192.168.1.20" in checks["topology_peer_address"].detail


def test_verify_result_includes_cross_host_evidence(capsys, tmp_path, monkeypatch):
    from macfleet.cli.verify import Report, _print_report

    monkeypatch.setenv("HOME", str(tmp_path))
    report = Report(mode="distributed", hostname="test")
    report.info["train"] = {
        "rank": 1, "steps": 24, "world_size": 2, "degraded": False,
        "unsynced_steps": 0, "params_sha256": "a" * 64,
    }
    _print_report(report)
    output = capsys.readouterr().out
    assert "rank=1 steps=24" in output
    assert "params_sha256=" + "a" * 64 in output
    assert "Compare both Macs" in output


def test_failed_hardware_probes_are_unknown(monkeypatch):
    from macfleet.pool import agent

    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(agent.subprocess, "run", missing)
    assert agent._detect_chip_name() == "unknown"
    assert agent._detect_gpu_cores() == 0
    assert agent._detect_ram_gb() == 0
    assert agent._detect_memory_bandwidth() == 0
