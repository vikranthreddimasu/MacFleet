"""Generated agent identities must fit the discovery protocol."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from macfleet.engines.base import ThermalPressure
from macfleet.pool import agent as agent_module
from macfleet.pool.discovery import PoolServiceListener
from macfleet.pool.network import NetworkTopology


@pytest.mark.parametrize("source", ["hostname", "name"])
@pytest.mark.parametrize(
    ("hostname", "expected_prefix"),
    [
        ("mac.local", "mac.local"),
        ("a" * 54, "a" * 54),
        ("a" * 55, "a" * 54),
        (
            "sat12-bq167-9ce84703-42a0-45a4-bdd1-bfc6d47b1ef7-728220639B35.local",
            "sat12-bq167-9ce84703-42a0-45a4-bdd1-bfc6d47b1ef7-72822",
        ),
        ("mac-" + "界" * 19, "mac-" + "界" * 16),
        ("🖥" * 15, "🖥" * 13),
    ],
    ids=["short", "exact-limit", "over-limit", "ci-runner", "unicode-3-byte", "unicode-4-byte"],
)
async def test_generated_identity_registers_with_discovery(monkeypatch, source, hostname, expected_prefix):
    monkeypatch.setattr(agent_module.socket, "gethostname", lambda: hostname if source == "hostname" else "base.local")
    monkeypatch.setattr(agent_module.secrets_mod, "token_hex", lambda size: "0123abcd")
    for detector, value in {
        "_detect_gpu_cores": 8,
        "_detect_ram_gb": 16.0,
        "_detect_memory_bandwidth": 100.0,
        "_detect_chip_name": "Apple M4",
        "_check_mps_available": False,
        "_check_mlx_available": False,
    }.items():
        monkeypatch.setattr(agent_module, detector, Mock(return_value=value))
    monkeypatch.setattr(agent_module, "get_thermal_state", lambda: SimpleNamespace(pressure=ThermalPressure.NOMINAL))
    monkeypatch.setattr(agent_module, "get_network_topology", NetworkTopology)
    agent = agent_module.PoolAgent(name=hostname if source == "name" else None, port=0, data_port=0)
    # Keep the real discovery builder and ServiceInfo validation; replace only
    # multicast publication so these cases cannot discover other test fleets.
    agent._discovery._async_zeroconf = SimpleNamespace(
        async_register_service=AsyncMock(),
        async_unregister_service=AsyncMock(return_value=None),
        async_close=AsyncMock(),
    )
    monkeypatch.setattr(agent._discovery, "start_discovery", Mock())
    try:
        await agent.start()
        assert agent.hardware.hostname == hostname
        assert agent.node_id == f"{expected_prefix}-0123abcd"
        assert len(agent.node_id.encode("utf-8")) <= 63
        info = agent._discovery._service_info
        assert info.properties[b"node_id"] == agent.node_id.encode("utf-8")
        discovered = PoolServiceListener()._parse_service_info(info)
        assert discovered is not None
        assert discovered.node_id == agent.node_id
        assert agent.registry.get_node(agent.node_id).is_alive
    finally:
        await agent.stop()
