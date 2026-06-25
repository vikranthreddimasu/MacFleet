"""Tests for gRPC control-plane authentication."""

import pytest

from macfleet.comm.grpc_service import (
    AUTH_METADATA_KEY,
    ClusterControlClient,
    ClusterControlServicer,
)
from macfleet.comm.proto import control_pb2
from macfleet.core.config import ClusterConfig, ClusterState, NodeConfig, NodeRole

VALID_TOKEN = "0123456789abcdef"


class FakeContext:
    """Minimal grpc.ServicerContext for direct servicer tests."""

    def __init__(self, metadata=()):
        self._metadata = metadata

    def invocation_metadata(self):
        return self._metadata

    def abort(self, code, details):
        raise PermissionError(details)


def _state_with_master() -> ClusterState:
    state = ClusterState()
    state.add_node(
        NodeConfig(
            hostname="master",
            ip_address="10.0.0.1",
            gpu_cores=16,
            ram_gb=24,
            memory_bandwidth_gbps=273.0,
            tensor_port=50052,
            rank=0,
            workload_weight=1.0,
        )
    )
    return state


def _register_request() -> control_pb2.RegisterRequest:
    return control_pb2.RegisterRequest(
        hostname="worker",
        ip_address="10.0.0.2",
        gpu_cores=10,
        ram_gb=16,
        memory_bandwidth_gbps=120.0,
        tensor_port=50053,
    )


def test_legacy_no_token_allows_registration():
    servicer = ClusterControlServicer(
        cluster_state=_state_with_master(),
        tensor_addr="10.0.0.1",
        tensor_port=50052,
    )

    response = servicer.Register(_register_request(), FakeContext())

    assert response.assigned_rank == 1


def test_protected_servicer_rejects_missing_token():
    state = _state_with_master()
    servicer = ClusterControlServicer(
        cluster_state=state,
        tensor_addr="10.0.0.1",
        tensor_port=50052,
        auth_token=VALID_TOKEN,
    )

    with pytest.raises(PermissionError, match="auth token"):
        servicer.Register(_register_request(), FakeContext())

    assert state.world_size == 1


def test_protected_servicer_accepts_matching_token():
    servicer = ClusterControlServicer(
        cluster_state=_state_with_master(),
        tensor_addr="10.0.0.1",
        tensor_port=50052,
        auth_token=VALID_TOKEN,
    )
    context = FakeContext(metadata=((AUTH_METADATA_KEY, VALID_TOKEN),))

    response = servicer.Register(_register_request(), context)

    assert response.assigned_rank == 1


def test_client_loads_auth_metadata_from_environment(monkeypatch):
    monkeypatch.setenv("MACFLEET_AUTH_TOKEN", VALID_TOKEN)

    client = ClusterControlClient("10.0.0.1", 50051)

    assert client._metadata() == ((AUTH_METADATA_KEY, VALID_TOKEN),)


def test_cluster_config_does_not_serialize_auth_token(monkeypatch):
    monkeypatch.setenv("MACFLEET_AUTH_TOKEN", VALID_TOKEN)

    cfg = ClusterConfig(role=NodeRole.MASTER)
    data = cfg.to_dict()

    assert cfg.auth_token == VALID_TOKEN
    assert "auth_token" not in data
    assert data["auth_required"] is True


def test_short_auth_token_rejected():
    with pytest.raises(ValueError, match="at least"):
        ClusterConfig(role=NodeRole.MASTER, auth_token="short")
