import asyncio
import json
import secrets
import time

import pytest

from macfleet.security.auth import RATE_LIMIT_MAX_FAILURES
from macfleet.security.enrollment import (
    ENROLLMENT_VERSION,
    EnrollmentError,
    EnrollmentServer,
    _proof,
    format_pair_command,
    generate_enrollment_code,
    normalize_enrollment_code,
    parse_host_port,
    request_enrollment,
)


class _ImmediateReader:
    def __init__(self, line: bytes):
        self._line = line

    async def readline(self) -> bytes:
        return self._line


class _ConcurrentReadBarrier:
    def __init__(self, waiters: int):
        self._waiters = waiters
        self._count = 0
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self._count += 1
        if self._count >= self._waiters:
            self.ready.set()
        await self.release.wait()


class _BarrierReader:
    def __init__(self, line: bytes, barrier: _ConcurrentReadBarrier):
        self._line = line
        self._barrier = barrier

    async def readline(self) -> bytes:
        await self._barrier.wait()
        return self._line


class _FakeWriter:
    def __init__(self):
        self.writes: list[bytes] = []
        self.closed = False

    def get_extra_info(self, name: str):
        if name == "peername":
            return ("127.0.0.1", 54321)
        return None

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


def _prepare_started_server(server: EnrollmentServer) -> None:
    server._cert_binding = b"test-channel-binding"
    server._started_at_monotonic = time.monotonic()
    server._expires_at_epoch = time.time() + server.ttl_sec


def _valid_enrollment_line(server: EnrollmentServer) -> bytes:
    nonce = secrets.token_bytes(16)
    payload = {
        "version": ENROLLMENT_VERSION,
        "nonce": nonce.hex(),
        "proof": _proof(server.code, nonce, server._cert_binding).hex(),
    }
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


async def _request_with_response_line(line: bytes, monkeypatch):
    async def fake_open_connection(*args, **kwargs):
        return _ImmediateReader(line), _FakeWriter()

    monkeypatch.setattr(
        "macfleet.security.enrollment.asyncio.open_connection",
        fake_open_connection,
    )
    monkeypatch.setattr(
        "macfleet.security.enrollment.tls_channel_binding_from_writer",
        lambda writer: b"test-channel-binding",
    )
    return await request_enrollment(
        "127.0.0.1",
        12345,
        "ABCD-EFGH-IJKL-MNOP",
        timeout_sec=1,
    )


def test_generate_enrollment_code_is_grouped_and_normalizable():
    code = generate_enrollment_code()
    assert "-" in code
    normalized = normalize_enrollment_code(code)
    assert normalized.isalnum()
    assert len(normalized) >= 16


def test_parse_host_port_validates_input():
    assert parse_host_port("127.0.0.1:50051") == ("127.0.0.1", 50051)
    with pytest.raises(EnrollmentError):
        parse_host_port("127.0.0.1")
    with pytest.raises(EnrollmentError):
        parse_host_port("127.0.0.1:not-a-port")
    with pytest.raises(EnrollmentError):
        parse_host_port("127.0.0.1:70000")


def test_format_pair_command_does_not_include_token():
    cmd = format_pair_command("192.168.1.10", 4242, "ABCD-EFGH-IJKL-MNOP")
    assert cmd == "macfleet pair --host 192.168.1.10:4242 --code ABCD-EFGH-IJKL-MNOP"
    assert "token" not in cmd.lower()


@pytest.mark.asyncio
async def test_enrollment_server_returns_token_to_valid_code():
    server = EnrollmentServer(
        token="enrollment-token-long-enough",
        fleet_id="lab",
        node_id="node-a",
        host="127.0.0.1",
        ttl_sec=30,
    )
    await server.start()
    try:
        result = await request_enrollment("127.0.0.1", server.bound_port, server.code)
    finally:
        await server.stop()

    assert result.token == "enrollment-token-long-enough"
    assert result.fleet_id == "lab"
    assert result.server_node == "node-a"


@pytest.mark.asyncio
async def test_enrollment_rejects_wrong_code():
    server = EnrollmentServer(
        token="enrollment-token-long-enough",
        host="127.0.0.1",
        ttl_sec=30,
    )
    await server.start()
    try:
        with pytest.raises(EnrollmentError):
            await request_enrollment(
                "127.0.0.1",
                server.bound_port,
                "WRONG-CODE-WRONG-CODE",
                timeout_sec=1,
            )
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_enrollment_is_single_use_by_default():
    server = EnrollmentServer(
        token="enrollment-token-long-enough",
        host="127.0.0.1",
        ttl_sec=30,
    )
    await server.start()
    try:
        first = await request_enrollment("127.0.0.1", server.bound_port, server.code)
        assert first.token == "enrollment-token-long-enough"
        with pytest.raises((EnrollmentError, OSError, ConnectionError)):
            await request_enrollment(
                "127.0.0.1",
                server.bound_port,
                server.code,
                timeout_sec=1,
            )
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_enrollment_single_use_survives_concurrent_valid_requests():
    server = EnrollmentServer(
        token="enrollment-token-long-enough",
        host="127.0.0.1",
        ttl_sec=30,
        max_uses=1,
    )
    _prepare_started_server(server)
    line = _valid_enrollment_line(server)
    barrier = _ConcurrentReadBarrier(waiters=2)
    writers = [_FakeWriter(), _FakeWriter()]

    tasks = [
        asyncio.create_task(server._handle_client(_BarrierReader(line, barrier), writer))
        for writer in writers
    ]
    await asyncio.wait_for(barrier.ready.wait(), timeout=1.0)
    barrier.release.set()
    await asyncio.gather(*tasks)

    responses = [
        json.loads(b"".join(writer.writes).decode("utf-8"))
        for writer in writers
        if writer.writes
    ]
    assert len(responses) == 2
    assert sum(1 for response in responses if response.get("ok")) == 1
    assert sum(1 for response in responses if not response.get("ok")) == 1
    assert server._uses == 1


@pytest.mark.asyncio
async def test_enrollment_rate_limits_banned_ip_before_valid_code():
    server = EnrollmentServer(
        token="enrollment-token-long-enough",
        host="127.0.0.1",
        ttl_sec=30,
    )
    await server.start()
    try:
        for _ in range(RATE_LIMIT_MAX_FAILURES):
            server._rate_limiter.record_failure("127.0.0.1")
        assert server._rate_limiter.is_banned("127.0.0.1")

        with pytest.raises(EnrollmentError):
            await request_enrollment(
                "127.0.0.1",
                server.bound_port,
                server.code,
                timeout_sec=1,
            )
        assert server._uses == 0
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_enrollment_malformed_proof_size_records_failure():
    server = EnrollmentServer(
        token="enrollment-token-long-enough",
        host="127.0.0.1",
        ttl_sec=30,
    )
    _prepare_started_server(server)
    nonce = secrets.token_bytes(16)
    line = (
        json.dumps(
            {
                "version": ENROLLMENT_VERSION,
                "nonce": nonce.hex(),
                "proof": "00",
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    writer = _FakeWriter()

    await server._handle_client(_ImmediateReader(line), writer)

    assert writer.writes == []
    assert server._uses == 0
    assert server._rate_limiter._failures["127.0.0.1"][0] == 1


@pytest.mark.asyncio
async def test_request_enrollment_rejects_non_object_response(monkeypatch):
    with pytest.raises(EnrollmentError, match="malformed response"):
        await _request_with_response_line(b"[]\n", monkeypatch)


@pytest.mark.asyncio
async def test_request_enrollment_rejects_non_string_token(monkeypatch):
    line = (
        json.dumps(
            {
                "ok": True,
                "version": ENROLLMENT_VERSION,
                "token": 123456789,
                "fleet_id": None,
                "server_node": "node-a",
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    with pytest.raises(EnrollmentError, match="invalid token"):
        await _request_with_response_line(line, monkeypatch)


@pytest.mark.asyncio
async def test_request_enrollment_rejects_non_string_fleet_id(monkeypatch):
    line = (
        json.dumps(
            {
                "ok": True,
                "version": ENROLLMENT_VERSION,
                "token": "enrollment-token-long-enough",
                "fleet_id": ["not", "a", "string"],
                "server_node": "node-a",
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    with pytest.raises(EnrollmentError, match="invalid fleet id"):
        await _request_with_response_line(line, monkeypatch)
