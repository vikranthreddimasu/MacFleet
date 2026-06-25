import pytest

from macfleet.security.enrollment import (
    EnrollmentError,
    EnrollmentServer,
    format_pair_command,
    generate_enrollment_code,
    normalize_enrollment_code,
    parse_host_port,
    request_enrollment,
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
