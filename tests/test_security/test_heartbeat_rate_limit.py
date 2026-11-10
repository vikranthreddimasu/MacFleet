"""Tests for v2.2 PR 6 heartbeat rate limiter (Issue 22).

Covers:
    - Repeated auth failures from the same IP earn a ban after N attempts
    - A banned IP gets zero response on subsequent pings (handler exits early)
    - A successful ping clears the failure counter
    - Slowloris-style timeouts count as failures (deter DoS)
    - Tightened 1s read timeout replaces the old 5s timeout
"""

from __future__ import annotations

import asyncio
import secrets
import time

from macfleet.pool.agent import HEARTBEAT_READ_TIMEOUT_SEC, PoolAgent
from macfleet.security.auth import (
    RATE_LIMIT_MAX_FAILURES,
    SecurityConfig,
    create_client_ssl_context,
    heartbeat_channel_key,
    sign_heartbeat,
    sign_heartbeat_with_hw,
)


def _start_agent(token: str = "fleet-token") -> PoolAgent:
    """Build an agent with a synthetic HardwareProfile, skip network bring-up."""
    from macfleet.engines.base import HardwareProfile

    agent = PoolAgent(
        token=token, tls=True, port=50051, data_port=50052,
    )
    agent.hardware = HardwareProfile(
        hostname="test-host",
        node_id="test-host-abcd1234",
        gpu_cores=10,
        ram_gb=24.0,
        memory_bandwidth_gbps=300.0,
        has_ane=True,
        chip_name="Apple M4 Pro (test)",
        mps_available=True,
        mlx_available=True,
    )
    return agent


class TestHeartbeatReadTimeout:
    def test_timeout_constant_is_one_second(self):
        """v2.2 PR 6: read timeout tightened from 5s → 1s."""
        assert HEARTBEAT_READ_TIMEOUT_SEC == 1.0

    async def test_slow_client_times_out_fast(self):
        """A client that stalls after connect gets dropped in ~1s, not 5s."""
        agent = _start_agent()
        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0, ssl=agent._heartbeat_ssl_ctx,
        )
        port = server.sockets[0].getsockname()[1]

        # Open a connection but never send anything
        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=create_client_ssl_context())
        start = time.monotonic()
        # Server will timeout and close. Read returns EOF quickly.
        _ = await asyncio.wait_for(reader.read(1024), timeout=3.0)
        elapsed = time.monotonic() - start

        # Tightened timeout: well under 2s total (1s read timeout + a little slack)
        assert elapsed < 2.0, f"slow client held server {elapsed:.2f}s (> 2s)"

        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionResetError):
            pass
        server.close()
        await server.wait_closed()


class TestHeartbeatRateLimiter:
    async def test_banned_ip_gets_dropped_before_read(self):
        """Once an IP is banned, the handler closes the connection without reading."""
        agent = _start_agent("fleet-token")
        # Pre-seed the limiter with failures past the threshold (avoids waiting
        # through exponential-backoff sleeps for each of the 5 natural failures)
        for _ in range(RATE_LIMIT_MAX_FAILURES):
            agent._heartbeat_rate_limiter.record_failure("127.0.0.1")
        assert agent._heartbeat_rate_limiter.is_banned("127.0.0.1")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0, ssl=agent._heartbeat_ssl_ctx,
        )
        port = server.sockets[0].getsockname()[1]

        # Even a VALID ping from a banned IP should get no reply
        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=create_client_ssl_context())
        fleet_key = heartbeat_channel_key(agent._security.fleet_key, agent._heartbeat_cert)
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(fleet_key, "honest-peer", nonce)
        try:
            writer.write(f"APING honest-peer {nonce.hex()} {sig.hex()}\n".encode())
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            # Server already closed the connection → exactly what we want
            pass
        try:
            data = await asyncio.wait_for(reader.read(1024), timeout=2.0)
            assert data == b"", f"banned IP should get no reply, got {data!r}"
        except (ConnectionResetError, asyncio.IncompleteReadError):
            # Connection reset is the expected outcome for a banned IP
            pass

        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionResetError):
            pass
        server.close()
        await server.wait_closed()

    async def test_wrong_key_ping_records_one_failure(self):
        """A single bad-sig ping bumps the failure counter by 1."""
        agent = _start_agent("correct-token")
        wrong_key = SecurityConfig(token="attacker-token").fleet_key

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0, ssl=agent._heartbeat_ssl_ctx,
        )
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=create_client_ssl_context())
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(wrong_key, "attacker", nonce)
        writer.write(f"APING attacker {nonce.hex()} {sig.hex()}\n".encode())
        await writer.drain()
        try:
            await asyncio.wait_for(reader.read(1024), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionResetError):
            pass

        entry = agent._heartbeat_rate_limiter._failures.get("127.0.0.1")
        assert entry is not None, "wrong-key APING should record a failure"
        assert entry[0] == 1

        server.close()
        await server.wait_closed()

    async def test_successful_ping_clears_failures(self):
        """A good ping resets the failure count for that IP."""
        agent = _start_agent("fleet-token")
        fleet_key = heartbeat_channel_key(agent._security.fleet_key, agent._heartbeat_cert)

        # Pre-seed a couple of failures (below the ban threshold)
        agent._heartbeat_rate_limiter.record_failure("127.0.0.1")
        agent._heartbeat_rate_limiter.record_failure("127.0.0.1")
        assert agent._heartbeat_rate_limiter._failures.get("127.0.0.1")[0] == 2
        assert not agent._heartbeat_rate_limiter.is_banned("127.0.0.1")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0, ssl=agent._heartbeat_ssl_ctx,
        )
        port = server.sockets[0].getsockname()[1]

        # A good ping should clear the counter
        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=create_client_ssl_context())
        nonce = secrets.token_bytes(16)
        sig = sign_heartbeat(fleet_key, "honest-peer", nonce)
        writer.write(f"APING honest-peer {nonce.hex()} {sig.hex()}\n".encode())
        await writer.drain()
        # Handler backoff (0.5s + 1.0s total sleep from count=2 → backoff=1s);
        # give the read a generous timeout to cover backoff.
        response = await asyncio.wait_for(reader.readline(), timeout=5.0)
        assert response.startswith(b"APONG")

        # Failure count cleared
        assert not agent._heartbeat_rate_limiter.is_banned("127.0.0.1")
        entry = agent._heartbeat_rate_limiter._failures.get("127.0.0.1")
        assert entry is None, f"good ping should clear failures, got {entry!r}"

        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionResetError):
            pass
        server.close()
        await server.wait_closed()

    async def test_tampered_hw_v2_records_failure(self):
        """APING v2 with bad HMAC over lying HW bumps the per-IP failure count."""
        from macfleet.comm.transport import HardwareExchange

        agent = _start_agent("fleet-token")
        fleet_key = heartbeat_channel_key(agent._security.fleet_key, agent._heartbeat_cert)

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0, ssl=agent._heartbeat_ssl_ctx,
        )
        port = server.sockets[0].getsockname()[1]

        # One bad APING v2
        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=create_client_ssl_context())
        nonce = secrets.token_bytes(16)
        real_hw = HardwareExchange(gpu_cores=8).to_json_bytes()
        lying_hw = HardwareExchange(gpu_cores=128).to_json_bytes()
        sig = sign_heartbeat_with_hw(fleet_key, "pinger", nonce, real_hw)
        # Wire the lying HW but sig was over the honest HW
        writer.write(
            f"APING pinger {nonce.hex()} {sig.hex()} {lying_hw.hex()}\n".encode()
        )
        await writer.drain()
        try:
            await asyncio.wait_for(reader.read(1024), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionResetError):
            pass

        # Failure recorded for 127.0.0.1
        entry = agent._heartbeat_rate_limiter._failures.get("127.0.0.1")
        assert entry is not None, "tampered APING v2 should record a failure"
        assert entry[0] == 1

        server.close()
        await server.wait_closed()

    async def test_plain_ping_to_secure_server_records_failure(self):
        """A plain PING to an authenticated server counts as a failed attempt."""
        agent = _start_agent("fleet-token")

        server = await asyncio.start_server(
            agent._handle_heartbeat_ping, "127.0.0.1", 0, ssl=agent._heartbeat_ssl_ctx,
        )
        port = server.sockets[0].getsockname()[1]

        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=create_client_ssl_context())
        writer.write(b"PING attacker\n")
        await writer.drain()
        try:
            await asyncio.wait_for(reader.read(1024), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionResetError):
            pass

        entry = agent._heartbeat_rate_limiter._failures.get("127.0.0.1")
        assert entry is not None, "plain PING to secure fleet should record failure"
        assert entry[0] == 1

        server.close()
        await server.wait_closed()
