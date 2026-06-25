"""Short-lived fleet enrollment over a local authenticated TLS channel.

Legacy pairing URLs embed the permanent fleet token. They are convenient, but
they turn terminal history, screenshots, and pasteboards into long-lived fleet
credentials. The enrollment flow here keeps the permanent token on the first
Mac and exposes only a short-lived, one-time code.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Optional, TextIO

from macfleet.security.audit import audit_event
from macfleet.security.auth import (
    MIN_TOKEN_LENGTH,
    AuthRateLimiter,
    create_client_ssl_context,
    create_server_tls_context,
    tls_channel_binding_from_writer,
)

ENROLLMENT_VERSION = 1
DEFAULT_ENROLLMENT_TTL_SEC = 300
DEFAULT_ENROLLMENT_MAX_USES = 1
ENROLLMENT_READ_LIMIT_BYTES = 4096
ENROLLMENT_READ_TIMEOUT_SEC = 10.0
ENROLLMENT_NONCE_BYTES = 16
ENROLLMENT_PROOF_BYTES = 32

_LABEL_CLIENT = b"MFENROLLv1-C:"


class EnrollmentError(ValueError):
    """Raised when short-lived enrollment fails."""


@dataclass(frozen=True)
class EnrollmentResult:
    """Token material returned by a successful enrollment."""

    token: str
    fleet_id: Optional[str]
    server_node: str


def normalize_enrollment_code(code: str) -> str:
    """Normalize a displayed enrollment code for HMAC use."""
    normalized = "".join(ch for ch in code.upper() if ch.isalnum())
    if len(normalized) < 16:
        raise EnrollmentError("Enrollment code is too short or malformed")
    return normalized


def generate_enrollment_code() -> str:
    """Generate a 120-bit human-copyable one-time code."""
    raw = base64.b32encode(secrets.token_bytes(15)).decode("ascii").rstrip("=")
    return "-".join(raw[i : i + 6] for i in range(0, len(raw), 6))


def _code_key(code: str) -> bytes:
    return hashlib.sha256(normalize_enrollment_code(code).encode("ascii")).digest()


def _proof(code: str, nonce: bytes, channel_binding: bytes) -> bytes:
    return hmac.new(
        _code_key(code),
        _LABEL_CLIENT + nonce + channel_binding,
        hashlib.sha256,
    ).digest()


def _verify_proof(code: str, nonce: bytes, channel_binding: bytes, proof: bytes) -> bool:
    return hmac.compare_digest(_proof(code, nonce, channel_binding), proof)


def parse_host_port(host_port: str) -> tuple[str, int]:
    """Parse `host:port` for CLI enrollment."""
    if not host_port or ":" not in host_port:
        raise EnrollmentError("Expected --host in HOST:PORT form")
    host, port_s = host_port.rsplit(":", 1)
    if not host:
        raise EnrollmentError("Enrollment host is empty")
    try:
        port = int(port_s)
    except ValueError as e:
        raise EnrollmentError(f"Enrollment port is not an integer: {port_s!r}") from e
    if not (0 < port < 65536):
        raise EnrollmentError(f"Enrollment port {port} is outside 1..65535")
    return host, port


class EnrollmentServer:
    """One-shot server that hands out a fleet token to clients with the code."""

    def __init__(
        self,
        *,
        token: str,
        fleet_id: Optional[str] = None,
        node_id: str = "unknown",
        host: str = "0.0.0.0",
        port: int = 0,
        ttl_sec: float = DEFAULT_ENROLLMENT_TTL_SEC,
        max_uses: int = DEFAULT_ENROLLMENT_MAX_USES,
        code: Optional[str] = None,
    ):
        if len(token) < MIN_TOKEN_LENGTH:
            raise EnrollmentError(
                f"Fleet token is too short for enrollment ({len(token)} chars)"
            )
        if ttl_sec <= 0:
            raise EnrollmentError("Enrollment TTL must be positive")
        if max_uses < 1:
            raise EnrollmentError("Enrollment max_uses must be >= 1")

        self.token = token
        self.fleet_id = fleet_id
        self.node_id = node_id
        self.host = host
        self.requested_port = port
        self.ttl_sec = ttl_sec
        self.max_uses = max_uses
        self.code = code or generate_enrollment_code()
        self._server: Optional[asyncio.Server] = None
        self._cert_binding = b""
        self._started_at_monotonic = 0.0
        self._expires_at_epoch = 0.0
        self._uses = 0
        self._use_lock = asyncio.Lock()
        self._rate_limiter = AuthRateLimiter()

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            return 0
        return int(self._server.sockets[0].getsockname()[1])

    @property
    def expires_at_epoch(self) -> float:
        return self._expires_at_epoch

    @property
    def is_running(self) -> bool:
        return self._server is not None

    async def start(self) -> None:
        """Start the enrollment listener."""
        if self._server is not None:
            return
        ssl_ctx, self._cert_binding = create_server_tls_context()
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.requested_port,
            ssl=ssl_ctx,
        )
        self._started_at_monotonic = time.monotonic()
        self._expires_at_epoch = time.time() + self.ttl_sec
        audit_event(
            "enrollment.started",
            node_id=self.node_id,
            fleet_id=self.fleet_id,
            port=self.bound_port,
            ttl_sec=self.ttl_sec,
            max_uses=self.max_uses,
        )

    async def stop(self) -> None:
        """Stop the enrollment listener."""
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        audit_event(
            "enrollment.stopped",
            node_id=self.node_id,
            fleet_id=self.fleet_id,
            uses=self._uses,
        )

    def _expired(self) -> bool:
        return (time.monotonic() - self._started_at_monotonic) > self.ttl_sec

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        peer_ip = peer[0] if peer else "unknown"
        try:
            if self._rate_limiter.is_banned(peer_ip):
                audit_event("enrollment.rejected", reason="rate_limited", peer_ip=peer_ip)
                return

            delay = self._rate_limiter.get_delay(peer_ip)
            if delay > 0:
                await asyncio.sleep(delay)

            if self._expired() or self._uses >= self.max_uses:
                audit_event("enrollment.rejected", reason="expired_or_used", peer_ip=peer_ip)
                await self._write_response(writer, {"ok": False, "error": "enrollment expired"})
                return

            line = await asyncio.wait_for(
                reader.readline(),
                timeout=ENROLLMENT_READ_TIMEOUT_SEC,
            )
            if not line:
                audit_event("enrollment.rejected", reason="empty_request", peer_ip=peer_ip)
                return
            if len(line) > ENROLLMENT_READ_LIMIT_BYTES:
                self._rate_limiter.record_failure(peer_ip)
                audit_event("enrollment.rejected", reason="bad_request_size", peer_ip=peer_ip)
                return
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                nonce = bytes.fromhex(str(request["nonce"]))
                received = bytes.fromhex(str(request["proof"]))
                if len(nonce) != ENROLLMENT_NONCE_BYTES:
                    raise ValueError("nonce has wrong size")
                if len(received) != ENROLLMENT_PROOF_BYTES:
                    raise ValueError("proof has wrong size")
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                self._rate_limiter.record_failure(peer_ip)
                audit_event("enrollment.rejected", reason="malformed_request", peer_ip=peer_ip)
                return

            if request.get("version") != ENROLLMENT_VERSION:
                self._rate_limiter.record_failure(peer_ip)
                audit_event("enrollment.rejected", reason="version_mismatch", peer_ip=peer_ip)
                return

            if not _verify_proof(self.code, nonce, self._cert_binding, received):
                self._rate_limiter.record_failure(peer_ip)
                audit_event("enrollment.rejected", reason="bad_code_or_mitm", peer_ip=peer_ip)
                return

            async with self._use_lock:
                if self._expired() or self._uses >= self.max_uses:
                    audit_event(
                        "enrollment.rejected",
                        reason="expired_or_used",
                        peer_ip=peer_ip,
                    )
                    await self._write_response(
                        writer, {"ok": False, "error": "enrollment expired"}
                    )
                    return
                self._uses += 1

            self._rate_limiter.record_success(peer_ip)
            await self._write_response(
                writer,
                {
                    "ok": True,
                    "version": ENROLLMENT_VERSION,
                    "token": self.token,
                    "fleet_id": self.fleet_id,
                    "server_node": self.node_id,
                    "expires_at": self._expires_at_epoch,
                },
            )
            audit_event(
                "enrollment.completed",
                node_id=self.node_id,
                fleet_id=self.fleet_id,
                peer_ip=peer_ip,
                uses=self._uses,
            )
            if self._uses >= self.max_uses and self._server is not None:
                self._server.close()
        except asyncio.TimeoutError:
            self._rate_limiter.record_failure(peer_ip)
            audit_event("enrollment.rejected", reason="timeout", peer_ip=peer_ip)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def _write_response(self, writer: asyncio.StreamWriter, payload: dict) -> None:
        writer.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        await writer.drain()


async def request_enrollment(
    host: str,
    port: int,
    code: str,
    *,
    node_name: Optional[str] = None,
    timeout_sec: float = ENROLLMENT_READ_TIMEOUT_SEC,
) -> EnrollmentResult:
    """Fetch fleet credentials from a short-lived enrollment server."""
    ssl_ctx = create_client_ssl_context()
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port, ssl=ssl_ctx),
        timeout=timeout_sec,
    )
    try:
        binding = tls_channel_binding_from_writer(writer)
        if not binding:
            raise EnrollmentError("Enrollment server did not negotiate TLS")
        nonce = secrets.token_bytes(16)
        payload = {
            "version": ENROLLMENT_VERSION,
            "node_name": node_name,
            "nonce": nonce.hex(),
            "proof": _proof(code, nonce, binding).hex(),
        }
        writer.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout_sec)
        if not line:
            raise EnrollmentError("Enrollment server closed without a response")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise EnrollmentError(f"Enrollment server returned malformed JSON: {e}") from e
        if not isinstance(response, dict):
            raise EnrollmentError("Enrollment server returned a malformed response")
        if not response.get("ok"):
            raise EnrollmentError(str(response.get("error") or "enrollment rejected"))
        token = response.get("token")
        if not isinstance(token, str) or len(token) < MIN_TOKEN_LENGTH:
            raise EnrollmentError("Enrollment server returned an invalid token")
        fleet_id = response.get("fleet_id")
        if fleet_id is not None and not isinstance(fleet_id, str):
            raise EnrollmentError("Enrollment server returned an invalid fleet id")
        server_node = response.get("server_node")
        if server_node is not None and not isinstance(server_node, str):
            raise EnrollmentError("Enrollment server returned an invalid server node")
        result = EnrollmentResult(
            token=token,
            fleet_id=fleet_id,
            server_node=server_node or "unknown",
        )
        audit_event(
            "pairing.enrolled",
            server_node=result.server_node,
            fleet_id=result.fleet_id,
            host=host,
            port=port,
        )
        return result
    except (OSError, asyncio.TimeoutError) as e:
        raise EnrollmentError(f"Enrollment failed: {e}") from e
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


def format_pair_command(host: str, port: int, code: str) -> str:
    """Return the command a second Mac should run."""
    return f"macfleet pair --host {host}:{port} --code {code}"


def print_enrollment_info(
    host: str,
    port: int,
    code: str,
    expires_at_epoch: float,
    *,
    out: Optional[TextIO] = None,
    to_pasteboard: bool = True,
) -> str:
    """Render enrollment instructions without exposing the permanent token."""
    command = format_pair_command(host, port, code)
    expires_local = time.strftime("%H:%M:%S", time.localtime(expires_at_epoch))
    rendered = "\n".join(
        [
            "",
            "Pair another Mac with this one-time command:",
            f"  {command}",
            f"Code expires at {expires_local} and can be used once.",
            "",
        ]
    )
    if to_pasteboard:
        try:
            from macfleet.security.bootstrap import copy_to_pasteboard

            copy_to_pasteboard(command)
        except OSError:
            pass
    if out is not None:
        out.write(rendered + "\n")
    return rendered
