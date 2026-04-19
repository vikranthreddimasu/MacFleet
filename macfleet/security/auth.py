"""Security primitives for MacFleet fleet isolation and authentication.

Provides:
- SecurityConfig: fleet key derivation, mDNS service type scoping
- HMAC challenge-response: mutual authentication without transmitting tokens
- TLS helpers: self-signed ephemeral certs for transport encryption
- Heartbeat authentication: HMAC-signed PING/PONG messages
- Gradient validation: NaN/Inf/magnitude bounds checking
- Rate limiting: per-IP exponential backoff on failed auth

v2.2 PR 3 (Issue 9+21+A6+A12): TLS cert generation migrated from subprocess
to the `cryptography` library — no openssl binary dependency, no EC/RSA
fallback fragility. Certs + keys live in-memory, are written to user-only
temp files only for `SSLContext.load_cert_chain` to consume (stdlib ssl
requires file paths), then immediately unlinked.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import logging
import os
import secrets
import ssl
import stat
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)

# Default mDNS service type (open fleet, no isolation)
DEFAULT_SERVICE_TYPE = "_macfleet._tcp.local."

# Challenge size in bytes
CHALLENGE_SIZE = 32

# Gradient validation limits
GRADIENT_MAX_MAGNITUDE = 1e6
GRADIENT_MAX_NUMEL = 2_000_000_000  # ~8GB at float32

# Rate limiting defaults
RATE_LIMIT_MAX_FAILURES = 5
RATE_LIMIT_BASE_DELAY_SEC = 0.5
RATE_LIMIT_MAX_DELAY_SEC = 60.0
RATE_LIMIT_BAN_DURATION_SEC = 300.0

# Environment variable for token (avoids CLI arg exposure in `ps`)
TOKEN_ENV_VAR = "MACFLEET_TOKEN"

# Minimum token length to prevent trivially bruteforceable keys
MIN_TOKEN_LENGTH = 8

# Token file location
TOKEN_DIR = os.path.expanduser("~/.macfleet")
TOKEN_FILE = os.path.join(TOKEN_DIR, "fleet-token")

# Auto-generated token length (hex chars → 32 bytes of entropy)
AUTO_TOKEN_LENGTH = 32


def _read_token_file() -> Optional[str]:
    """Read saved fleet token from ~/.macfleet/fleet-token.

    Warns if the file is readable by group or other (v2.2 PR 3 / A6).
    The warning is non-blocking — we still return the token so the user's
    workflow isn't broken, but the log tells them another local user can
    read their fleet credential.
    """
    try:
        st = os.stat(TOKEN_FILE)
    except FileNotFoundError:
        return None
    _check_token_file_mode(st.st_mode)
    try:
        with open(TOKEN_FILE) as f:
            token = f.read().strip()
            return token if token else None
    except FileNotFoundError:
        # Race: someone deleted it between stat and open
        return None


def _check_token_file_mode(st_mode: int) -> None:
    """Log a warning if the token file has group or other permission bits set."""
    perms = stat.S_IMODE(st_mode)
    if perms & 0o077:
        logger.warning(
            "Fleet token at %s has permissive mode %o (group/other bits set). "
            "Another local user can read your fleet credential. Fix with: "
            "`chmod 600 %s`",
            TOKEN_FILE, perms, TOKEN_FILE,
        )


def _write_token_file(token: str) -> None:
    """Save fleet token to ~/.macfleet/fleet-token with restricted permissions.

    Creates the file with mode 0600 via O_CREAT. If the file already exists
    with broader permissions, O_CREAT won't tighten them — so we explicitly
    chmod after the write to enforce 0o600 on every call (repairs a
    previously-mis-permissioned file).
    """
    os.makedirs(TOKEN_DIR, mode=0o700, exist_ok=True)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    # Re-enforce 0600 in case the file pre-existed with broader mode
    os.chmod(TOKEN_FILE, 0o600)


def generate_fleet_token() -> str:
    """Generate a cryptographically random fleet token."""
    return secrets.token_hex(AUTO_TOKEN_LENGTH)


def resolve_token(token: Optional[str] = None) -> Optional[str]:
    """Resolve token from explicit value or MACFLEET_TOKEN env var.

    Priority: explicit argument > environment variable > None.
    Used by SecurityConfig — does NOT read from file or auto-generate.
    """
    if token is not None:
        return token
    return os.environ.get(TOKEN_ENV_VAR)


def resolve_token_with_file(token: Optional[str] = None, *, auto_generate: bool = False) -> Optional[str]:
    """Resolve token from explicit value, env var, saved file, or auto-generate.

    Priority: explicit argument > environment variable > saved file > auto-generate.
    Used by CLI and SDK — reads from ~/.macfleet/fleet-token.
    """
    if token is not None:
        return token
    env_token = os.environ.get(TOKEN_ENV_VAR)
    if env_token is not None:
        return env_token
    saved = _read_token_file()
    if saved is not None:
        return saved
    if auto_generate:
        new_token = generate_fleet_token()
        _write_token_file(new_token)
        return new_token
    return None


class SecurityConfig:
    """Security configuration for a MacFleet pool.

    When token is None, the pool is open (no auth, no isolation).
    When token is set:
      - Fleet isolation (scoped mDNS) activates automatically
      - HMAC mutual authentication activates automatically
      - TLS encryption activates automatically (mandatory with auth)

    The raw token is never stored. Only the derived fleet_key is kept.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        fleet_id: Optional[str] = None,
        tls: bool = False,
    ):
        self.fleet_id = fleet_id

        # Resolve token from env var if not passed directly
        resolved = resolve_token(token)

        if resolved is not None:
            if len(resolved) < MIN_TOKEN_LENGTH:
                raise ValueError(
                    f"Token must be at least {MIN_TOKEN_LENGTH} characters "
                    f"(got {len(resolved)}). Short tokens are trivially bruteforceable."
                )
            # Derive fleet key immediately, then discard raw token
            effective_fleet_id = fleet_id or "default"
            self._fleet_key: Optional[bytes] = hmac_mod.new(
                resolved.encode("utf-8"),
                f"macfleet-v2:{effective_fleet_id}".encode("utf-8"),
                hashlib.sha256,
            ).digest()
            # TLS is mandatory when auth is enabled — never send
            # authenticated handshakes or gradients over plaintext
            self.tls = True
        else:
            self._fleet_key = None
            self.tls = tls  # only meaningful for open fleets (rare)

        # Precompute mDNS service type
        if self._fleet_key is not None:
            fleet_hash = hashlib.sha256(self._fleet_key).hexdigest()[:8]
            self._mdns_service_type = f"_mf-{fleet_hash}._tcp.local."
        else:
            self._mdns_service_type = DEFAULT_SERVICE_TYPE

    @property
    def fleet_key(self) -> Optional[bytes]:
        """The derived fleet-wide HMAC key (32 bytes), or None if open."""
        return self._fleet_key

    @property
    def mdns_service_type(self) -> str:
        """Fleet-scoped mDNS service type."""
        return self._mdns_service_type

    @property
    def is_secure(self) -> bool:
        """Whether authentication is enabled."""
        return self._fleet_key is not None


# ------------------------------------------------------------------ #
# Rate Limiter (per-IP exponential backoff)                           #
# ------------------------------------------------------------------ #


class AuthRateLimiter:
    """Track failed authentication attempts per IP and enforce backoff.

    After RATE_LIMIT_MAX_FAILURES consecutive failures from one IP,
    the IP is banned for RATE_LIMIT_BAN_DURATION_SEC.

    Capped at max_entries to prevent memory exhaustion from port scanning.
    Evicts the oldest entry when full.
    """

    def __init__(self, max_entries: int = 10_000):
        # ip -> (consecutive_failures, last_failure_time)
        self._failures: dict[str, tuple[int, float]] = {}
        self._max_entries = max_entries

    def record_failure(self, ip: str) -> None:
        """Record a failed auth attempt from this IP."""
        count, _ = self._failures.get(ip, (0, 0.0))
        self._failures[ip] = (count + 1, time.monotonic())
        # Evict oldest entry if over capacity
        if len(self._failures) > self._max_entries:
            oldest_ip = min(self._failures, key=lambda k: self._failures[k][1])
            self._failures.pop(oldest_ip, None)

    def record_success(self, ip: str) -> None:
        """Clear failure count for this IP on successful auth."""
        self._failures.pop(ip, None)

    def is_banned(self, ip: str) -> bool:
        """Check if this IP is temporarily banned."""
        entry = self._failures.get(ip)
        if entry is None:
            return False
        count, last_time = entry
        if count < RATE_LIMIT_MAX_FAILURES:
            return False
        elapsed = time.monotonic() - last_time
        if elapsed > RATE_LIMIT_BAN_DURATION_SEC:
            # Ban expired, reset
            self._failures.pop(ip, None)
            return False
        return True

    def get_delay(self, ip: str) -> float:
        """Get the backoff delay in seconds for this IP."""
        entry = self._failures.get(ip)
        if entry is None:
            return 0.0
        count, _ = entry
        if count == 0:
            return 0.0
        delay = RATE_LIMIT_BASE_DELAY_SEC * (2 ** min(count - 1, 10))
        return min(delay, RATE_LIMIT_MAX_DELAY_SEC)


# ------------------------------------------------------------------ #
# HMAC Challenge-Response (mutual authentication)                     #
# ------------------------------------------------------------------ #


def generate_challenge() -> bytes:
    """Generate a random challenge for the HMAC handshake."""
    return secrets.token_bytes(CHALLENGE_SIZE)


def compute_response(fleet_key: bytes, challenge: bytes) -> bytes:
    """Compute HMAC-SHA256 response to a challenge.

    Args:
        fleet_key: The derived fleet key (from SecurityConfig.fleet_key).
        challenge: The random challenge bytes to respond to.

    Returns:
        32-byte HMAC-SHA256 digest.
    """
    return hmac_mod.new(fleet_key, challenge, hashlib.sha256).digest()


def verify_response(fleet_key: bytes, challenge: bytes, response: bytes) -> bool:
    """Verify an HMAC challenge response using constant-time comparison.

    Args:
        fleet_key: The derived fleet key.
        challenge: The original challenge that was sent.
        response: The response received from the peer.

    Returns:
        True if the response is valid (peer knows the token).
    """
    expected = compute_response(fleet_key, challenge)
    return hmac_mod.compare_digest(expected, response)


# ------------------------------------------------------------------ #
# Heartbeat Authentication                                            #
# ------------------------------------------------------------------ #


def sign_heartbeat(fleet_key: bytes, node_id: str, nonce: bytes) -> bytes:
    """Sign a heartbeat message with HMAC.

    Args:
        fleet_key: The derived fleet key.
        node_id: The sender's node ID.
        nonce: Random bytes to prevent replay.

    Returns:
        32-byte HMAC-SHA256 signature.
    """
    msg = node_id.encode("utf-8") + b":" + nonce
    return hmac_mod.new(fleet_key, msg, hashlib.sha256).digest()


def verify_heartbeat(
    fleet_key: bytes, node_id: str, nonce: bytes, signature: bytes
) -> bool:
    """Verify a signed heartbeat message.

    Args:
        fleet_key: The derived fleet key.
        node_id: The claimed sender's node ID.
        nonce: The nonce from the heartbeat message.
        signature: The HMAC signature to verify.

    Returns:
        True if the signature is valid.
    """
    expected = sign_heartbeat(fleet_key, node_id, nonce)
    return hmac_mod.compare_digest(expected, signature)


# ------------------------------------------------------------------ #
# Gradient Validation (anti-poisoning)                                #
# ------------------------------------------------------------------ #


class GradientValidationError(ValueError):
    """Raised when received gradients fail validation."""
    pass


def validate_gradients(
    gradients: np.ndarray,
    max_magnitude: float = GRADIENT_MAX_MAGNITUDE,
) -> None:
    """Validate gradient array for NaN, Inf, and extreme magnitudes.

    Called after allreduce but BEFORE applying gradients to the model.
    Prevents gradient poisoning attacks where a malicious peer sends
    corrupt values that would destroy training on all nodes.

    Args:
        gradients: The averaged gradient array from allreduce.
        max_magnitude: Maximum allowed absolute value for any gradient element.

    Raises:
        GradientValidationError: If gradients contain invalid values.
    """
    # Single pass for NaN+Inf (avoids 2 separate scans + temp arrays)
    if not np.isfinite(gradients).all():
        if np.isnan(gradients).any():
            raise GradientValidationError(
                "Received gradients contain NaN — possible poisoning attack. "
                "Gradients rejected; model state preserved."
            )
        raise GradientValidationError(
            "Received gradients contain Inf — possible poisoning attack. "
            "Gradients rejected; model state preserved."
        )
    abs_max = np.abs(gradients).max()
    if abs_max > max_magnitude:
        raise GradientValidationError(
            f"Gradient magnitude {abs_max:.2e} exceeds limit {max_magnitude:.2e} — "
            f"possible poisoning attack. Gradients rejected; model state preserved."
        )


def validate_gradient_metadata(
    original_numel: int,
    topk_count: int,
) -> None:
    """Validate compressed gradient metadata from wire protocol.

    Prevents memory allocation bombs from malicious metadata.

    Args:
        original_numel: Claimed original tensor element count.
        topk_count: Claimed number of TopK entries.

    Raises:
        GradientValidationError: If metadata is suspicious.
    """
    if original_numel < 0 or original_numel > GRADIENT_MAX_NUMEL:
        raise GradientValidationError(
            f"Suspicious gradient metadata: original_numel={original_numel} "
            f"(limit={GRADIENT_MAX_NUMEL})"
        )
    if topk_count < 0 or topk_count > original_numel:
        raise GradientValidationError(
            f"Suspicious gradient metadata: topk_count={topk_count} > "
            f"original_numel={original_numel}"
        )


# ------------------------------------------------------------------ #
# TLS Helpers (mandatory when auth is enabled)                        #
# ------------------------------------------------------------------ #


def create_server_ssl_context() -> ssl.SSLContext:
    """Create a server-side TLS context with an ephemeral self-signed cert.

    Authentication is handled by HMAC challenge-response, not certificates.
    TLS is used purely for encryption of gradient data in transit.

    Cert + private key are generated in-process via the `cryptography` library.
    They are written to user-only temp files ($TMPDIR is user-scoped on macOS)
    just long enough for `SSLContext.load_cert_chain` to consume them (stdlib
    ssl requires file paths), then immediately unlinked — the key stays in
    the SSLContext's memory for the lifetime of the server but never sits
    on disk after this function returns.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    cert_pem, key_pem = _generate_cert_bytes()
    certfile, keyfile = _write_ephemeral_pem(cert_pem, key_pem)
    try:
        ctx.load_cert_chain(certfile, keyfile)
    finally:
        for path in (certfile, keyfile):
            try:
                os.unlink(path)
            except OSError:
                pass
    return ctx


def create_client_ssl_context() -> ssl.SSLContext:
    """Create a client-side TLS context that accepts self-signed certs.

    No hostname verification — auth is via HMAC, TLS is encryption-only.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _generate_cert_bytes() -> tuple[bytes, bytes]:
    """Generate an ephemeral self-signed EC (P-256) cert + private key.

    Returns (cert_pem, key_pem) — both as PEM-encoded bytes, never touches
    disk. SHA-256 signature, 25-hour validity (5-min clock-skew leeway on
    the not-before bound so agents behind slightly-off clocks still accept
    the cert), SubjectAlternativeName=localhost for server-name checks if
    the client ever enables them.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "macfleet-node"),
    ])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


def _write_ephemeral_pem(cert_pem: bytes, key_pem: bytes) -> tuple[str, str]:
    """Write PEM blobs to mode-0600 tempfiles. Returns (cert_path, key_path).

    Uses `tempfile.mkstemp`, which on macOS resolves to $TMPDIR
    (/var/folders/xx/.../T/) — user-owned, 0700 directory, not the shared
    /tmp. Files are created with mode 0600 by mkstemp. Caller MUST unlink
    after consuming them.
    """
    cert_fd, cert_path = tempfile.mkstemp(suffix=".pem", prefix="macfleet_cert_")
    key_fd, key_path = tempfile.mkstemp(suffix=".pem", prefix="macfleet_key_")
    try:
        os.write(cert_fd, cert_pem)
        os.write(key_fd, key_pem)
    finally:
        os.close(cert_fd)
        os.close(key_fd)
    return cert_path, key_path


def _generate_self_signed_cert() -> tuple[str, str]:
    """Deprecated shim retained for callers that imported the old name.

    Returns tempfile paths the caller must unlink. Kept for one release to
    avoid breaking anything outside this module that imported the private
    helper; `create_server_ssl_context` no longer uses it.
    """
    cert_pem, key_pem = _generate_cert_bytes()
    return _write_ephemeral_pem(cert_pem, key_pem)
