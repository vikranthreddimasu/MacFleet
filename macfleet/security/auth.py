"""Security primitives for MacFleet fleet isolation and authentication.

Provides:
- SecurityConfig: fleet key derivation, mDNS service type scoping
- HMAC challenge-response: mutual authentication without transmitting tokens
- TLS helpers: self-signed ephemeral certs for transport encryption
- Heartbeat authentication: HMAC-signed PING/PONG messages
- Gradient validation: NaN/Inf/magnitude bounds checking
- Rate limiting: per-IP exponential backoff on failed auth

Zero new dependencies — uses stdlib hmac, hashlib, secrets, ssl.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import logging
import os
import secrets
import ssl
import tempfile
import time
from typing import Optional

import numpy as np

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


def resolve_token(token: Optional[str] = None) -> Optional[str]:
    """Resolve token from explicit value or MACFLEET_TOKEN env var.

    Priority: explicit argument > environment variable > None.
    """
    if token is not None:
        return token
    return os.environ.get(TOKEN_ENV_VAR)


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
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Generate ephemeral self-signed cert
    certfile, keyfile = _generate_self_signed_cert()
    ctx.load_cert_chain(certfile, keyfile)

    # Clean up temp files
    try:
        os.unlink(certfile)
        os.unlink(keyfile)
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


def _generate_self_signed_cert() -> tuple[str, str]:
    """Generate a temporary self-signed certificate and key.

    Returns (certfile_path, keyfile_path). Caller should clean up.
    Uses OpenSSL CLI since stdlib ssl doesn't have cert generation.
    """
    import subprocess

    cert_fd, certfile = tempfile.mkstemp(suffix=".pem", prefix="macfleet_cert_")
    key_fd, keyfile = tempfile.mkstemp(suffix=".pem", prefix="macfleet_key_")
    os.close(cert_fd)
    os.close(key_fd)

    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "ec",
                "-pkeyopt", "ec_paramgen_curve:prime256v1",
                "-keyout", keyfile, "-out", certfile,
                "-days", "1", "-nodes",
                "-subj", "/CN=macfleet-node",
            ],
            capture_output=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # Fallback: RSA if EC isn't available
        try:
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048",
                    "-keyout", keyfile, "-out", certfile,
                    "-days", "1", "-nodes",
                    "-subj", "/CN=macfleet-node",
                ],
                capture_output=True,
                timeout=10,
                check=True,
            )
        except Exception:
            logger.warning("Failed to generate self-signed cert. TLS will not work.")
            raise

    return certfile, keyfile
