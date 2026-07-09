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

import errno
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

# Tokens shorter than this get a logged warning: with scrypt the keyspace
# of a random 8-char token is still expensive to brute-force, but short
# HUMAN-CHOSEN tokens are dictionary-attackable from one captured handshake.
RECOMMENDED_TOKEN_LENGTH = 16

# scrypt parameters for fleet-key derivation (v2.3 security hardening).
# Memory-hard KDF makes offline dictionary attacks against captured
# handshakes ~10^4-10^5x more expensive than the previous single
# HMAC-SHA256 derivation. n=2^14, r=8 → ~16 MB, ~30-80 ms one-time cost
# per SecurityConfig construction.
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024

# Domain-separation labels for the v3 authenticated handshake (v2.3).
# Each HMAC in the handshake covers a distinct label so a digest from one
# protocol step can never be replayed as another step's digest.
HS_LABEL_CLIENT_HELLO = b"MFHSv3-C1:"
HS_LABEL_SERVER_RESP = b"MFHSv3-S:"
HS_LABEL_CLIENT_RESP = b"MFHSv3-C2:"
HB_LABEL_RESPONSE = b"MFHBv3-R:"

# Token file location
TOKEN_DIR = os.path.expanduser("~/.macfleet")
TOKEN_FILE = os.path.join(TOKEN_DIR, "fleet-token")

# Auto-generated token length (hex chars → 32 bytes of entropy)
AUTO_TOKEN_LENGTH = 32


def _nofollow_flag() -> int:
    """Return O_NOFOLLOW when the platform exposes it."""
    return getattr(os, "O_NOFOLLOW", 0)


def _ensure_private_token_dir() -> None:
    """Create or repair the token directory with user-only permissions."""
    if os.path.islink(TOKEN_DIR):
        raise PermissionError(
            f"Refusing to use fleet token directory symlink: {TOKEN_DIR}"
        )
    os.makedirs(TOKEN_DIR, mode=0o700, exist_ok=True)
    st = os.stat(TOKEN_DIR)
    if not stat.S_ISDIR(st.st_mode):
        raise PermissionError(f"Fleet token path is not a directory: {TOKEN_DIR}")
    perms = stat.S_IMODE(st.st_mode)
    if perms & 0o077:
        os.chmod(TOKEN_DIR, 0o700)


def _read_token_file() -> Optional[str]:
    """Read saved fleet token from ~/.macfleet/fleet-token.

    Warns if the file is readable by group or other (v2.2 PR 3 / A6).
    The warning is non-blocking — we still return the token so the user's
    workflow isn't broken, but the log tells them another local user can
    read their fleet credential.
    """
    try:
        st = os.lstat(TOKEN_FILE)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode):
        logger.warning(
            "Refusing to read fleet token symlink at %s. Replace it with a "
            "regular file owned by your user and mode 0600.",
            TOKEN_FILE,
        )
        return None
    if not stat.S_ISREG(st.st_mode):
        logger.warning(
            "Refusing to read fleet token at %s because it is not a regular file.",
            TOKEN_FILE,
        )
        return None
    _check_token_file_mode(st.st_mode)
    flags = os.O_RDONLY | _nofollow_flag()
    try:
        fd = os.open(TOKEN_FILE, flags)
    except FileNotFoundError:
        # Race: someone deleted it between lstat and open
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            logger.warning("Refusing to read fleet token symlink at %s.", TOKEN_FILE)
            return None
        raise

    try:
        with os.fdopen(fd) as f:
            fd = -1
            token = f.read().strip()
            return token if token else None
    finally:
        if fd != -1:
            os.close(fd)


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
    _ensure_private_token_dir()
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _nofollow_flag()
    try:
        fd = os.open(TOKEN_FILE, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PermissionError(
                f"Refusing to write fleet token through symlink: {TOKEN_FILE}"
            ) from exc
        raise
    try:
        os.write(fd, token.encode("utf-8"))
        os.fsync(fd)
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    # Re-enforce 0600 in case the file pre-existed with broader mode
    os.chmod(TOKEN_FILE, 0o600)


def generate_fleet_token() -> str:
    """Generate a cryptographically random fleet token."""
    return secrets.token_hex(AUTO_TOKEN_LENGTH)


def save_fleet_token(token: str) -> None:
    """Persist a fleet token with private file permissions."""
    if not isinstance(token, str):
        raise ValueError("Token must be a string")
    if len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(
            f"Token must be at least {MIN_TOKEN_LENGTH} characters "
            f"(got {len(token)})."
        )
    _write_token_file(token)


def rotate_saved_fleet_token() -> str:
    """Generate and persist a new fleet token.

    Running agents keep using the token they started with; users should restart
    every Mac after rotation.
    """
    from macfleet.security.audit import audit_event

    token = generate_fleet_token()
    _write_token_file(token)
    audit_event("token.rotated", token_file=TOKEN_FILE)
    return token


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
        if fleet_id is not None and (not isinstance(fleet_id, str) or not fleet_id):
            raise ValueError("fleet_id must be a non-empty string when provided")
        if not isinstance(tls, bool):
            raise ValueError("tls must be a boolean")
        self.fleet_id = fleet_id

        # Resolve token from env var if not passed directly
        resolved = resolve_token(token)

        if resolved is not None:
            if not isinstance(resolved, str):
                raise ValueError("Token must be a string")
            if len(resolved) < MIN_TOKEN_LENGTH:
                raise ValueError(
                    f"Token must be at least {MIN_TOKEN_LENGTH} characters "
                    f"(got {len(resolved)}). Short tokens are trivially bruteforceable."
                )
            if len(resolved) < RECOMMENDED_TOKEN_LENGTH:
                logger.warning(
                    "Fleet token is only %d characters. Human-chosen short "
                    "tokens are dictionary-attackable offline from one captured "
                    "handshake. Prefer the auto-generated token (delete any "
                    "--token/MACFLEET_TOKEN override and re-run 'macfleet join') "
                    "or use %d+ random characters.",
                    len(resolved), RECOMMENDED_TOKEN_LENGTH,
                )
            # Derive fleet key immediately, then discard raw token.
            # v2.3: scrypt (memory-hard) replaces the single-HMAC derivation —
            # a captured handshake no longer permits cheap offline dictionary
            # attacks. NOTE: this changes the derived key for a given token,
            # so all nodes in a secure fleet must run the same MacFleet version.
            effective_fleet_id = fleet_id or "default"
            self._fleet_key: Optional[bytes] = hashlib.scrypt(
                resolved.encode("utf-8"),
                salt=f"macfleet-v3:{effective_fleet_id}".encode("utf-8"),
                n=SCRYPT_N,
                r=SCRYPT_R,
                p=SCRYPT_P,
                maxmem=SCRYPT_MAXMEM,
                dklen=32,
            )
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
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
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
        return min(delay, RATE_LIMIT_MAX_DELAY_SEC)  # type: ignore[no-any-return]


# ------------------------------------------------------------------ #
# HMAC Challenge-Response (mutual authentication)                     #
# ------------------------------------------------------------------ #


def generate_challenge() -> bytes:
    """Generate a random challenge for the HMAC handshake."""
    return secrets.token_bytes(CHALLENGE_SIZE)


def compute_response(
    fleet_key: bytes,
    challenge: bytes,
    *,
    label: bytes = b"",
    channel_binding: bytes = b"",
) -> bytes:
    """Compute HMAC-SHA256 response to a challenge.

    Args:
        fleet_key: The derived fleet key (from SecurityConfig.fleet_key).
        challenge: The random challenge bytes to respond to.
        label: Domain-separation label (HS_LABEL_*). Distinct labels mean a
            digest produced for one handshake step can never be replayed as
            another step's digest.
        channel_binding: SHA-256 of the server's TLS certificate (DER), or
            b"" when TLS is off. Binding the response to the TLS channel
            defeats MITM relays: an attacker that terminates TLS on both
            legs presents a different certificate to the client, so digests
            computed by the two victims disagree and the handshake fails.

    Returns:
        32-byte HMAC-SHA256 digest.
    """
    msg = label + challenge + channel_binding
    return hmac_mod.new(fleet_key, msg, hashlib.sha256).digest()


def verify_response(
    fleet_key: bytes,
    challenge: bytes,
    response: bytes,
    *,
    label: bytes = b"",
    channel_binding: bytes = b"",
) -> bool:
    """Verify an HMAC challenge response using constant-time comparison.

    Args:
        fleet_key: The derived fleet key.
        challenge: The original challenge that was sent.
        response: The response received from the peer.
        label: Domain-separation label (must match the signer's).
        channel_binding: TLS channel binding (must match the signer's).

    Returns:
        True if the response is valid (peer knows the token).
    """
    expected = compute_response(
        fleet_key, challenge, label=label, channel_binding=channel_binding,
    )
    return hmac_mod.compare_digest(expected, response)


def compute_client_hello_proof(
    fleet_key: bytes,
    local_id: str,
    challenge: bytes,
    channel_binding: bytes = b"",
) -> bytes:
    """Proof of token knowledge carried in the FIRST handshake message.

    v2.3 security hardening: before this existed, a secure server would
    compute and send HMAC(fleet_key, attacker_chosen_challenge) — plus its
    signed hardware profile — to ANY unauthenticated connector. That was a
    free offline brute-force oracle and hardware reconnaissance for anyone
    on the LAN. With the hello proof, the server verifies the client knows
    the token BEFORE revealing anything.

    The proof covers `label || local_id || ':' || challenge || binding`.
    Replaying a captured hello against the same server process yields only
    the byte-identical ACK the attacker already captured (the HW suffix is
    bound to the replayed challenge); it cannot complete the handshake
    because step 3 requires answering the server's fresh challenge_b.
    """
    msg = (
        HS_LABEL_CLIENT_HELLO
        + local_id.encode("utf-8")
        + b":"
        + challenge
        + channel_binding
    )
    return hmac_mod.new(fleet_key, msg, hashlib.sha256).digest()


def verify_client_hello_proof(
    fleet_key: bytes,
    local_id: str,
    challenge: bytes,
    proof: bytes,
    channel_binding: bytes = b"",
) -> bool:
    """Verify a client hello proof (constant-time)."""
    expected = compute_client_hello_proof(
        fleet_key, local_id, challenge, channel_binding,
    )
    return hmac_mod.compare_digest(expected, proof)


# ------------------------------------------------------------------ #
# Handshake HW exchange (v2.2 PR 4 — Issue 2 + A5 + A7)                #
# ------------------------------------------------------------------ #

# Wire version for the structured handshake ACK/RESP payload that carries
# a signed hardware profile. Bumped on breaking protocol changes. A7 (autoplan):
# include a wire_version byte so future revisions are negotiable.
HW_HANDSHAKE_WIRE_VERSION = 1

# Max serialized HW-json size on the wire. Belt-and-braces limit to prevent
# a malicious peer from flooding the handshake parser with huge payloads
# after auth succeeds but before the full message is validated.
HW_HANDSHAKE_MAX_JSON_BYTES = 8 * 1024  # 8 KB is ~50x any realistic HW profile


class HandshakeHwValidationError(ValueError):
    """Raised when an HW-profile payload from a peer fails HMAC verification
    or structural validation. The connection should be closed."""


def sign_hw_profile(
    fleet_key: bytes,
    wire_version: int,
    peer_challenge: bytes,
    node_id: str,
    hw_json: bytes,
) -> bytes:
    """Sign a hardware-profile payload with the fleet HMAC key.

    The signature covers `wire_version || peer_challenge || node_id || hw_json`
    so the HW payload is bound to THIS session's challenge — replay protection
    for A5 (captured HW profiles from another session cannot be replayed).

    Args:
        fleet_key: Derived fleet HMAC key.
        wire_version: Protocol version byte (currently 1).
        peer_challenge: The 32-byte challenge this peer sent to us. Including
            it in the HMAC input binds the HW payload to this specific auth
            session.
        node_id: The sender's node id (also included to prevent id swap).
        hw_json: The HW profile serialized as JSON bytes.

    Returns:
        32-byte HMAC-SHA256 signature.
    """
    msg = (
        bytes([wire_version & 0xFF])
        + peer_challenge
        + node_id.encode("utf-8")
        + b":"
        + hw_json
    )
    return hmac_mod.new(fleet_key, msg, hashlib.sha256).digest()


def verify_hw_profile(
    fleet_key: bytes,
    wire_version: int,
    peer_challenge: bytes,
    node_id: str,
    hw_json: bytes,
    signature: bytes,
) -> bool:
    """Verify the HMAC of a received HW-profile payload (constant-time).

    Returns True if the signature is valid — meaning the sender knows the
    fleet key and the HW payload is bound to the challenge WE sent them.
    """
    expected = sign_hw_profile(fleet_key, wire_version, peer_challenge, node_id, hw_json)
    return hmac_mod.compare_digest(expected, signature)


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


def sign_heartbeat_with_hw(
    fleet_key: bytes, node_id: str, nonce: bytes, hw_json: bytes,
) -> bytes:
    """Sign a v2.2 heartbeat that carries a hardware-profile payload.

    v2.2 PR 5 (Issue 6): used by `--peer` manual-peer bootstrap to exchange
    real hardware info during the APING/APONG round-trip, so manual peers
    don't register with zero compute_score.

    HMAC covers `node_id || nonce || hw_json`. The nonce binds the HW payload
    to this specific heartbeat — the same replay-protection pattern as
    `sign_hw_profile` in the transport handshake.
    """
    msg = node_id.encode("utf-8") + b":" + nonce + b":" + hw_json
    return hmac_mod.new(fleet_key, msg, hashlib.sha256).digest()


def verify_heartbeat_with_hw(
    fleet_key: bytes,
    node_id: str,
    nonce: bytes,
    hw_json: bytes,
    signature: bytes,
) -> bool:
    """Verify the HMAC of a v2.2 heartbeat-with-HW message (constant-time)."""
    expected = sign_heartbeat_with_hw(fleet_key, node_id, nonce, hw_json)
    return hmac_mod.compare_digest(expected, signature)


def sign_heartbeat_response(
    fleet_key: bytes,
    node_id: str,
    resp_nonce: bytes,
    req_nonce: bytes,
    hw_json: Optional[bytes] = None,
) -> bytes:
    """Sign an APONG heartbeat response, bound to the request's nonce.

    v2.3 security hardening: the old APONG signature covered only the
    responder's own fresh nonce, so a captured APONG from one exchange was
    a valid-looking response to ANY later APING. Binding `req_nonce` (the
    nonce from the APING being answered) pins each response to exactly one
    request — a relay or replay of a stale APONG fails verification.

    HMAC covers `label || node_id || ':' || resp_nonce || ':' || req_nonce
    [|| ':' || hw_json]` with the v3 response label, so a response digest
    can never double as a request digest.
    """
    msg = (
        HB_LABEL_RESPONSE
        + node_id.encode("utf-8")
        + b":"
        + resp_nonce
        + b":"
        + req_nonce
    )
    if hw_json is not None:
        msg += b":" + hw_json
    return hmac_mod.new(fleet_key, msg, hashlib.sha256).digest()


def verify_heartbeat_response(
    fleet_key: bytes,
    node_id: str,
    resp_nonce: bytes,
    req_nonce: bytes,
    signature: bytes,
    hw_json: Optional[bytes] = None,
) -> bool:
    """Verify a request-bound APONG signature (constant-time)."""
    expected = sign_heartbeat_response(
        fleet_key, node_id, resp_nonce, req_nonce, hw_json=hw_json,
    )
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
    # Empty arrays vacuously pass isfinite().all() and then crash .max() with a
    # ValueError (not GradientValidationError) — reject them explicitly so a peer
    # cannot bypass the validator with a zero-length payload.
    if gradients.size == 0:
        raise GradientValidationError("Received empty gradient array — rejecting.")
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


def create_server_tls_context() -> tuple[ssl.SSLContext, bytes]:
    """Create a server TLS context plus its certificate fingerprint.

    Returns (ctx, sha256_of_der_cert). The fingerprint is the channel
    binding mixed into every handshake HMAC (v2.3): both peers commit to
    the SAME TLS channel, so a MITM that terminates TLS on both legs — easy
    against self-signed certs with CERT_NONE — produces mismatched digests
    and the handshake fails.

    Authentication is handled by HMAC challenge-response, not certificates.
    TLS provides encryption; the fingerprint binding upgrades it from
    passive-only protection to active-MITM resistance.

    Cert + private key are generated in-process via the `cryptography` library.
    They are written to user-only temp files ($TMPDIR is user-scoped on macOS)
    just long enough for `SSLContext.load_cert_chain` to consume them (stdlib
    ssl requires file paths), then immediately unlinked — the key stays in
    the SSLContext's memory for the lifetime of the server but never sits
    on disk after this function returns.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    cert_pem, key_pem, cert_der = _generate_cert_bytes()
    certfile, keyfile = _write_ephemeral_pem(cert_pem, key_pem)
    try:
        ctx.load_cert_chain(certfile, keyfile)
    finally:
        for path in (certfile, keyfile):
            try:
                os.unlink(path)
            except OSError:
                pass
    return ctx, hashlib.sha256(cert_der).digest()


def create_server_ssl_context() -> ssl.SSLContext:
    """Create a server-side TLS context with an ephemeral self-signed cert.

    Compatibility wrapper around create_server_tls_context() for callers
    that don't need the certificate fingerprint (e.g. the heartbeat
    server, where channel binding is provided by request-nonce binding).
    """
    ctx, _ = create_server_tls_context()
    return ctx


def tls_channel_binding_from_writer(writer) -> bytes:
    """Extract the channel binding (peer-cert SHA-256) from a TLS stream.

    Returns b"" when the connection is not TLS (open fleets) — both sides
    then mix an empty binding into their HMACs, which stays consistent.
    For a client connected via `asyncio.open_connection(ssl=...)`, the
    peer certificate is the server's ephemeral self-signed cert.
    """
    ssl_obj = writer.get_extra_info("ssl_object")
    if ssl_obj is None:
        return b""
    der = ssl_obj.getpeercert(binary_form=True)
    if not der:
        return b""
    return hashlib.sha256(der).digest()


def create_client_ssl_context() -> ssl.SSLContext:
    """Create a client-side TLS context that accepts self-signed certs.

    No hostname verification — auth is via HMAC, TLS is encryption-only.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _generate_cert_bytes() -> tuple[bytes, bytes, bytes]:
    """Generate an ephemeral self-signed EC (P-256) cert + private key.

    Returns (cert_pem, key_pem, cert_der) — all as bytes, never touches
    disk. cert_der feeds the channel-binding fingerprint. SHA-256
    signature, 25-hour validity (5-min clock-skew leeway on the
    not-before bound so agents behind slightly-off clocks still accept
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
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem, cert_der


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
    cert_pem, key_pem, _ = _generate_cert_bytes()
    return _write_ephemeral_pem(cert_pem, key_pem)
