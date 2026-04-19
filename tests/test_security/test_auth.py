"""Tests for MacFleet security auth primitives."""

from __future__ import annotations

import os
import ssl

import numpy as np
import pytest

from macfleet.security.auth import (
    CHALLENGE_SIZE,
    DEFAULT_SERVICE_TYPE,
    GRADIENT_MAX_MAGNITUDE,
    GRADIENT_MAX_NUMEL,
    MIN_TOKEN_LENGTH,
    TOKEN_ENV_VAR,
    AuthRateLimiter,
    GradientValidationError,
    SecurityConfig,
    compute_response,
    create_client_ssl_context,
    create_server_ssl_context,
    generate_challenge,
    generate_fleet_token,
    resolve_token,
    resolve_token_with_file,
    sign_heartbeat,
    validate_gradient_metadata,
    validate_gradients,
    verify_heartbeat,
    verify_response,
)

# ------------------------------------------------------------------ #
# SecurityConfig                                                      #
# ------------------------------------------------------------------ #


class TestSecurityConfig:
    def test_no_token_is_not_secure(self):
        cfg = SecurityConfig()
        assert not cfg.is_secure
        assert cfg.fleet_key is None

    def test_token_is_secure(self):
        cfg = SecurityConfig(token="my-secret-token")
        assert cfg.is_secure
        assert cfg.fleet_key is not None
        assert len(cfg.fleet_key) == 32  # SHA-256 digest

    def test_fleet_key_deterministic(self):
        cfg1 = SecurityConfig(token="my-secret-token")
        cfg2 = SecurityConfig(token="my-secret-token")
        assert cfg1.fleet_key == cfg2.fleet_key

    def test_different_tokens_different_keys(self):
        cfg1 = SecurityConfig(token="secret-a-long")
        cfg2 = SecurityConfig(token="secret-b-long")
        assert cfg1.fleet_key != cfg2.fleet_key

    def test_different_fleet_ids_different_keys(self):
        cfg1 = SecurityConfig(token="same-long-token", fleet_id="fleet-a")
        cfg2 = SecurityConfig(token="same-long-token", fleet_id="fleet-b")
        assert cfg1.fleet_key != cfg2.fleet_key

    def test_default_fleet_id(self):
        cfg1 = SecurityConfig(token="same-long-token")
        cfg2 = SecurityConfig(token="same-long-token", fleet_id="default")
        assert cfg1.fleet_key == cfg2.fleet_key

    def test_mdns_service_type_no_token(self):
        cfg = SecurityConfig()
        assert cfg.mdns_service_type == DEFAULT_SERVICE_TYPE

    def test_mdns_service_type_with_token(self):
        cfg = SecurityConfig(token="my-secret-token")
        stype = cfg.mdns_service_type
        assert stype != DEFAULT_SERVICE_TYPE
        assert stype.startswith("_mf-")
        assert stype.endswith("._tcp.local.")
        assert len(stype.split(".")[0]) == len("_mf-") + 8  # 8-char hex hash

    def test_mdns_service_type_deterministic(self):
        cfg1 = SecurityConfig(token="same-token-long")
        cfg2 = SecurityConfig(token="same-token-long")
        assert cfg1.mdns_service_type == cfg2.mdns_service_type

    def test_mdns_service_type_different_tokens(self):
        cfg1 = SecurityConfig(token="token-a-long")
        cfg2 = SecurityConfig(token="token-b-long")
        assert cfg1.mdns_service_type != cfg2.mdns_service_type

    def test_mdns_service_type_different_fleet_ids(self):
        cfg1 = SecurityConfig(token="same-long-token", fleet_id="fleet-a")
        cfg2 = SecurityConfig(token="same-long-token", fleet_id="fleet-b")
        assert cfg1.mdns_service_type != cfg2.mdns_service_type

    def test_tls_forced_when_token_set(self):
        """TLS is mandatory when auth is enabled — prevents unencrypted auth."""
        cfg = SecurityConfig(token="secret-token")
        assert cfg.tls is True

    def test_tls_forced_even_if_explicitly_false(self):
        """Cannot disable TLS when token is set."""
        cfg = SecurityConfig(token="secret-token", tls=False)
        assert cfg.tls is True

    def test_tls_not_forced_when_open(self):
        cfg = SecurityConfig()
        assert cfg.tls is False

    def test_raw_token_not_stored(self):
        """SecurityConfig should not retain the raw token string."""
        cfg = SecurityConfig(token="super-secret-token")
        assert not hasattr(cfg, "token") or cfg.__dict__.get("token") is None
        # The fleet_key should be derived but the raw token gone
        assert cfg.fleet_key is not None

    def test_short_token_rejected(self):
        """Tokens shorter than MIN_TOKEN_LENGTH are rejected."""
        with pytest.raises(ValueError, match="at least"):
            SecurityConfig(token="short")

    def test_empty_token_rejected(self):
        """Empty string token is rejected."""
        with pytest.raises(ValueError, match="at least"):
            SecurityConfig(token="")


# ------------------------------------------------------------------ #
# Token Resolution (env var support)                                  #
# ------------------------------------------------------------------ #


class TestResolveToken:
    def test_explicit_token_wins(self):
        assert resolve_token("explicit") == "explicit"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, "from-env")
        assert resolve_token(None) == "from-env"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, "from-env")
        assert resolve_token("explicit") == "explicit"

    def test_none_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
        assert resolve_token(None) is None


# ------------------------------------------------------------------ #
# Challenge-Response                                                  #
# ------------------------------------------------------------------ #


class TestChallengeResponse:
    def test_generate_challenge_length(self):
        c = generate_challenge()
        assert len(c) == CHALLENGE_SIZE

    def test_generate_challenge_random(self):
        c1 = generate_challenge()
        c2 = generate_challenge()
        assert c1 != c2

    def test_compute_response_deterministic(self):
        key = b"test-key-32-bytes-exactly-padded"
        challenge = b"A" * CHALLENGE_SIZE
        r1 = compute_response(key, challenge)
        r2 = compute_response(key, challenge)
        assert r1 == r2

    def test_compute_response_length(self):
        key = b"key"
        challenge = generate_challenge()
        r = compute_response(key, challenge)
        assert len(r) == 32  # SHA-256

    def test_verify_response_correct(self):
        key = b"fleet-key"
        challenge = generate_challenge()
        response = compute_response(key, challenge)
        assert verify_response(key, challenge, response) is True

    def test_verify_response_wrong_key(self):
        key_a = b"key-a"
        key_b = b"key-b"
        challenge = generate_challenge()
        response = compute_response(key_a, challenge)
        assert verify_response(key_b, challenge, response) is False

    def test_verify_response_wrong_challenge(self):
        key = b"key"
        c1 = generate_challenge()
        c2 = generate_challenge()
        response = compute_response(key, c1)
        assert verify_response(key, c2, response) is False

    def test_verify_response_tampered(self):
        key = b"key"
        challenge = generate_challenge()
        response = compute_response(key, challenge)
        tampered = bytearray(response)
        tampered[0] ^= 0xFF
        assert verify_response(key, challenge, bytes(tampered)) is False

    def test_mutual_auth_flow(self):
        """Simulate the full 3-message mutual authentication."""
        fleet_key = SecurityConfig(token="shared-secret-token").fleet_key

        challenge_a = generate_challenge()
        response_a = compute_response(fleet_key, challenge_a)
        challenge_b = generate_challenge()

        assert verify_response(fleet_key, challenge_a, response_a)
        response_b = compute_response(fleet_key, challenge_b)
        assert verify_response(fleet_key, challenge_b, response_b)

    def test_mutual_auth_mismatched_tokens(self):
        key_a = SecurityConfig(token="secret-a-long").fleet_key
        key_b = SecurityConfig(token="secret-b-long").fleet_key

        challenge_a = generate_challenge()
        response_a = compute_response(key_b, challenge_a)
        assert not verify_response(key_a, challenge_a, response_a)


# ------------------------------------------------------------------ #
# Heartbeat Authentication                                            #
# ------------------------------------------------------------------ #


class TestHeartbeatAuth:
    def test_sign_verify_roundtrip(self):
        key = b"fleet-key"
        nonce = b"random-nonce-16b"
        sig = sign_heartbeat(key, "node-0", nonce)
        assert verify_heartbeat(key, "node-0", nonce, sig)

    def test_wrong_key_rejected(self):
        key_a = b"key-a"
        key_b = b"key-b"
        nonce = b"nonce"
        sig = sign_heartbeat(key_a, "node-0", nonce)
        assert not verify_heartbeat(key_b, "node-0", nonce, sig)

    def test_wrong_node_id_rejected(self):
        key = b"key"
        nonce = b"nonce"
        sig = sign_heartbeat(key, "node-0", nonce)
        assert not verify_heartbeat(key, "node-1", nonce, sig)

    def test_wrong_nonce_rejected(self):
        key = b"key"
        sig = sign_heartbeat(key, "node-0", b"nonce-a")
        assert not verify_heartbeat(key, "node-0", b"nonce-b", sig)

    def test_signature_length(self):
        key = b"key"
        sig = sign_heartbeat(key, "node-0", b"nonce")
        assert len(sig) == 32

    def test_tampered_signature_rejected(self):
        key = b"key"
        nonce = b"nonce"
        sig = sign_heartbeat(key, "node-0", nonce)
        tampered = bytearray(sig)
        tampered[-1] ^= 0xFF
        assert not verify_heartbeat(key, "node-0", nonce, bytes(tampered))


# ------------------------------------------------------------------ #
# Gradient Validation (anti-poisoning)                                #
# ------------------------------------------------------------------ #


class TestGradientValidation:
    def test_valid_gradients_pass(self):
        grads = np.random.randn(1000).astype(np.float32)
        validate_gradients(grads)  # should not raise

    def test_nan_gradients_rejected(self):
        grads = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        with pytest.raises(GradientValidationError, match="NaN"):
            validate_gradients(grads)

    def test_inf_gradients_rejected(self):
        grads = np.array([1.0, float("inf"), 3.0], dtype=np.float32)
        with pytest.raises(GradientValidationError, match="Inf"):
            validate_gradients(grads)

    def test_neg_inf_gradients_rejected(self):
        grads = np.array([1.0, float("-inf"), 3.0], dtype=np.float32)
        with pytest.raises(GradientValidationError, match="Inf"):
            validate_gradients(grads)

    def test_extreme_magnitude_rejected(self):
        grads = np.array([GRADIENT_MAX_MAGNITUDE + 1], dtype=np.float32)
        with pytest.raises(GradientValidationError, match="magnitude"):
            validate_gradients(grads)

    def test_borderline_magnitude_passes(self):
        grads = np.array([GRADIENT_MAX_MAGNITUDE - 1], dtype=np.float32)
        validate_gradients(grads)  # should not raise

    def test_all_zeros_passes(self):
        grads = np.zeros(100, dtype=np.float32)
        validate_gradients(grads)  # should not raise

    def test_custom_max_magnitude(self):
        grads = np.array([50.0], dtype=np.float32)
        with pytest.raises(GradientValidationError):
            validate_gradients(grads, max_magnitude=10.0)


class TestGradientMetadataValidation:
    def test_valid_metadata_passes(self):
        validate_gradient_metadata(1000, 100)  # should not raise

    def test_numel_too_large_rejected(self):
        with pytest.raises(GradientValidationError, match="original_numel"):
            validate_gradient_metadata(GRADIENT_MAX_NUMEL + 1, 100)

    def test_topk_larger_than_numel_rejected(self):
        with pytest.raises(GradientValidationError, match="topk_count"):
            validate_gradient_metadata(100, 200)

    def test_negative_numel_rejected(self):
        with pytest.raises(GradientValidationError):
            validate_gradient_metadata(-1, 0)

    def test_negative_topk_rejected(self):
        with pytest.raises(GradientValidationError):
            validate_gradient_metadata(100, -1)

    def test_zero_values_pass(self):
        validate_gradient_metadata(0, 0)  # edge case, should pass


# ------------------------------------------------------------------ #
# Rate Limiter                                                        #
# ------------------------------------------------------------------ #


class TestAuthRateLimiter:
    def test_no_failures_not_banned(self):
        rl = AuthRateLimiter()
        assert not rl.is_banned("1.2.3.4")

    def test_few_failures_not_banned(self):
        rl = AuthRateLimiter()
        for _ in range(4):
            rl.record_failure("1.2.3.4")
        assert not rl.is_banned("1.2.3.4")

    def test_max_failures_banned(self):
        rl = AuthRateLimiter()
        for _ in range(5):
            rl.record_failure("1.2.3.4")
        assert rl.is_banned("1.2.3.4")

    def test_success_clears_failures(self):
        rl = AuthRateLimiter()
        for _ in range(4):
            rl.record_failure("1.2.3.4")
        rl.record_success("1.2.3.4")
        assert not rl.is_banned("1.2.3.4")
        assert rl.get_delay("1.2.3.4") == 0.0

    def test_different_ips_independent(self):
        rl = AuthRateLimiter()
        for _ in range(5):
            rl.record_failure("1.1.1.1")
        assert rl.is_banned("1.1.1.1")
        assert not rl.is_banned("2.2.2.2")

    def test_delay_increases_exponentially(self):
        rl = AuthRateLimiter()
        rl.record_failure("1.2.3.4")
        d1 = rl.get_delay("1.2.3.4")
        rl.record_failure("1.2.3.4")
        d2 = rl.get_delay("1.2.3.4")
        assert d2 > d1

    def test_no_delay_without_failures(self):
        rl = AuthRateLimiter()
        assert rl.get_delay("1.2.3.4") == 0.0


# ------------------------------------------------------------------ #
# TLS Context                                                        #
# ------------------------------------------------------------------ #


class TestTLS:
    def test_client_ssl_context(self):
        ctx = create_client_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE

    def test_server_ssl_context(self):
        ctx = create_server_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)

    def test_server_and_client_compatible(self):
        server_ctx = create_server_ssl_context()
        client_ctx = create_client_ssl_context()
        assert server_ctx is not None
        assert client_ctx is not None

    def test_server_ssl_minimum_tls_version(self):
        ctx = create_server_ssl_context()
        assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2

    def test_client_ssl_minimum_tls_version(self):
        ctx = create_client_ssl_context()
        assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


class TestCertGeneration:
    """v2.2 PR 3 — cert generation via cryptography library, no openssl subprocess."""

    def test_generate_cert_bytes_produces_valid_pem(self):
        """_generate_cert_bytes returns parseable PEM cert + PKCS8 key."""
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        from macfleet.security.auth import _generate_cert_bytes

        cert_pem, key_pem = _generate_cert_bytes()
        assert cert_pem.startswith(b"-----BEGIN CERTIFICATE-----")
        assert key_pem.startswith(b"-----BEGIN PRIVATE KEY-----")

        # Cert parses and has expected CN
        cert = x509.load_pem_x509_certificate(cert_pem)
        cn = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
        assert cn[0].value == "macfleet-node"

        # Key parses as EC
        key = serialization.load_pem_private_key(key_pem, password=None)
        # EC keys have a .curve attribute; RSA has .key_size but no .curve
        assert hasattr(key, "curve"), "expected EC key, got something else"

    def test_generate_cert_bytes_unique_per_call(self):
        """Every call produces a fresh key pair (no key reuse across instances)."""
        from macfleet.security.auth import _generate_cert_bytes

        _, key_a = _generate_cert_bytes()
        _, key_b = _generate_cert_bytes()
        assert key_a != key_b

    def test_server_ssl_context_cleans_up_tempfiles(self, tmp_path, monkeypatch):
        """After create_server_ssl_context returns, no PEM tempfiles linger.

        Monkeypatches tempfile.mkstemp to return paths under tmp_path, then
        verifies both files are unlinked by the time the context is returned.
        """
        import tempfile as _tf

        from macfleet.security import auth as auth_mod

        created_paths = []
        real_mkstemp = _tf.mkstemp

        def tracking_mkstemp(suffix=None, prefix=None, dir=None):
            fd, path = real_mkstemp(suffix=suffix, prefix=prefix, dir=str(tmp_path))
            created_paths.append(path)
            return fd, path

        monkeypatch.setattr(auth_mod.tempfile, "mkstemp", tracking_mkstemp)

        ctx = auth_mod.create_server_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert len(created_paths) == 2, "expected 2 tempfiles (cert + key)"
        for p in created_paths:
            assert not os.path.exists(p), f"tempfile {p} should have been unlinked"

    def test_no_openssl_dependency(self, monkeypatch):
        """Cert generation does not shell out to openssl anymore."""
        import subprocess as _sp

        from macfleet.security.auth import create_server_ssl_context

        called = []

        def should_not_run(*args, **kwargs):
            called.append(args)
            raise AssertionError("subprocess.run called — openssl dependency not removed")

        monkeypatch.setattr(_sp, "run", should_not_run)
        ctx = create_server_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert called == [], "subprocess.run should never be invoked"


# ------------------------------------------------------------------ #
# Rate limiter eviction                                                #
# ------------------------------------------------------------------ #


class TestRateLimiterEviction:
    def test_eviction_at_max_entries(self):
        rl = AuthRateLimiter(max_entries=5)
        for i in range(10):
            rl.record_failure(f"10.0.0.{i}")
        # Should have at most 5 entries
        assert len(rl._failures) <= 5

    def test_oldest_entry_evicted(self):
        rl = AuthRateLimiter(max_entries=3)
        rl.record_failure("10.0.0.1")
        rl.record_failure("10.0.0.2")
        rl.record_failure("10.0.0.3")
        # Adding a 4th should evict 10.0.0.1 (oldest)
        rl.record_failure("10.0.0.4")
        assert "10.0.0.1" not in rl._failures
        assert "10.0.0.4" in rl._failures


# ------------------------------------------------------------------ #
# Gradient validation fallback (integration-style)                     #
# ------------------------------------------------------------------ #


class TestGradientFallback:
    def test_nan_triggers_fallback(self):
        """validate_gradients raises on NaN, allowing caller to fall back."""
        clean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        validate_gradients(clean)  # should not raise

        poisoned = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        with pytest.raises(GradientValidationError):
            validate_gradients(poisoned)

    def test_inf_triggers_fallback(self):
        poisoned = np.array([1.0, float("inf"), 3.0], dtype=np.float32)
        with pytest.raises(GradientValidationError):
            validate_gradients(poisoned)

    def test_magnitude_triggers_fallback(self):
        poisoned = np.array([GRADIENT_MAX_MAGNITUDE + 1], dtype=np.float32)
        with pytest.raises(GradientValidationError):
            validate_gradients(poisoned)

    def test_clean_gradients_pass(self):
        """Normal gradients pass validation without exception."""
        grads = np.random.randn(10000).astype(np.float32)
        validate_gradients(grads)  # should not raise


# ------------------------------------------------------------------ #
# Token file management                                               #
# ------------------------------------------------------------------ #


class TestTokenFileManagement:
    """Tests for auto-generated fleet tokens and file persistence."""

    def test_generate_fleet_token_length(self):
        token = generate_fleet_token()
        assert len(token) >= MIN_TOKEN_LENGTH

    def test_generate_fleet_token_unique(self):
        t1 = generate_fleet_token()
        t2 = generate_fleet_token()
        assert t1 != t2

    def test_resolve_token_explicit_takes_priority(self):
        assert resolve_token("explicit") == "explicit"

    def test_resolve_token_none_when_no_env(self, monkeypatch):
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
        assert resolve_token() is None

    def test_resolve_token_env_var(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, "from-env")
        assert resolve_token() == "from-env"

    def test_resolve_token_with_file_reads_file(self, tmp_path, monkeypatch):
        """resolve_token_with_file reads from token file."""
        token_file = tmp_path / "fleet-token"
        token_file.write_text("saved-token-value")
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_file))
        assert resolve_token_with_file() == "saved-token-value"

    def test_resolve_token_with_file_auto_generate(self, tmp_path, monkeypatch):
        """resolve_token_with_file auto-generates and saves when no token exists."""
        token_file = tmp_path / "fleet-token"
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_file))
        monkeypatch.setattr("macfleet.security.auth.TOKEN_DIR", str(tmp_path))

        result = resolve_token_with_file(auto_generate=True)
        assert result is not None
        assert len(result) >= MIN_TOKEN_LENGTH
        assert token_file.read_text() == result

    def test_resolve_token_with_file_no_auto_generate(self, tmp_path, monkeypatch):
        """Without auto_generate, returns None when no token found."""
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(tmp_path / "nonexistent"))
        assert resolve_token_with_file() is None

    def test_resolve_token_with_file_explicit_overrides_file(self, tmp_path, monkeypatch):
        """Explicit token overrides saved file."""
        token_file = tmp_path / "fleet-token"
        token_file.write_text("saved-value")
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_file))
        assert resolve_token_with_file("explicit-value") == "explicit-value"

    def test_resolve_token_does_not_read_file(self, tmp_path, monkeypatch):
        """resolve_token (non-file version) ignores token file."""
        token_file = tmp_path / "fleet-token"
        token_file.write_text("saved-value")
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_file))
        assert resolve_token() is None


class TestTokenFilePermissions:
    """v2.2 PR 3 / A6 — token file must be 0600 on write, warn on broader mode."""

    def test_write_creates_file_with_mode_0600(self, tmp_path, monkeypatch):
        """_write_token_file produces a file with exactly mode 0o600."""
        import stat as _stat

        from macfleet.security import auth as auth_mod

        token_file = tmp_path / "fleet-token"
        monkeypatch.setattr(auth_mod, "TOKEN_FILE", str(token_file))
        monkeypatch.setattr(auth_mod, "TOKEN_DIR", str(tmp_path))

        auth_mod._write_token_file("some-token-value-long-enough-to-pass")

        assert token_file.exists()
        mode = _stat.S_IMODE(token_file.stat().st_mode)
        assert mode == 0o600, f"expected mode 0o600, got {oct(mode)}"

    def test_write_repairs_permissive_mode(self, tmp_path, monkeypatch):
        """If token file pre-exists with broader mode, write tightens it to 0600."""
        import stat as _stat

        from macfleet.security import auth as auth_mod

        token_file = tmp_path / "fleet-token"
        monkeypatch.setattr(auth_mod, "TOKEN_FILE", str(token_file))
        monkeypatch.setattr(auth_mod, "TOKEN_DIR", str(tmp_path))

        # Seed with a mode 0644 file — simulates a previously mis-permissioned token
        token_file.write_text("old-value")
        os.chmod(token_file, 0o644)
        assert _stat.S_IMODE(token_file.stat().st_mode) == 0o644

        auth_mod._write_token_file("new-token-value-long-enough-to-pass")

        mode = _stat.S_IMODE(token_file.stat().st_mode)
        assert mode == 0o600, f"expected 0o600 after write, got {oct(mode)}"

    def test_read_warns_on_permissive_mode(self, tmp_path, monkeypatch, caplog):
        """_read_token_file emits a warning when mode has group/other bits set."""
        import logging as _logging

        from macfleet.security import auth as auth_mod

        token_file = tmp_path / "fleet-token"
        token_file.write_text("some-fleet-token-value")
        os.chmod(token_file, 0o644)
        monkeypatch.setattr(auth_mod, "TOKEN_FILE", str(token_file))

        with caplog.at_level(_logging.WARNING, logger="macfleet.security.auth"):
            token = auth_mod._read_token_file()

        assert token == "some-fleet-token-value"  # still returns the token
        assert any("permissive mode" in rec.message for rec in caplog.records), (
            "expected permissive-mode warning in log"
        )

    def test_read_no_warning_when_mode_is_0600(self, tmp_path, monkeypatch, caplog):
        """Proper mode 0600 produces no warning."""
        import logging as _logging

        from macfleet.security import auth as auth_mod

        token_file = tmp_path / "fleet-token"
        token_file.write_text("token-value")
        os.chmod(token_file, 0o600)
        monkeypatch.setattr(auth_mod, "TOKEN_FILE", str(token_file))

        with caplog.at_level(_logging.WARNING, logger="macfleet.security.auth"):
            token = auth_mod._read_token_file()

        assert token == "token-value"
        perm_warnings = [r for r in caplog.records if "permissive mode" in r.message]
        assert not perm_warnings, f"unexpected warnings: {perm_warnings}"

    def test_read_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        """Missing file returns None without crashing or warning."""
        from macfleet.security import auth as auth_mod

        monkeypatch.setattr(auth_mod, "TOKEN_FILE", str(tmp_path / "does-not-exist"))
        assert auth_mod._read_token_file() is None
