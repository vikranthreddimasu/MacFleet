"""Tests for the token-bootstrap UX (Issue 26 / PR 13)."""

from __future__ import annotations

import io

import pytest

from macfleet.security.bootstrap import (
    PairingError,
    parse_pairing_url,
    print_pairing_info,
    render_qr_ascii,
    token_to_url,
)


class TestTokenToUrl:
    def test_basic(self):
        url = token_to_url("secret-token-123")
        assert url.startswith("macfleet://pair?token=")
        assert "secret-token-123" in url

    def test_with_fleet_id(self):
        url = token_to_url("tkn", fleet_id="team-a")
        assert "token=tkn" in url
        assert "fleet=team-a" in url

    def test_url_encodes_special_chars(self):
        url = token_to_url("token with spaces!")
        # Special chars must be percent-encoded
        assert " " not in url
        assert "+" not in url  # quote uses %20, not +
        assert "%20" in url

    def test_empty_token_rejected(self):
        with pytest.raises(PairingError):
            token_to_url("")


class TestParsePairingUrl:
    def test_roundtrip(self):
        url = token_to_url("my-secret-token", fleet_id="fleet-x")
        token, fleet_id = parse_pairing_url(url)
        assert token == "my-secret-token"
        assert fleet_id == "fleet-x"

    def test_no_fleet_id(self):
        url = token_to_url("just-a-token")
        token, fleet_id = parse_pairing_url(url)
        assert token == "just-a-token"
        assert fleet_id is None

    def test_roundtrip_with_special_chars(self):
        url = token_to_url("t+o=ken/with:chars")
        token, _ = parse_pairing_url(url)
        assert token == "t+o=ken/with:chars"

    def test_wrong_scheme_rejected(self):
        with pytest.raises(PairingError, match="scheme"):
            parse_pairing_url("http://pair?token=abc")

    def test_missing_token_rejected(self):
        with pytest.raises(PairingError, match="token"):
            parse_pairing_url("macfleet://pair?fleet=x")

    def test_empty_token_rejected(self):
        with pytest.raises(PairingError, match="token"):
            parse_pairing_url("macfleet://pair?token=")

    def test_gibberish_rejected(self):
        with pytest.raises(PairingError):
            parse_pairing_url("not a url at all")


class TestRenderQrAscii:
    def test_non_empty_output(self):
        qr = render_qr_ascii("macfleet://pair?token=abc")
        assert qr  # non-empty
        assert "\n" in qr  # multi-line

    def test_uses_half_blocks(self):
        """Half-block glyphs keep the QR square-ish in terminal fonts."""
        qr = render_qr_ascii("hello")
        # Should contain at least some half-block chars, not all full blocks
        assert "▀" in qr or "▄" in qr or "█" in qr

    def test_deterministic(self):
        """Same input → same QR output."""
        content = "macfleet://pair?token=repro&fleet=test"
        qr1 = render_qr_ascii(content)
        qr2 = render_qr_ascii(content)
        assert qr1 == qr2


class TestPrintPairingInfo:
    def test_default_hides_permanent_token(self):
        rendered = print_pairing_info(
            "secret-token-123",
            fleet_id="test-fleet",
        )

        assert "secret-token-123" not in rendered
        assert "macfleet://pair?" not in rendered
        assert "Permanent fleet token hidden" in rendered

    def test_reveal_token_returns_full_block(self):
        rendered = print_pairing_info(
            "secret-token-123",
            fleet_id="test-fleet",
            reveal_token=True,
        )

        assert "macfleet://pair?" in rendered
        assert "secret-token-123" in rendered
        assert "WARNING" in rendered
        assert "macfleet pair" in rendered

    def test_writes_to_out(self):
        buf = io.StringIO()
        print_pairing_info(
            "tkn12345678",
            reveal_token=True,
            out=buf,
        )
        out = buf.getvalue()
        assert "macfleet://pair?" in out
        assert "tkn12345678" in out

    def test_pasteboard_requires_explicit_reveal(self):
        with pytest.raises(PairingError, match="reveal_token"):
            print_pairing_info(
                "secret-token-123",
                to_pasteboard=True,
            )

    def test_pasteboard_failure_not_fatal_when_revealed(self, monkeypatch):
        """pbcopy failure shouldn't crash the function — URL still printed."""
        from macfleet.security import bootstrap

        def boom(value):
            raise OSError("pbcopy unavailable")

        monkeypatch.setattr(bootstrap, "copy_to_pasteboard", boom)

        # Must not raise
        rendered = print_pairing_info(
            "resilient-token",
            reveal_token=True,
            to_pasteboard=True,
        )
        assert "macfleet://pair?" in rendered
