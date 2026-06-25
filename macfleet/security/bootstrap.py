"""Legacy token-bearing pairing URL helpers.

The current CLI pairing flow lives in `macfleet.security.enrollment` and
uses short-lived host/code enrollment so the permanent token is not printed.
This module remains only for migration from older `macfleet://pair?token=...`
URLs and for parsing pasteboard/stdin input in `macfleet pair`.
"""

from __future__ import annotations

import subprocess
from typing import Optional, TextIO
from urllib.parse import parse_qs, quote, urlparse

from macfleet.security.auth import MIN_TOKEN_LENGTH


class PairingError(ValueError):
    """Raised when a pairing URL can't be parsed or is invalid."""


def token_to_url(token: str, fleet_id: Optional[str] = None) -> str:
    """Encode a fleet token + id as a pairing URL.

    Format: `macfleet://pair?token=<quoted>&fleet=<quoted>`

    The `macfleet://` scheme is recognized by iOS once the MacFleet app
    is installed (future v2.3 work) and surfaces as a tappable link in
    the camera app's QR preview. For now, users copy the URL into a
    second Mac's `macfleet pair` CLI.
    """
    if not token:
        raise PairingError("token must be non-empty")
    params = [f"token={quote(token, safe='')}"]
    if fleet_id:
        params.append(f"fleet={quote(fleet_id, safe='')}")
    return "macfleet://pair?" + "&".join(params)


def parse_pairing_url(url: str) -> tuple[str, Optional[str]]:
    """Inverse of token_to_url. Returns (token, fleet_id).

    Raises PairingError on any malformed input — bad scheme, missing
    token, empty values. The caller should surface this as a CLI error,
    not a stack trace.
    """
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError) as e:
        raise PairingError(f"unparseable URL: {e}") from e

    if parsed.scheme != "macfleet":
        raise PairingError(
            f"expected scheme 'macfleet://', got {parsed.scheme!r}"
        )
    if parsed.netloc != "pair" and parsed.hostname != "pair":
        # urlparse treats `macfleet://pair?foo=bar` as netloc=pair.
        # Accept both spellings defensively.
        if parsed.path.strip("/") != "pair":
            raise PairingError(
                f"expected 'pair' path, got {parsed.netloc!r} / {parsed.path!r}"
            )

    params = parse_qs(parsed.query, strict_parsing=False)
    token_values = params.get("token") or []
    if not token_values or not token_values[0]:
        raise PairingError("URL missing required 'token' parameter")

    token = token_values[0]
    if len(token) < MIN_TOKEN_LENGTH:
        raise PairingError(
            f"token is too short ({len(token)} chars); minimum is {MIN_TOKEN_LENGTH}"
        )
    fleet_values = params.get("fleet") or []
    fleet_id = fleet_values[0] if fleet_values else None

    return token, fleet_id


def render_qr_ascii(content: str) -> str:
    """Render legacy pairing content as a compact ASCII-art QR code.

    Uses half-block glyphs so the QR is half the height a naive ASCII
    renderer would produce. Fits on a standard 80-col terminal for
    content up to ~250 chars (typical pairing URL is 60-80).

    Not used by the current enrollment flow; retained for compatibility
    with older token-bearing pairing URLs.
    """
    try:
        import qrcode
    except ImportError as e:
        raise ImportError(
            "Token QR bootstrap requires the `qrcode` package. "
            "Install it with `pip install qrcode` (or as part of "
            "`pip install macfleet`)."
        ) from e

    qr = qrcode.QRCode(
        version=None,  # auto-pick size
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    qr.add_data(content)
    qr.make(fit=True)

    matrix = qr.get_matrix()
    lines: list[str] = []
    # Process two rows at a time → half-block chars make output square-ish
    for i in range(0, len(matrix), 2):
        top = matrix[i]
        bot = matrix[i + 1] if i + 1 < len(matrix) else [False] * len(top)
        row_chars: list[str] = []
        for t, b in zip(top, bot):
            if t and b:
                row_chars.append("█")
            elif t and not b:
                row_chars.append("▀")
            elif not t and b:
                row_chars.append("▄")
            else:
                row_chars.append(" ")
        lines.append("".join(row_chars))
    return "\n".join(lines)


def copy_to_pasteboard(value: str) -> None:
    """Write `value` to the macOS pasteboard via `pbcopy`.

    Silently no-ops on platforms without pbcopy (Linux CI, etc.) so
    tests don't break. Callers can check return value of
    `read_from_pasteboard()` to verify the round-trip when needed.

    Raises OSError if pbcopy exists but fails (e.g. permission denied
    writing to another user's session).
    """
    try:
        proc = subprocess.Popen(
            ["pbcopy"], stdin=subprocess.PIPE, close_fds=True,
        )
    except FileNotFoundError:
        return  # not on macOS
    proc.communicate(input=value.encode("utf-8"))
    if proc.returncode != 0:
        raise OSError(f"pbcopy exited with status {proc.returncode}")


def read_from_pasteboard() -> Optional[str]:
    """Read current pasteboard contents, or None if unreadable.

    Used by `macfleet pair` to pull a pairing URL a user copied on
    another Mac (macOS Handoff pasteboard sync makes this work
    seamlessly across devices signed into the same Apple ID).
    """
    try:
        proc = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or None


def print_pairing_info(
    token: str,
    fleet_id: Optional[str] = None,
    *,
    to_pasteboard: bool = True,
    out: Optional[TextIO] = None,
) -> str:
    """Format the legacy token URL + QR pairing block and return it.

    If `to_pasteboard=True` (default), also copies the URL to the local
    pasteboard. This exposes the permanent token and should not be used
    for new flows; prefer `print_enrollment_info`.

    Returns the rendered text so the CLI can print it; accepts `out`
    override for tests.
    """
    url = token_to_url(token, fleet_id=fleet_id)
    qr = render_qr_ascii(url)
    lines = [
        "",
        "Legacy fleet pairing URL (contains the permanent token):",
        f"  {url}",
        "",
        "Prefer `macfleet join --bootstrap` enrollment for new pairing.",
        "This QR/URL is kept for migration from older MacFleet versions:",
        "  echo '<url>' | macfleet pair --stdin",
        "",
        qr,
        "",
    ]
    rendered = "\n".join(lines)

    if to_pasteboard:
        try:
            copy_to_pasteboard(url)
        except OSError:
            pass  # not fatal; URL is already printed

    if out is not None:
        out.write(rendered + "\n")
    return rendered
