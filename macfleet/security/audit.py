"""Local security audit trail for MacFleet.

The audit log is intentionally local-only JSONL. It records security and
reliability events that matter during incident response without shipping
anything off the machine.
"""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import time
from pathlib import Path
from typing import Any

AUDIT_DIR = os.path.expanduser("~/.macfleet")
AUDIT_FILE = os.path.join(AUDIT_DIR, "audit.jsonl")

_SENSITIVE_FIELD_PARTS = (
    "token",
    "secret",
    "password",
    "key",
    "code",
    "proof",
    "sig",
    "signature",
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"\b(token|secret|password|key|code|proof|sig|signature)=([^&\s]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)


def _nofollow_flag() -> int:
    """Return O_NOFOLLOW when the platform supports it."""
    return getattr(os, "O_NOFOLLOW", 0)


def _ensure_private_audit_dir() -> bool:
    """Create or repair the audit directory without following symlinks."""
    try:
        try:
            st = os.lstat(AUDIT_DIR)
        except FileNotFoundError:
            os.makedirs(AUDIT_DIR, mode=0o700, exist_ok=True)
            st = os.lstat(AUDIT_DIR)

        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            return False
        if stat.S_IMODE(st.st_mode) & 0o077:
            os.chmod(AUDIT_DIR, 0o700)
        return True
    except OSError:
        return False


def _redact_string(value: str) -> str:
    """Redact credential-shaped substrings in otherwise safe fields."""
    value = _SENSITIVE_ASSIGNMENT_RE.sub(r"\1=[REDACTED]", value)
    return _BEARER_RE.sub("Bearer [REDACTED]", value)


def _redact_value(key: str, value: Any) -> Any:
    """Redact obvious credential fields before writing audit JSON."""
    lowered = key.lower()
    if any(part in lowered for part in _SENSITIVE_FIELD_PARTS):
        if value in (None, "", False):
            return value
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(key, item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _redact_value(str(key), value) for key, value in fields.items()}


def audit_event(event: str, **fields: Any) -> None:
    """Append a local audit event.

    Audit writes are best-effort: security enforcement must never depend on the
    filesystem being writable. The file is mode 0600 and the directory is 0700.
    """
    if not event:
        return

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "host": socket.gethostname(),
        **_safe_fields(fields),
    }

    try:
        if not _ensure_private_audit_dir():
            return
        path = Path(AUDIT_FILE)
        try:
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                return
        except FileNotFoundError:
            pass

        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | _nofollow_flag(),
            0o600,
        )
        try:
            os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        current = stat.S_IMODE(path.stat().st_mode)
        if current & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        # Deliberately swallow. The primary action should still proceed.
        return


def audit_log_path() -> str:
    """Return the path used for local audit events."""
    return AUDIT_FILE
