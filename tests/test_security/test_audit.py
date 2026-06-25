import json
import stat
from pathlib import Path

from macfleet.security import audit


def test_audit_event_redacts_sensitive_fields(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(audit, "AUDIT_FILE", str(audit_path))

    audit.audit_event(
        "pairing.completed",
        token="super-secret-token",
        enrollment_code="ABCD-EFGH",
        peer="macbook",
        nested={"api_key": "abc123", "safe": "ok"},
    )

    record = json.loads(audit_path.read_text().strip())
    assert record["event"] == "pairing.completed"
    assert record["token"] == "[REDACTED]"
    assert record["enrollment_code"] == "[REDACTED]"
    assert record["nested"]["api_key"] == "[REDACTED]"
    assert record["nested"]["safe"] == "ok"
    assert record["peer"] == "macbook"


def test_audit_file_is_private(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(audit, "AUDIT_FILE", str(audit_path))

    audit.audit_event("auth.failure", peer_ip="127.0.0.1")

    mode = stat.S_IMODE(Path(audit_path).stat().st_mode)
    assert mode & 0o077 == 0
