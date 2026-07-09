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


def test_audit_event_redacts_sensitive_strings_by_value(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(audit, "AUDIT_FILE", str(audit_path))

    audit.audit_event(
        "pairing.failed",
        url="macfleet://pair?token=super-secret-token&fleet=lab",
        message="retry with code=ABCD-EFGH and Authorization: Bearer abc.def",
        nested={"detail": "proof=abcdef123456 status=failed"},
    )

    record = json.loads(audit_path.read_text().strip())
    encoded = json.dumps(record, sort_keys=True)
    assert "super-secret-token" not in encoded
    assert "ABCD-EFGH" not in encoded
    assert "abc.def" not in encoded
    assert "abcdef123456" not in encoded
    assert record["url"] == "macfleet://pair?token=[REDACTED]&fleet=lab"
    assert "Bearer [REDACTED]" in record["message"]
    assert record["nested"]["detail"] == "proof=[REDACTED] status=failed"


def test_audit_file_is_private(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(audit, "AUDIT_FILE", str(audit_path))

    audit.audit_event("auth.failure", peer_ip="127.0.0.1")

    mode = stat.S_IMODE(Path(audit_path).stat().st_mode)
    assert mode & 0o077 == 0


def test_audit_directory_mode_is_repaired(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit-dir"
    audit_dir.mkdir(mode=0o755)
    audit_path = audit_dir / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_DIR", str(audit_dir))
    monkeypatch.setattr(audit, "AUDIT_FILE", str(audit_path))

    audit.audit_event("auth.failure", peer_ip="127.0.0.1")

    mode = stat.S_IMODE(audit_dir.stat().st_mode)
    assert mode == 0o700


def test_audit_file_symlink_is_not_followed(tmp_path, monkeypatch):
    target = tmp_path / "target.log"
    target.write_text("keep-me\n")
    audit_path = tmp_path / "audit.jsonl"
    audit_path.symlink_to(target)
    monkeypatch.setattr(audit, "AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(audit, "AUDIT_FILE", str(audit_path))

    audit.audit_event("auth.failure", peer_ip="127.0.0.1")

    assert target.read_text() == "keep-me\n"


def test_audit_event_with_non_json_field_does_not_interrupt_caller(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(audit, "AUDIT_FILE", str(audit_path))

    audit.audit_event("auth.failure", diagnostic=object())

    assert not audit_path.exists() or audit_path.read_text() == ""


def test_audit_ignores_non_string_event_names(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(audit, "AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(audit, "AUDIT_FILE", str(audit_path))

    audit.audit_event(["not", "an", "event"])

    assert not audit_path.exists()
