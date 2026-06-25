from pathlib import Path

from click.testing import CliRunner

from macfleet.cli.main import cli


def test_rotate_token_creates_private_token(tmp_path: Path, monkeypatch):
    token_path = tmp_path / "fleet-token"
    monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))
    monkeypatch.setattr("macfleet.security.auth.TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr("macfleet.security.audit.AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr("macfleet.security.audit.AUDIT_FILE", str(tmp_path / "audit.jsonl"))

    runner = CliRunner()
    result = runner.invoke(cli, ["rotate-token", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Rotated fleet token" in result.output
    token = token_path.read_text().strip()
    assert len(token) == 64
    assert token not in result.output


def test_rotate_token_can_reveal_when_explicit(tmp_path: Path, monkeypatch):
    token_path = tmp_path / "fleet-token"
    monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))
    monkeypatch.setattr("macfleet.security.auth.TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr("macfleet.security.audit.AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr("macfleet.security.audit.AUDIT_FILE", str(tmp_path / "audit.jsonl"))

    runner = CliRunner()
    result = runner.invoke(cli, ["rotate-token", "--yes", "--show-token"])

    assert result.exit_code == 0, result.output
    token = token_path.read_text().strip()
    assert token in result.output


def test_rotate_token_prompts_before_replacing_existing(tmp_path: Path, monkeypatch):
    token_path = tmp_path / "fleet-token"
    token_path.write_text("old-token-long-enough")
    monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))
    monkeypatch.setattr("macfleet.security.auth.TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr("macfleet.security.audit.AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr("macfleet.security.audit.AUDIT_FILE", str(tmp_path / "audit.jsonl"))

    runner = CliRunner()
    result = runner.invoke(cli, ["rotate-token"], input="n\n")

    assert result.exit_code == 1
    assert token_path.read_text() == "old-token-long-enough"


def test_rotate_token_reports_write_failure_without_traceback(tmp_path: Path, monkeypatch):
    target = tmp_path / "target-token"
    target.write_text("old-token-long-enough")
    token_path = tmp_path / "fleet-token"
    token_path.symlink_to(target)
    monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))
    monkeypatch.setattr("macfleet.security.auth.TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr("macfleet.security.audit.AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr("macfleet.security.audit.AUDIT_FILE", str(tmp_path / "audit.jsonl"))

    runner = CliRunner()
    result = runner.invoke(cli, ["rotate-token", "--yes"])

    assert result.exit_code == 1
    assert "couldn't rotate fleet token" in result.output
    assert "Traceback" not in result.output
    assert target.read_text() == "old-token-long-enough"
