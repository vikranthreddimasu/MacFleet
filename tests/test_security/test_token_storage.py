"""Token replacement must preserve the old credential on a failed write."""

import errno
import os
import stat

import pytest

from macfleet.security import auth


@pytest.fixture
def token_path(tmp_path, monkeypatch):
    path = tmp_path / "fleet-token"
    path.write_text("original-fleet-token")
    path.chmod(0o600)
    monkeypatch.setattr(auth, "TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr(auth, "TOKEN_FILE", str(path))
    return path


def test_failed_rotation_preserves_previous_token(token_path, monkeypatch):
    write = os.write

    def disk_full(fd, data):
        write(fd, data[:3])
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(os, "write", disk_full)
    with pytest.raises(OSError, match="disk full"):
        auth.save_fleet_token("replacement-fleet-token")
    assert token_path.read_text() == "original-fleet-token"
    assert list(token_path.parent.iterdir()) == [token_path]


def test_rotation_does_not_modify_other_hardlinks(token_path):
    other = token_path.parent / "other-file"
    os.link(token_path, other)
    auth.save_fleet_token("replacement-fleet-token")
    assert other.read_text() == "original-fleet-token"
    assert token_path.read_text() == "replacement-fleet-token"
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_read_checks_permissions_of_opened_file(token_path, monkeypatch):
    replacement = token_path.parent / "replacement"
    replacement.write_text("exposed-fleet-token")
    replacement.chmod(0o644)
    open_file = os.open

    def replace_before_open(path, flags, *args, **kwargs):
        if path == str(token_path):
            os.replace(replacement, token_path)
        return open_file(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises(PermissionError, match="permissive mode"):
        auth._read_token_file()
