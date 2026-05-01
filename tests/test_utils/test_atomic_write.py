"""Tests for atomic checkpoint writes (A3 from docs/designs/v3-cathedral.md).

The contract: after `atomic_write_bytes(path, data)` returns, `path`
either holds the complete new `data` or the complete previous file's
data. Never a partial write.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from macfleet.utils.atomic_write import atomic_write_bytes, atomic_write_via


class TestAtomicWriteBytes:
    def test_writes_new_file(self, tmp_path: Path):
        target = tmp_path / "checkpoint.pt"
        atomic_write_bytes(target, b"hello world")
        assert target.read_bytes() == b"hello world"

    def test_overwrites_existing(self, tmp_path: Path):
        target = tmp_path / "checkpoint.pt"
        target.write_bytes(b"OLD CONTENTS")
        atomic_write_bytes(target, b"NEW CONTENTS")
        assert target.read_bytes() == b"NEW CONTENTS"

    def test_leaves_old_data_intact_on_failure(self, tmp_path: Path, monkeypatch):
        """If os.replace fails, the original file must still be usable."""
        target = tmp_path / "checkpoint.pt"
        target.write_bytes(b"ORIGINAL")

        real_replace = os.replace

        def flaky_replace(src, dst):
            if str(dst).endswith("checkpoint.pt"):
                raise OSError("simulated rename failure")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", flaky_replace)

        with pytest.raises(OSError, match="simulated rename failure"):
            atomic_write_bytes(target, b"NEW")

        # Original content must still be on disk
        assert target.read_bytes() == b"ORIGINAL"

    def test_temp_file_cleaned_on_failure(self, tmp_path: Path, monkeypatch):
        target = tmp_path / "checkpoint.pt"

        def flaky_replace(src, dst):
            raise OSError("simulated rename failure")

        monkeypatch.setattr(os, "replace", flaky_replace)

        with pytest.raises(OSError):
            atomic_write_bytes(target, b"NEW")

        # .tmp file should have been removed
        assert not (tmp_path / "checkpoint.pt.tmp").exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        target = tmp_path / "runs" / "exp-42" / "checkpoint.pt"
        atomic_write_bytes(target, b"payload")
        assert target.read_bytes() == b"payload"

    def test_empty_bytes(self, tmp_path: Path):
        target = tmp_path / "empty.bin"
        atomic_write_bytes(target, b"")
        assert target.read_bytes() == b""

    def test_large_payload(self, tmp_path: Path):
        target = tmp_path / "large.bin"
        payload = b"\x42" * (8 * 1024 * 1024)  # 8 MB
        atomic_write_bytes(target, payload)
        assert target.read_bytes() == payload

    def test_path_accepts_str(self, tmp_path: Path):
        target = tmp_path / "from_str.bin"
        atomic_write_bytes(str(target), b"ok")
        assert target.read_bytes() == b"ok"

    def test_fsync_dir_option(self, tmp_path: Path):
        target = tmp_path / "durable.bin"
        atomic_write_bytes(target, b"durable", fsync_dir=True)
        assert target.read_bytes() == b"durable"


class TestAtomicWriteVia:
    def test_writer_callback(self, tmp_path: Path):
        target = tmp_path / "model.pt"

        def torch_like_save(p):
            Path(p).write_bytes(b"<<fake torch state_dict>>")

        atomic_write_via(target, torch_like_save)
        assert target.read_bytes() == b"<<fake torch state_dict>>"

    def test_writer_failure_leaves_no_temp(self, tmp_path: Path):
        target = tmp_path / "model.pt"

        def bad_writer(p):
            raise RuntimeError("writer exploded")

        with pytest.raises(RuntimeError, match="writer exploded"):
            atomic_write_via(target, bad_writer)

        # temp shouldn't stick around
        assert not (tmp_path / "model.pt.tmp").exists()
        assert not target.exists()

    def test_writer_preserves_old_on_failure(self, tmp_path: Path):
        target = tmp_path / "model.pt"
        target.write_bytes(b"KEEP ME")

        def bad_writer(p):
            Path(p).write_bytes(b"partial")
            raise RuntimeError("after write, before rename")

        with pytest.raises(RuntimeError):
            atomic_write_via(target, bad_writer)

        # old file intact
        assert target.read_bytes() == b"KEEP ME"
        # partial temp file must not be left behind
        assert not (tmp_path / "model.pt.tmp").exists()
