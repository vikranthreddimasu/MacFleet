"""Repository source hygiene checks for package initializers."""

from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _package_initializers() -> list[Path]:
    return sorted((ROOT / "macfleet").rglob("__init__.py"))


def test_package_initializers_compile():
    for path in _package_initializers():
        py_compile.compile(str(path), doraise=True)


def test_package_initializers_do_not_contain_shell_transcripts():
    transcript_markers = (
        "[master ",
        " files changed, ",
        "OK: docs(",
    )

    offenders: list[str] = []
    for path in _package_initializers():
        text = path.read_text()
        if any(marker in text for marker in transcript_markers):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
