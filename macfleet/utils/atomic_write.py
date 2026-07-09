"""Atomic file writes for checkpoint safety.

v2.2 PR 9 (A3 from docs/designs/v3-cathedral.md): training checkpoints
MUST be written atomically so a crash mid-write doesn't leave a
partially-written (and unloadable) `.pt` file on disk. The fix is the
classic temp-file + os.replace pattern:

    1. Write the full payload to `path.tmp` (same directory so we can
       guarantee an atomic rename on the same filesystem)
    2. fsync the temp file so the data hits the platter before rename
    3. os.replace(path.tmp, path) — this is atomic on POSIX + Windows
    4. Optionally fsync the containing directory (extra safety, slow)

Training runs that take hours are fair game for power outages, OOM kills,
and user Ctrl-C. Without this, users lose their last N steps when any of
those happen during the save. With this, they either get the fully-saved
previous checkpoint or the fully-saved new one — never a corrupt mix.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, Union


def _new_temp_path(path: Path) -> Path:
    """Create a private temp path beside ``path`` for an atomic rename."""
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    return Path(temp_name)


def atomic_write_bytes(
    path: Union[str, Path],
    data: bytes,
    fsync_dir: bool = False,
    mode: int = 0o644,
) -> None:
    """Write `data` to `path` atomically.

    Args:
        path: Destination file path.
        data: Bytes to write.
        fsync_dir: If True, fsync the containing directory after the
            rename so the directory entry itself is durable. Adds a
            few ms per save; enable for paranoid deployments.
        mode: POSIX permission bits for the temp file (and therefore
            the destination after rename). Default 0o644. Use 0o600
            for sensitive payloads (tokens, keys, encrypted ckpts).

    Raises:
        OSError: If the write or rename fails. In that case the temp
            file is left behind for post-mortem (caller can inspect it).
    """
    path = Path(path)
    # Same directory guarantees os.replace works without crossing
    # filesystems (rename(2) is atomic only within one fs).
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # A unique sibling prevents simultaneous checkpoint saves from deleting or
    # renaming each other's temporary file.
    tmp = _new_temp_path(path)
    try:
        # Write + fsync the data file
        fd = os.open(tmp, os.O_WRONLY | os.O_TRUNC)
        try:
            os.fchmod(fd, mode)
            written = 0
            while written < len(data):
                bytes_written = os.write(fd, data[written:])
                # POSIX permits a short write. A zero-byte write would leave
                # this loop spinning forever, which is especially harmful on
                # the checkpoint path during a storage failure.
                if bytes_written <= 0:
                    raise OSError("failed to make progress writing temporary file")
                written += bytes_written
            os.fsync(fd)
        finally:
            os.close(fd)

        # Atomic rename — this is the magic. After this call, `path`
        # either points at the full new data or the previous full data.
        os.replace(tmp, path)

        if fsync_dir:
            # Durably record the directory entry itself. Best-effort: the data
            # is already atomically renamed into place, so a dir-fsync failure
            # must NOT surface as a write failure (the except below would also
            # wrongly try to unlink the already-renamed temp).
            try:
                dir_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
    except BaseException:
        # Clean up the temp file on any failure (KeyboardInterrupt,
        # MemoryError, etc.) so a retry doesn't load partial data.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_via(
    path: Union[str, Path],
    writer: Callable[[Union[str, Path]], None],
    fsync_dir: bool = False,
) -> None:
    """Atomic write for libraries that need a path, not bytes.

    Example:
        atomic_write_via(
            "model.pt",
            lambda p: torch.save(state_dict, p),
        )

    The `writer` callable is invoked with a temp path; on success we
    atomically rename it over `path`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _new_temp_path(path)
    try:
        writer(tmp)
        # fsync the file itself for durability. Open O_RDWR (not O_RDONLY):
        # fsync on a read-only fd fails with EBADF/EINVAL on Linux.
        fd = os.open(tmp, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        if fsync_dir:
            # Best-effort (see atomic_write_bytes): data is already renamed in.
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
    except BaseException:
        # User writer can raise anything (PicklingError, RuntimeError,
        # MemoryError, KeyboardInterrupt). Always remove the partial temp
        # so a retry doesn't load it.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
