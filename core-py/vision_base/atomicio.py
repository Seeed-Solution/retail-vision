"""Atomic, symlink-safe JSON writes into the service state directory.

Standard library only. Two properties matter (BASE-1 §6.9, review item 23):

- the temporary file is created exclusively with a random name inside the
  target directory, so the predictable ``<target>.tmp`` path cannot be
  pre-created — or pre-symlinked — by another local user;
- neither the target nor its directory may be a symlink, so an attacker who
  can write next to the state directory cannot redirect the write.
"""
from __future__ import annotations

import json
import os
import tempfile

__all__ = ["ensure_private_dir", "write_json_atomic"]


def ensure_private_dir(path: str, *, tighten: bool = False) -> None:
    """Create *path* (and parents) owner-only where the platform supports it.

    ``tighten`` additionally chmods an existing directory; only pass it for a
    directory the service owns.
    """
    os.makedirs(path, mode=0o700, exist_ok=True)
    if tighten:
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


def write_json_atomic(path: str, obj, *, mode: int = 0o600) -> None:
    """Write *obj* as JSON to *path* atomically, refusing symlinked targets."""
    target = os.path.abspath(path)
    if os.path.islink(target):
        raise OSError(f"refusing to write through symlink {path}")
    directory = os.path.dirname(target) or "."
    if os.path.islink(directory):
        raise OSError(f"refusing to write into symlinked directory {directory}")
    ensure_private_dir(directory)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(target) + ".",
                               suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
