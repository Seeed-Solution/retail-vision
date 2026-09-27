"""Tests for vision_base.atomicio and the state-file writes that use it
(spec BASE-1 §6.9 rule 4; review item 23).

The point is that a predictable `<target>.tmp` path can be pre-created — or
pre-symlinked — by another local user with write access next to the state
directory, and that following such a link would let them redirect a write.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from vision_base import atomicio  # noqa: E402


def test_write_json_atomic_refuses_a_symlinked_target(tmp_path):
    victim = tmp_path / "victim.json"
    victim.write_text("untouched")
    target = tmp_path / "streams.json"
    target.symlink_to(victim)

    with pytest.raises(OSError):
        atomicio.write_json_atomic(str(target), [{"stream_id": "cam-0"}])
    assert victim.read_text() == "untouched"
    assert target.is_symlink()


def test_write_json_atomic_refuses_a_symlinked_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        atomicio.write_json_atomic(str(link / "streams.json"), [])
    assert list(real.iterdir()) == []


def test_write_json_atomic_writes_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "state" / "streams.json"
    atomicio.write_json_atomic(str(target), [{"stream_id": "cam-0"}])
    assert json.loads(target.read_text()) == [{"stream_id": "cam-0"}]
    assert target.parent.is_dir()
    assert [p.name for p in target.parent.iterdir()] == ["streams.json"]
    if os.name == "posix":
        assert (target.stat().st_mode & 0o777) == 0o600
        assert (target.parent.stat().st_mode & 0o777) == 0o700


def test_write_json_atomic_random_temp_name_is_not_the_old_fixed_path(tmp_path):
    """The previous implementation wrote `<target>.tmp` and called os.replace
    on it: whoever could create that name (or a symlink at it) controlled the
    write."""
    seen = []
    real_mkstemp = atomicio.tempfile.mkstemp

    def spy(*a, **kw):
        fd, name = real_mkstemp(*a, **kw)
        seen.append((name, kw.get("dir")))
        return fd, name

    atomicio.tempfile.mkstemp = spy
    try:
        target = tmp_path / "rt-0.json"
        atomicio.write_json_atomic(str(target), {"a": 1})
    finally:
        atomicio.tempfile.mkstemp = real_mkstemp
    assert seen, "mkstemp must be used (exclusive create, random name)"
    name, directory = seen[0]
    assert name != str(target) + ".tmp"
    assert directory == str(tmp_path)
    assert pathlib.Path(name).name.startswith("rt-0.json.")
