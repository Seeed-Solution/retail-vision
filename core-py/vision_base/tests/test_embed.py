"""Tests for vision_base.embed.Runtime (spec BASE-1 §8 M1.21, §6.13.1).

Uses tests/fake_runtime.py as the child; the ``native``-marked test runs the
real vb-runtime + synthetic backend via VB_RUNTIME_BIN.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pytest

import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from vision_base.embed import Runtime, RuntimeGone
from vision_base.types import Event, FrameResult

FAKE = os.path.join(os.path.dirname(__file__), "fake_runtime.py")
FAKE_ARGV = [sys.executable, FAKE, "--ipc-fd", "{fd}"]


def make_rt(tmp_path, **kw):
    """from_config + fake runtime argv (no real vb-runtime on the host)."""
    rt = Runtime.from_config(write_config(tmp_path), **kw)
    rt._runtime_argv = list(FAKE_ARGV)
    return rt


def fake_env(**kw):
    env = {"FAKE_FPS": "20"}
    env.update(kw)
    return env


def set_env(env):
    saved = {}
    for k, v in env.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    return saved


def restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def write_config(tmp_path, *, extra=None, native_binary=None):
    cfg = {
        "schema": "vb.config/1", "device_id": "embed-dev-01",
        "backend": {"name": "synthetic", "model_path": "/models/m.onnx"},
    }
    if native_binary:
        cfg["native"] = {"binary": native_binary}
    cfg.update(extra or {})
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def wait_for(predicate, timeout_s=8.0, what="condition"):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


def collect(rt, n, timeout_s=8.0):
    out = []
    deadline = time.time() + timeout_s
    for item in rt.results(timeout_s=0.5):
        out.append(item)
        if len(out) >= n:
            return out
        if time.time() > deadline:
            break
    return out


# ------------------------------------------------------------------- factory


def test_from_config_partial_and_binary_precedence(tmp_path, monkeypatch):
    """from_config succeeds without mqtt/app; binary precedence in 4 cases."""
    monkeypatch.delenv("VB_RUNTIME_BIN", raising=False)
    # 4) default when neither arg, env nor config native.binary
    rt = Runtime.from_config(write_config(tmp_path))
    assert rt._runtime_argv[0] == "/opt/vb/bin/vb-runtime"
    # 3) config native.binary wins over default
    rt = Runtime.from_config(
        write_config(tmp_path, native_binary="/opt/custom/vb-runtime"))
    assert rt._runtime_argv[0] == "/opt/custom/vb-runtime"
    # 2) env VB_RUNTIME_BIN wins over config
    monkeypatch.setenv("VB_RUNTIME_BIN", "/env/vb-runtime")
    rt = Runtime.from_config(
        write_config(tmp_path, native_binary="/opt/custom/vb-runtime"))
    assert rt._runtime_argv[0] == "/env/vb-runtime"
    # 1) explicit binary argument wins over env
    rt = Runtime.from_config(
        write_config(tmp_path, native_binary="/opt/custom/vb-runtime"),
        binary="/arg/vb-runtime")
    assert rt._runtime_argv[0] == "/arg/vb-runtime"
    assert os.path.exists(rt._cfg_path)
    rt.stop()
    assert not os.path.exists(rt._cfg_path)


# -------------------------------------------------------------------- runtime


def test_results_alternate_frames_and_events(tmp_path):
    saved = set_env(fake_env(FAKE_FPS="15"))
    try:
        with make_rt(tmp_path, queue_size=64) as rt:
            rt.add_stream("cam-0", "fake://0")
            rt.add_stream("cam-1", "fake://1")
            items = collect(rt, 8, timeout_s=6)
            frames = [i for i in items if isinstance(i, FrameResult)]
            events = [i for i in items if isinstance(i, Event)]
            assert len(frames) >= 2 and len(events) >= 2
            # each stream's first event follows its first frame (fake sends
            # the tick right after the stream's first frame)
            first_frame = {}
            for i, item in enumerate(items):
                if isinstance(item, FrameResult):
                    first_frame.setdefault(item.stream_id, i)
            for item in events:
                if item.stream_id in first_frame:
                    assert items.index(item) > first_frame[item.stream_id]
    finally:
        restore_env(saved)


def test_frame_queue_drops_oldest_events_never_dropped(tmp_path):
    saved = set_env(fake_env(FAKE_FPS="200"))
    try:
        with make_rt(tmp_path, queue_size=4) as rt:
            rt.add_stream("cam-0", "fake://0")
            rt.add_stream("cam-1", "fake://1")
            time.sleep(1.5)          # flood the bounded frame queue
            items = collect(rt, 1000, timeout_s=2.0)
            events = [i for i in items if isinstance(i, Event)]
            assert len(events) == 2  # one per stream, never dropped
            assert rt.frames_dropped > 0
    finally:
        restore_env(saved)


def test_results_raises_runtime_gone_after_crash(tmp_path):
    saved = set_env(fake_env(FAKE_FPS="30", FAKE_CRASH_AFTER="1"))
    try:
        with make_rt(tmp_path) as rt:
            rt.add_stream("cam-0", "fake://0")
            # second control line crashes the fake child
            with pytest.raises((RuntimeGone, TimeoutError)):
                rt.remove_stream("cam-0", timeout_s=2.0)
            got_frames = False
            with pytest.raises(RuntimeGone):
                for item in rt.results(timeout_s=3.0):
                    got_frames = isinstance(item, FrameResult) or got_frames
            assert got_frames
    finally:
        restore_env(saved)


def test_context_manager_stops_child(tmp_path):
    saved = set_env(fake_env(FAKE_FPS="20"))
    rt = None
    try:
        with make_rt(tmp_path) as rt:
            rt.add_stream("cam-0", "fake://0")
            assert rt._client is not None and rt._client.proc is not None
        assert rt._client is None
    finally:
        restore_env(saved)


def test_snapshot_and_configure_analyzer(tmp_path):
    saved = set_env(fake_env(FAKE_FPS="20"))
    try:
        with make_rt(tmp_path) as rt:
            rt.add_stream("cam-0", "fake://0")
            applied = rt.configure_analyzer("cam-0", "zone",
                                            {"zones": []})
            assert applied.get("name") == "zone"
            jpeg = rt.snapshot("cam-0")
            assert jpeg.startswith(b"\xff\xd8")
    finally:
        restore_env(saved)


@pytest.mark.native
@pytest.mark.skipif(not os.environ.get("VB_RUNTIME_BIN"),
                    reason="needs VB_RUNTIME_BIN")
def test_real_runtime_synthetic(tmp_path):
    with Runtime.from_config(write_config(tmp_path)) as rt:
        rt.add_stream("cam-0", "synthetic://?w=320&h=240&fps=15&boxes=2")
        items = collect(rt, 20, timeout_s=3.0)
        frames = [i for i in items if isinstance(i, FrameResult)]
        assert len(frames) >= 20
