"""Tests for vision_base.geom + hook CPU budget metering (spec BASE-1
§6.5.5 / §10.5④, M1.22). Uses tests/fake_runtime.py at 15 fps.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

jsonschema = pytest.importorskip("jsonschema")

from vision_base import geom
from vision_base import letterbox
from vision_base.health import HealthServer, build_healthz
from vision_base.shard import HOOK_BUDGET_WINDOW_S, Shard
from vision_base.types import Detection, FrameResult, StreamSpec

FAKE = os.path.join(os.path.dirname(__file__), "fake_runtime.py")
FAKE_ARGV = [sys.executable, FAKE, "--ipc-fd", "{fd}"]

ROOT = pathlib.Path(__file__).resolve().parents[3] / "contracts"
HEALTHZ_SCHEMA = json.loads((ROOT / "vb-healthz.schema.json").read_text())

RT_CFG = {"backend": {"name": "synthetic", "max_batch": 2, "model_w": 320,
                      "model_h": 320, "boxes": 2, "infer_ms": 0},
          "contexts_per_worker": 1, "tracker": {},
          "analyzers": {"plugins": []}, "snapshot_ring": 2}


class FakePublisher:
    def submit(self, topic, payload, qos=0, retain=False):
        pass


class EmptyApp:
    name = "empty"
    wants_frames = True

    def analyzers(self, spec):
        return []

    def plugins(self):
        return []

    def on_stream_added(self, ctx):
        pass

    def on_stream_removed(self, ctx):
        pass

    def on_event(self, ctx, ev):
        return []

    def on_frame(self, ctx, res):
        return []      # empty hook: ~0 CPU


class BusyApp(EmptyApp):
    name = "busy"

    def on_frame(self, ctx, res):
        end = time.monotonic() + 0.03   # busy-wait 30 ms per frame
        while time.monotonic() < end:
            pass
        return []


def wait_for(predicate, timeout_s=15.0, what="condition"):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {what}")


def make_shard(app, env=None):
    saved = {}
    for k, v in (env or {"FAKE_FPS": "15"}).items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    shard = Shard(0, app, FakePublisher(), runtime_argv=FAKE_ARGV,
                  runtime_cfg=RT_CFG, state_dir="/tmp/vb-test-hookbudget",
                  topic_root="vb", restart_backoff_s=0.3,
                  hello_timeout_s=5.0)
    return shard, saved


def restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ------------------------------------------------------------------- geom


def test_joint_angle_deg_matches_pose_angle_formula():
    """Same cases as the pose_angle fixture plan (§6.2.5): collinear = 180,
    right angle = 90 (1e-4); pixel-computed != normalized-computed."""
    # collinear points -> 180 deg
    assert abs(geom.joint_angle_deg((0.1, 0.1), (0.5, 0.5), (0.9, 0.9),
                                    1920, 1080) - 180.0) < 1e-4
    # right angle at B (axis-aligned legs survive anisotropic scaling)
    assert abs(geom.joint_angle_deg((0.4, 0.5), (0.5, 0.5), (0.5, 0.2),
                                    1920, 1080) - 90.0) < 1e-4
    # 1920x1080 source: angle in pixels differs from the (wrong) value
    # computed on raw normalized coords. Hand case: A=(384,216), B=(960,540),
    # C=(1536,216); v1=(-576,-324), v2=(576,-324);
    # cos = (324^2-576^2)/(576^2+324^2) = -226800/436752
    a, b, c = (0.2, 0.2), (0.5, 0.5), (0.8, 0.2)
    ang_px = geom.joint_angle_deg(a, b, c, 1920, 1080)
    expected = math.degrees(math.acos(-226800.0 / 436752.0))
    assert abs(ang_px - expected) < 1e-4
    ang_norm = math.degrees(math.acos(  # isotropic/normalized: v1⊥v2 -> 90
        (((a[0] - b[0]) * (c[0] - b[0])) + ((a[1] - b[1]) * (c[1] - b[1]))) /
        (math.hypot(a[0] - b[0], a[1] - b[1]) *
         math.hypot(c[0] - b[0], c[1] - b[1]))))
    assert abs(ang_norm - 90.0) < 1e-9
    assert abs(ang_px - ang_norm) > 1.0   # pixel and normalized differ


def test_source_boxes_matches_letterbox():
    g = letterbox.fit(1280, 720, 320, 320)
    det = Detection(cx=0.5, cy=0.5, w=0.2, h=0.3, score=0.9,
                    class_id=3, track_id=7)
    res = FrameResult(stream_id="cam-0", seq=1, wall_ms=0.0, geom=g,
                      detections=[det], inference_ms=0.0, queue_delay_ms=0.0)
    boxes = geom.source_boxes(res)
    assert len(boxes) == 1
    track_id, class_id, score, (x0, y0, x1, y1) = boxes[0]
    assert (track_id, class_id, score) == (7, 3, 0.9)
    cx, cy, w, h = letterbox.box_to_source_norm(g, 0.5, 0.5, 0.2, 0.3)
    assert x0 == pytest.approx(cx - w / 2)
    assert y0 == pytest.approx(cy - h / 2)
    assert x1 == pytest.approx(cx + w / 2)
    assert y1 == pytest.approx(cy + h / 2)


def test_source_keypoints_and_point_in_polygon():
    g = letterbox.fit(100, 100, 100, 100)      # identity geometry
    det = Detection(cx=0.5, cy=0.5, w=0.1, h=0.1, score=0.9,
                    class_id=0, track_id=1, keypoints=(0.2, 0.4, 0.9,
                                                       0.6, 0.4, 0.5))
    res = FrameResult(stream_id="s", seq=0, wall_ms=0.0, geom=g,
                      detections=[det], inference_ms=0.0, queue_delay_ms=0.0)
    kpts = geom.source_keypoints(res, det)
    assert kpts[0] == pytest.approx((0.2, 0.4, 0.9))
    assert kpts[1] == pytest.approx((0.6, 0.4, 0.5))
    square = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
    assert geom.point_in_polygon(0.5, 0.5, square) is True
    assert geom.point_in_polygon(0.1, 0.5, square) is False
    assert geom.point_in_polygon(0.5, 0.5, [[0, 0], [1, 1]]) is False


# ------------------------------------------------------------- hook budget


def test_busy_hook_exceeds_budget():
    shard, saved = make_shard(BusyApp(), env={"FAKE_FPS": "15"})
    try:
        shard.start()
        wait_for(lambda: shard._client is not None, timeout_s=25.0,
                 what="runtime hello")
        assert shard.add_stream(StreamSpec("cam-0", "fake://"))["ok"] is True
        # 30 ms busy per frame at 15 fps -> ~0.45 core > 0.10 budget
        wait_for(lambda: shard.hook_budget_status()["exceeded"],
                 timeout_s=12, what="hook_budget.exceeded")
        st = shard.hook_budget_status()
        assert st["advice"]
        assert "C ABI" in st["advice"]
        total = sum(st["by_method"].values())
        assert st["by_method"]["on_frame"] / total > 0.9
        assert st["limit"] == 0.10
    finally:
        restore_env(saved)
        shard.stop()


def test_empty_hook_within_budget():
    shard, saved = make_shard(EmptyApp())
    try:
        shard.start()
        wait_for(lambda: shard._client is not None, timeout_s=25.0,
                 what="runtime hello")
        assert shard.add_stream(StreamSpec("cam-0", "fake://"))["ok"] is True
        time.sleep(2.5)
        st = shard.hook_budget_status()
        assert st["exceeded"] is False
        assert st["advice"] == ""
        assert st["core"] < st["limit"]
    finally:
        restore_env(saved)
        shard.stop()


def test_hook_budget_custom_limit():
    app = BusyApp()
    app.options = {"hook_budget_core": 0.05}   # tighter than default
    shard, saved = make_shard(app)
    try:
        shard.start()
        wait_for(lambda: shard._client is not None, timeout_s=25.0,
                 what="runtime hello")
        assert shard.add_stream(StreamSpec("cam-0", "fake://"))["ok"] is True
        wait_for(lambda: shard.hook_budget_status()["exceeded"],
                 timeout_s=25, what="exceeded with custom limit")
        assert shard.hook_budget_status()["limit"] == 0.05
    finally:
        restore_env(saved)
        shard.stop()


def test_healthz_schema_and_200_with_exceeded():
    """hook_budget rides the heartbeat into /healthz; exceeded does not
    change the 200 status (§10.5④)."""
    hb = {
        "core": 0.45, "limit": 0.10, "exceeded": True,
        "by_method": {"on_frame": 0.44, "on_event": 0.0, "other": 0.01},
        "advice": "per-frame hook exceeds budget; move it to a C ABI "
                  "analyzer plugin (docs/extending.md)",
    }
    shards = [{
        "index": 0, "pid": 1234, "alive": True, "restarts": 0,
        "rss_kb": None, "cpu_s": None, "contexts": 1,
        "stream_ids": ["cam-0"],
        "runtime": {"pid": 1235, "alive": True, "restarts": 0,
                    "rss_kb": None, "cpu_s": None, "version": "0",
                    "backend": "synthetic"},
        "mqtt": {"connected": True, "queue_depth": 0, "dropped": 0},
        "hook_budget": hb,
    }]
    body = build_healthz(device_id="dev-1", backend="synthetic",
                         uptime_s=1.0, control_connected=True,
                         supervisor_pid=os.getpid(), shards=shards,
                         streams=[])
    assert body["shards"][0]["hook_budget"] == hb
    assert body["status"] == "ok"
    jsonschema.validate(instance=body, schema=HEALTHZ_SCHEMA)

    server = HealthServer("127.0.0.1", 0, lambda: body)
    server.start()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{server.actual_port}/healthz") as r:
            assert r.status == 200          # exceeded != degraded
            got = json.loads(r.read())
        assert got["shards"][0]["hook_budget"]["exceeded"] is True
        jsonschema.validate(instance=got, schema=HEALTHZ_SCHEMA)
    finally:
        server._srv.shutdown()
        server._srv.server_close()


# ------------------------------------------------------------- review item 13


def test_hook_budget_divides_by_the_window_not_the_sample_age():
    """§6.5.5: `core = Σ window CPU ÷ 10 s`. Dividing by the age of the oldest
    sample turned one 0.02 s call sampled 0.1 s later into 0.2 core instead of
    0.002 (review item 13)."""
    shard, saved = make_shard(EmptyApp())
    try:
        with shard._cv:
            shard._hook_win_start = time.monotonic() - 0.1
            shard._hook_win_cpu["on_frame"] = 0.02
        st = shard.hook_budget_status()
        assert st["core"] == pytest.approx(0.002, abs=1e-9)
        assert st["exceeded"] is False
        assert st["by_method"]["on_frame"] == pytest.approx(0.02)
    finally:
        restore_env(saved)
        shard.stop()


def test_hook_budget_window_rolls_over():
    """The heartbeat polls the running window every second, so the final
    reading of a window is taken just before it rolls; once it rolls the burst
    no longer counts."""
    shard, saved = make_shard(EmptyApp())
    try:
        with shard._cv:
            shard._hook_win_start = time.monotonic() - HOOK_BUDGET_WINDOW_S + 1.0
            shard._hook_win_cpu["on_event"] = 3.0     # 0.3 core over the window
        st = shard.hook_budget_status()               # still inside the window
        assert st["core"] == pytest.approx(0.3, abs=1e-9)
        assert st["exceeded"] is True
        with shard._cv:                               # ... now it expires
            shard._hook_win_start = time.monotonic() - HOOK_BUDGET_WINDOW_S - 0.5
        st2 = shard.hook_budget_status()
        assert st2["core"] == 0.0 and st2["exceeded"] is False
        assert st2["by_method"] == {"on_frame": 0.0, "on_event": 0.0, "other": 0.0}
    finally:
        restore_env(saved)
        shard.stop()


def test_hook_slow_counts_and_warns(caplog):
    """§6.5.1: one hook call over a second is warned about and counted."""
    class SlowOnce(EmptyApp):
        def on_frame(self, ctx, res):
            # Burn *thread CPU*, which is what the meter reads: a sleeping hook
            # is (correctly) not slow, and on macOS a wall-clock busy loop
            # accrues thread_time at roughly half the wall rate, so burning by
            # wall clock cannot reach the 1 s threshold reliably.
            t0 = time.thread_time()
            cap = time.monotonic() + 15.0
            while time.thread_time() - t0 < 1.05 and time.monotonic() < cap:
                pass
            return []

    shard, saved = make_shard(SlowOnce())
    try:
        shard.start()
        # a loaded machine can outrun hello_timeout_s; without a live runtime
        # the add answers ok:false and no frame (and no slow hook) ever comes
        wait_for(lambda: shard._client is not None, timeout_s=25.0,
                 what="runtime hello")
        assert shard.add_stream(StreamSpec("cam-0", "fake://"))["ok"] is True
        with caplog.at_level(logging.WARNING, logger="vb.shard"):
            wait_for(lambda: shard.hook_slow >= 1, timeout_s=25,
                     what="hook_slow counted")
        assert any("hook on_frame took" in r.message for r in caplog.records)
        assert shard.hook_budget_status()["slow"] == shard.hook_slow
    finally:
        restore_env(saved)
        shard.stop()


# ------------------------------------------------------------- review item 20


def test_health_server_sheds_connections_over_the_cap():
    """Review item 20: the unauthenticated endpoint must not let slow clients
    exhaust the supervisor's threads and FDs."""
    import socket

    body = {"status": "ok", "device_id": "d", "backend": "cpu",
            "base_version": "0", "uptime_s": 0.0,
            "mqtt": {"control_connected": True, "publish_connected": [],
                     "queue_depth": [], "dropped": []},
            "supervisor": {"pid": 1, "rss_kb": None, "cpu_s": None},
            "shards": [], "rss_kb_total": 0, "streams": []}
    server = HealthServer("127.0.0.1", 0, lambda: body, max_connections=2)
    server.start()
    held = []
    try:
        for _ in range(2):
            s = socket.create_connection(("127.0.0.1", server.actual_port))
            s.settimeout(5.0)
            held.append(s)          # open but never sends a request
        wait_for(lambda: server._srv.served == 2, timeout_s=5,
                 what="two connections in flight")

        extra = socket.create_connection(("127.0.0.1", server.actual_port))
        extra.settimeout(5.0)
        try:
            data = extra.recv(4096)
        finally:
            extra.close()
        assert b"503" in data
        assert server._srv.shed == 1
    finally:
        for s in held:
            s.close()
        server.stop()


def test_health_server_has_a_request_timeout():
    """A connection that never sends anything is dropped instead of holding a
    thread forever."""
    from vision_base.health import REQUEST_TIMEOUT_S
    assert 0 < REQUEST_TIMEOUT_S <= 30
