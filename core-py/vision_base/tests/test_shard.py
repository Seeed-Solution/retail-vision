"""Tests for vision_base.shard + hooks (spec BASE-1 §8 M1.12): hook thread
(§6.5.1), stream-state delivery (§6.5.2), crash restart (§6.9 rule 2) and
EchoApp publish_hz. Uses tests/fake_runtime.py; the ``native``-marked test
runs against the real vb-runtime + synthetic backend via VB_RUNTIME_BIN.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

import pytest

from vision_base.hooks import EchoApp, StreamContext
from vision_base.runtime_client import RuntimeGone
from vision_base.shard import Shard
from vision_base.types import StreamSpec

FAKE = os.path.join(os.path.dirname(__file__), "fake_runtime.py")
FAKE_ARGV = [sys.executable, FAKE, "--ipc-fd", "{fd}"]

RT_CFG = {"backend": {"name": "synthetic", "max_batch": 2, "model_w": 320,
                      "model_h": 320, "boxes": 2, "infer_ms": 0},
          "contexts_per_worker": 1, "tracker": {},
          "analyzers": {"plugins": []}, "snapshot_ring": 2}


class FakePublisher:
    def __init__(self):
        self.lock = threading.Lock()
        self.messages: list[tuple] = []

    def submit(self, topic, payload, qos=0, retain=False):
        with self.lock:
            self.messages.append((topic, payload, qos, retain))

    def topics(self):
        with self.lock:
            return [m[0] for m in self.messages]

    def count(self, topic):
        with self.lock:
            return sum(1 for m in self.messages if m[0] == topic)


class RecordingApp:
    """Records lifecycle/state transitions; optionally exercises sync
    request_snapshot from the hook thread (§6.5.1)."""

    name = "recording"
    wants_frames = False

    def __init__(self, snapshot: bool = True):
        self.snapshot = snapshot
        self.lock = threading.Lock()
        self.states: list[tuple] = []
        self.snapshots: list[tuple] = []
        self.events: list[tuple] = []
        self.added: list[str] = []
        self.removed: list[str] = []

    def analyzers(self, spec):
        return []

    def plugins(self):
        return []

    def on_stream_added(self, ctx):
        with self.lock:
            self.added.append(ctx.stream_id)

    def on_stream_removed(self, ctx):
        with self.lock:
            self.removed.append(ctx.stream_id)

    def on_event(self, ctx, ev):
        if self.snapshot:
            meta, jpeg = ctx.request_snapshot(max_side=192, timeout_s=3.0)
        else:
            meta, jpeg = {}, b""
        with self.lock:
            self.snapshots.append((ctx.stream_id, meta, jpeg))
            self.events.append((ctx.stream_id, ev.type))
        return []

    def on_frame(self, ctx, res):
        return []

    def on_stream_state(self, ctx, state, error):
        with self.lock:
            self.states.append((ctx.stream_id, state))
        return []


class SlowFrameApp:
    name = "slowframe"
    wants_frames = True

    def __init__(self):
        self.lock = threading.Lock()
        self.frames: list[tuple] = []

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
        time.sleep(0.2)  # slow hook: frames pile up in the length-1 slot
        with self.lock:
            self.frames.append((ctx.stream_id, res.seq))
        return []



def wait_for(predicate, timeout_s=6.0, what="condition"):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


def start_ready(shard, timeout_s=25.0):
    """start() + wait for hello: a loaded machine can outrun hello_timeout_s,
    after which add_stream answers ok:false instead of raising."""
    shard.start()
    wait_for(lambda: shard._client is not None, timeout_s=timeout_s,
             what="runtime hello")


def make_shard(app, publisher, *, env=None, backoff=0.3, argv=None):
    old = {}
    if env:
        for k, v in env.items():
            old[k] = os.environ.get(k)
            os.environ[k] = v
    try:
        shard = Shard(0, app, publisher, runtime_argv=argv or FAKE_ARGV,
                      runtime_cfg=RT_CFG, state_dir="/tmp/vb-test-shard",
                      topic_root="vb", restart_backoff_s=backoff,
                      hello_timeout_s=5.0)
        return shard, old
    finally:
        pass


def cleanup_env(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --------------------------------------------------------------------- tests


def test_hook_thread_snapshot_no_deadlock():
    app = RecordingApp()
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_FPS": "30"})
    try:
        shard.start()
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        # on_event (hook thread) synchronously calls ctx.request_snapshot;
        # must return within 3 s (§6.5.1 acceptance).
        wait_for(lambda: app.snapshots, what="snapshot from hook thread")
        sid, meta, jpeg = app.snapshots[0]
        assert sid == "cam-0"
        assert meta["mime"] == "image/jpeg"
        assert jpeg.startswith(b"\xff\xd8")
    finally:
        cleanup_env(old)
        shard.stop()


def test_reader_thread_request_raises():
    app = RecordingApp()
    pub = FakePublisher()
    shard, old = make_shard(app, pub)
    errors = []
    orig = shard._on_stats

    def on_stats(stats):
        try:
            shard._client.request("set_threshold", 0.5, stream_index=0,
                                  value=0.5)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    shard._on_stats = on_stats
    try:
        os.environ["FAKE_STATS"] = "1"
        shard.start()
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        wait_for(lambda: errors, what="reader-thread RuntimeError")
        assert isinstance(errors[0], RuntimeError)
        assert "deadlock" in str(errors[0])
    finally:
        os.environ.pop("FAKE_STATS", None)
        cleanup_env(old)
        shard.stop()


def test_stream_states_and_dedup():
    app = RecordingApp()
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_FPS": "10"})
    try:
        shard.start()
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        wait_for(lambda: ("cam-0", "running") in app.states, what="running")
        assert ("cam-0", "starting") in app.states
        # no duplicate starting/running deliveries
        starting = [s for s in app.states if s == ("cam-0", "starting")]
        running = [s for s in app.states if s == ("cam-0", "running")]
        assert len(starting) == 1 and len(running) == 1
    finally:
        cleanup_env(old)
        shard.stop()


def test_crash_reconnect_restart_same_stream_index():
    app = RecordingApp(snapshot=False)
    pub = FakePublisher()
    # crash after control line 1 (the add): the next line (set_threshold)
    # kills the child mid-request.
    shard, old = make_shard(app, pub, env={"FAKE_CRASH_AFTER": "1",
                                           "FAKE_FPS": "30"})
    try:
        shard.start()
        assert shard.add_stream(StreamSpec("cam-0", "fake://"))["ok"] is True
        wait_for(lambda: ("cam-0", "running") in app.states, what="first running")
        try:
            shard.set_threshold("cam-0", 0.5)   # line 2 -> fake crashes
        except RuntimeGone:
            pass  # crash mid-request is the expected outcome
        wait_for(lambda: ("cam-0", "reconnecting") in app.states,
                 what="reconnecting after crash")
        assert shard.runtime_restarts >= 1
        # restarted runtime re-adds with the same stream_index -> the app sees
        # starting -> running again.
        def recovered():
            states = app.states
            i = states.index(("cam-0", "reconnecting"))
            return ("cam-0", "running") in states[i + 1:] \
                and ("cam-0", "starting") in states[i + 1:]
        wait_for(recovered, what="restart recovery to running")
        assert app.added == ["cam-0"]   # on_stream_added not re-run on restart
    finally:
        cleanup_env(old)
        shard.stop()


def test_three_pre_hello_exits_error():
    app = RecordingApp()
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_EXIT_BEFORE_HELLO": "1"},
                            backoff=0.2)
    try:
        shard.start()     # fails immediately (exit before hello)
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        wait_for(lambda: ("cam-0", "error") in app.states, timeout_s=15,
                 what="error after 3 pre-hello exits")
        # exactly max_start_failures real pre-hello exits before "error"
        assert shard._start_failures == shard.max_start_failures == 3
        assert ("cam-0", "reconnecting") in app.states
        # restarts stop: no further state churn
        n = len(app.states)
        time.sleep(0.8)
        assert len(app.states) == n
    finally:
        cleanup_env(old)
        shard.stop()


def test_hook_frames_dropped_increments():
    app = SlowFrameApp()
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_FPS": "60"})
    try:
        shard.start()
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        wait_for(lambda: len(app.frames) >= 2, what="slow hook frames")
        status = shard.stream_status()
        assert status and status[0]["stream_id"] == "cam-0"
        assert status[0]["hook_frames_dropped"] > 0
    finally:
        cleanup_env(old)
        shard.stop()


def test_echo_app_publishes_at_publish_hz():
    app = EchoApp()
    app.configure({"publish_hz": 10.0, "frame_stride": 1}, "dev-1")
    assert app.publish_hz == 10.0
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_FPS": "30"})
    try:
        shard.start()
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        topic = "vb/results/cam-0"
        wait_for(lambda: pub.count(topic) >= 3, timeout_s=8,
                 what="echo publications")
        with pub.lock:
            payloads = [m[1] for m in pub.messages if m[0] == topic]
        p = payloads[-1]
        assert p["stream_id"] == "cam-0"
        assert p["detections"] and p["detections"][0]["track_id"] >= 1
        # publish_hz ~10 with 30 fps source: bounded number of publications
        time.sleep(1.0)
        assert pub.count(topic) <= 10 * 2 + 5
    finally:
        cleanup_env(old)
        shard.stop()


def test_remove_stream():
    app = RecordingApp()
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_FPS": "30"})
    try:
        shard.start()
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        wait_for(lambda: ("cam-0", "running") in app.states, what="running")
        shard.remove_stream("cam-0")
        wait_for(lambda: ("cam-0", "stopped") in app.states, what="stopped")
        wait_for(lambda: app.removed == ["cam-0"], what="on_stream_removed")
        assert shard.stream_status() == []
    finally:
        cleanup_env(old)
        shard.stop()


@pytest.mark.native
@pytest.mark.skipif(not os.environ.get("VB_RUNTIME_BIN"),
                    reason="needs VB_RUNTIME_BIN")
def test_real_runtime_synthetic_backend(tmp_path):
    bin_path = os.environ["VB_RUNTIME_BIN"]
    app = EchoApp()
    app.configure({"publish_hz": 5.0, "frame_stride": 1}, "dev-native")
    pub = FakePublisher()
    shard = Shard(0, app, pub,
                  runtime_argv=[bin_path, "--ipc-fd", "{fd}"],
                  runtime_cfg=RT_CFG, state_dir=str(tmp_path),
                  topic_root="vb", restart_backoff_s=0.5,
                  hello_timeout_s=10.0)
    try:
        shard.start()
        assert shard.hello.backend == "synthetic"
        assert shard.add_stream(StreamSpec(
            "cam-0", "synthetic://?w=320&h=240&fps=15&boxes=2"))["ok"] is True
        topic = "vb/results/cam-0"
        wait_for(lambda: pub.count(topic) >= 3, timeout_s=15,
                 what="native runtime publications")
        with pub.lock:
            payloads = [m[1] for m in pub.messages if m[0] == topic]
        assert payloads[-1]["detections"], "synthetic backend must emit boxes"
    finally:
        shard.stop()


def test_repeated_state_delivered_once():
    """Review 2026-09-26: the same state arriving twice reaches the hook once."""
    app = RecordingApp()
    pub = FakePublisher()
    shard, old = make_shard(app, pub)
    try:
        shard.start()
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        wait_for(lambda: ("cam-0", "running") in app.states, timeout_s=10, what="running")
        idx = next(iter(shard._active))
        n = sum(1 for st in app.states if st == ("cam-0", "running"))
        shard._on_state(idx, "running", "")
        shard._on_state(idx, "running", "")
        time.sleep(0.3)
        assert sum(1 for st in app.states if st == ("cam-0", "running")) == n
    finally:
        cleanup_env(old)
        shard.stop()


# ------------------------------------------------------------- review items


def test_replay_continues_after_one_failed_add(tmp_path):
    """Review item 2: one rejected `add` during the restart replay must not
    abandon the remaining streams (§6.9 rule 2 replays in index order)."""
    marker = tmp_path / "reject"
    app = RecordingApp(snapshot=False)
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_FPS": "10",
                                           "FAKE_REJECT_STREAM": "cam-0",
                                           "FAKE_REJECT_FILE": str(marker)},
                            backoff=0.3)
    try:
        start_ready(shard)
        assert shard.add_stream(StreamSpec("cam-0", "fake://"))["ok"] is True
        assert shard.add_stream(StreamSpec("cam-1", "fake://"))["ok"] is True
        wait_for(lambda: ("cam-0", "running") in app.states)
        wait_for(lambda: ("cam-1", "running") in app.states)
        marker.write_text("x")              # from now on cam-0 is rejected
        os.kill(shard._client.pid, signal.SIGKILL)
        wait_for(lambda: ("cam-0", "error") in app.states, timeout_s=15,
                 what="failed replay reported")
        # cam-0 is index 0, so cam-1 is only re-added if the loop continued
        wait_for(lambda: sum(1 for s in app.states
                             if s == ("cam-1", "running")) >= 2,
                 timeout_s=15, what="cam-1 replayed after cam-0 failed")
    finally:
        cleanup_env(old)
        shard.stop()


def test_configure_analyzer_keeps_the_rest_of_the_set():
    """Review item 5: reconfiguring one analyzer must not drop the others from
    the set replayed after a vb-runtime restart (§6.5.3)."""
    class TwoAnalyzerApp:
        name, wants_frames = "two", False
        def analyzers(self, spec):
            return [{"name": "line_cross", "config": {"a": 1}},
                    {"name": "zone", "config": {"b": 2}}]

    class FakeRuntime:
        def request(self, op, timeout_s, **fields):
            return {"ok": True, "applied": {"name": fields["name"]}}

    from vision_base.hooks import StreamContext
    app = TwoAnalyzerApp()
    ctx = StreamContext(0, StreamSpec("cam-0", "fake://"), FakeRuntime())
    base = [{"name": "line_cross", "config": {"a": 1}},
            {"name": "zone", "config": {"b": 2}}]
    assert ctx.analyzer_configs(app) == base
    ctx.configure_analyzer("zone", {"b": 9})
    assert ctx.analyzer_configs(app) == [
        {"name": "line_cross", "config": {"a": 1}},
        {"name": "zone", "config": {"b": 9}}]
    ctx.configure_analyzer("dwell", {"t": 1.0})     # not in analyzers(spec)
    assert [a["name"] for a in ctx.analyzer_configs(app)] == \
        ["line_cross", "zone", "dwell"]


def test_set_threshold_survives_runtime_restart():
    """Review item 6: an applied threshold must be what the shard keeps and
    replays, not the value it was created with."""
    app = RecordingApp(snapshot=False)
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_FPS": "10"}, backoff=0.3)
    try:
        start_ready(shard)
        shard.add_stream(StreamSpec("cam-0", "fake://", score_threshold=0.3))
        wait_for(lambda: ("cam-0", "running") in app.states)
        shard.set_threshold("cam-0", 0.9)
        assert shard.status_streams()[0]["score_threshold"] == 0.9

        replayed = []
        orig = shard._send_add
        def spy(client, ctx, *, first):
            replayed.append(ctx.spec.score_threshold)
            return orig(client, ctx, first=first)
        shard._send_add = spy
        os.kill(shard._client.pid, signal.SIGKILL)
        wait_for(lambda: 0.9 in replayed, timeout_s=15,
                 what="replayed add carries the new threshold")
    finally:
        cleanup_env(old)
        shard.stop()


def test_status_streams_preserves_native_telemetry_without_deriving_missing_values():
    """Native stream stats reach health with their units and optionality intact."""
    app = RecordingApp(snapshot=False)
    pub = FakePublisher()
    shard, old = make_shard(app, pub)
    try:
        ctx = StreamContext(0, StreamSpec("cam-0", "fake://"), runtime=None)
        ctx.last_state = "running"
        shard.contexts[0] = ctx
        shard._active.add(0)
        shard.last_stats = {"streams": [{
            "stream_index": 0,
            "fps": 12.5,
            "decode": "nvdec",
            "fallback_active": False,
            "processed_frames": 7,
            "inference_ms_p50": 12.5,
            "inference_ms_p95": 18.75,
            "queue_delay_ms_p95": 42.0,
            "dropped_frames": 4,
            "rate_skipped": 3,
        }]}
        status = shard.status_streams()[0]
        assert status["processed_frames"] == 7
        assert status["inference_ms_p50"] == 12.5
        assert status["inference_ms_p95"] == 18.75
        assert status["queue_delay_ms_p95"] == 42.0
        assert status["dropped_frames"] == 4
        assert status["rate_skipped"] == 3
        assert status["decode"] == "nvdec"
        assert status["fallback_active"] is False

        shard.last_stats = {"streams": [{"stream_index": 0, "fps": 0.0}]}
        missing = shard.status_streams()[0]
        assert "inference_ms_p95" not in missing
        assert "processed_frames" not in missing
        assert "inference_ms_p50" not in missing
        assert "queue_delay_ms_p95" not in missing
        assert "dropped_frames" not in missing
        assert "rate_skipped" not in missing
        assert missing["decode"] == ""
        assert missing["fallback_active"] is False
    finally:
        cleanup_env(old)
        shard.stop()


def test_runtime_status_preserves_native_snapshot_metadata():
    app = RecordingApp(snapshot=False)
    pub = FakePublisher()
    shard, old = make_shard(app, pub)
    try:
        shard.last_stats = {"effective_contexts": 2,
                            "stats_monotonic_ms": 1234,
                            "stats_wall_ms": 1700000000123,
                            "stats_pid": 9876}
        status = shard.runtime_status()
        assert status["effective_contexts"] == 2
        assert status["stats_monotonic_ms"] == 1234
        assert status["stats_wall_ms"] == 1700000000123
        assert status["stats_pid"] == 9876
        shard.last_stats = {}
        missing = shard.runtime_status()
        assert "effective_contexts" not in missing
        assert "stats_monotonic_ms" not in missing
        assert "stats_wall_ms" not in missing
        assert "stats_pid" not in missing
    finally:
        cleanup_env(old)
        shard.stop()


def test_failed_add_releases_the_stream_id():
    """Review item 7: a rejected add must roll its registration back instead of
    reserving the stream_id forever."""
    app = RecordingApp(snapshot=False)
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_REJECT_STREAM": "cam-0"})
    try:
        start_ready(shard)
        with pytest.raises(Exception) as first:
            shard.add_stream(StreamSpec("cam-0", "fake://"))
        assert "duplicate" not in str(first.value).lower()
        assert "cam-0" not in shard._by_stream_id
        # the second attempt reaches the native side again (and is rejected
        # there), rather than failing locally as a duplicate stream_id
        with pytest.raises(Exception) as second:
            shard.add_stream(StreamSpec("cam-0", "fake://"))
        assert "duplicate" not in str(second.value).lower()
        assert shard.stream_status() == []
    finally:
        cleanup_env(old)
        shard.stop()


def test_remove_stream_releases_the_stream_id_and_context():
    """Review item 7: after on_stream_removed the mapping and the context are
    released, so the same stream_id can be added again."""
    app = RecordingApp(snapshot=False)
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_FPS": "10"})
    try:
        start_ready(shard)
        assert shard.add_stream(StreamSpec("cam-0", "fake://"))["ok"] is True
        wait_for(lambda: ("cam-0", "running") in app.states)
        old_idx = shard._by_stream_id["cam-0"]
        shard.remove_stream("cam-0")
        wait_for(lambda: app.removed == ["cam-0"], what="on_stream_removed")
        wait_for(lambda: old_idx not in shard.contexts, what="context released")
        assert "cam-0" not in shard._by_stream_id
        assert shard.add_stream(StreamSpec("cam-0", "fake://"))["ok"] is True
    finally:
        cleanup_env(old)
        shard.stop()


def test_stop_while_runtime_is_starting_reclaims_the_child():
    """Review item 11: stop() during the hello wait must not leave a runtime
    behind, and the start path must not install it afterwards."""
    app = RecordingApp(snapshot=False)
    pub = FakePublisher()
    shard, old = make_shard(app, pub, env={"FAKE_NO_HELLO": "1"})
    try:
        t = threading.Thread(target=shard.start, daemon=True)
        t.start()
        wait_for(lambda: shard._starting is not None, what="start in flight")
        child = shard._starting.proc
        assert child.poll() is None

        t0 = time.monotonic()
        shard.stop()
        assert time.monotonic() - t0 < 6.0
        t.join(6.0)
        assert not t.is_alive(), "start path did not converge"
        assert shard._client is None
        wait_for(lambda: child.poll() is not None, what="child reclaimed")
    finally:
        cleanup_env(old)
        shard.stop()


def test_shard_config_write_refuses_a_symlinked_path(tmp_path):
    """Review item 23: the per-shard runtime config is written through an
    exclusive temp file and never follows a pre-placed symlink."""
    state = tmp_path / "shard-0"
    state.mkdir()
    victim = tmp_path / "victim.json"
    victim.write_text("untouched")
    (state / "rt-0.json").symlink_to(victim)

    shard = Shard(0, EchoApp(), FakePublisher(), runtime_argv=FAKE_ARGV,
                  runtime_cfg=RT_CFG, state_dir=str(state))
    with pytest.raises(OSError):
        shard.start()
    assert victim.read_text() == "untouched"
    assert not list(state.glob("*.tmp"))
