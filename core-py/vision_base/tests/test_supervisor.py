"""Tests for vision_base.supervisor / health / main (spec BASE-1 §8 M1.13):
sharding formula, client_id suffixes, LWT control session, shard restart
(§6.9 rule 1), vb-runtime restart with control session intact (rule 2),
/healthz + vb.status/1 schema conformance. Uses tests/fake_broker.py and
tests/fake_runtime.py as the vb-runtime children.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

jsonschema = pytest.importorskip("jsonschema")

from vision_base.config import load
from vision_base.supervisor import Supervisor

from fake_broker import FakeBroker  # noqa: E402

FAKE = os.path.join(os.path.dirname(__file__), "fake_runtime.py")
FAKE_ARGV = [sys.executable, FAKE, "--ipc-fd", "{fd}"]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "..", "contracts"))
STATUS_SCHEMA = json.load(open(os.path.join(ROOT, "vb-status.schema.json")))
HEALTHZ_SCHEMA = json.load(open(os.path.join(ROOT, "vb-healthz.schema.json")))


def wait_for(predicate, timeout_s=20.0, what="condition"):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {what}")


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_config(tmp_path, broker, *, n_streams=5, per_worker=2,
                backoff=5.0, device="vb-test-01", workers="auto",
                app_module="vision_base.hooks:EchoApp", plugins=None):
    streams = [{"stream_id": f"cam-{i}", "url": f"fake://{i}"}
               for i in range(n_streams)]
    cfg = {
        "schema": "vb.config/1",
        "device_id": device,
        "backend": {"name": "fake", "model_path": "/tmp/fake.onnx",
                    "input_size": [320, 320]},
        "runtime": {"workers": workers, "max_streams_per_worker": per_worker,
                    "restart_backoff_s": backoff, "open_timeout_s": 5.0},
        "native": {"binary": sys.executable, "hello_timeout_s": 10.0},
        "state_dir": str(tmp_path / "state"),
        "mqtt": {"host": broker.host, "port": broker.port,
                 "topic_root": f"site/vision/{device}",
                 "status_interval_s": 1.0},
        "health": {"host": "127.0.0.1", "port": free_port()},
        "app": {"module": app_module},
        "streams": streams,
    }
    if plugins is not None:
        cfg["analyzers"] = {"plugins": list(plugins)}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return load(str(path))


def wait_ack(broker, topic, request_id, timeout_s=15.0):
    """Wait for the ack of one specific request_id."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for m in list(broker.published):
            if m["topic"] != topic:
                continue
            try:
                ack = json.loads(m["payload"])
            except ValueError:
                continue
            if ack.get("request_id") == request_id:
                return ack
        time.sleep(0.05)
    raise AssertionError(f"no ack for request_id {request_id!r}")


def send_command(broker, root, device, request_id, command, params):
    broker.publish_down(f"{root}/cmd/control", json.dumps({
        "schema": "vb.command/1", "device_id": device,
        "request_id": request_id, "command": command,
        "params": params}).encode())


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def get_healthz(port):
    import urllib.error
    url = f"http://127.0.0.1:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def kill_pid(pid):
    os.kill(pid, signal.SIGKILL)


@pytest.fixture
def broker():
    b = FakeBroker().start()
    yield b
    b.stop()


# ------------------------------------------------------------------- tests

def test_sharding_client_ids_and_schemas(tmp_path, broker):
    cfg = make_config(tmp_path, broker, n_streams=5, per_worker=2, backoff=0.5)
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port
        # 5 streams / max 2 per worker -> 3 shards, each with a live runtime
        def all_up():
            code, body = get_healthz(port)
            return (code == 200 and len(body["shards"]) == 3
                    and all(s["alive"] and s["runtime"]["alive"]
                            for s in body["shards"]))
        wait_for(all_up, timeout_s=30, what="3 shards + runtimes alive")

        code, body = get_healthz(port)
        assert code == 200
        jsonschema.validate(body, HEALTHZ_SCHEMA)
        # round-robin: cam-0/3 -> shard 0, cam-1/4 -> 1, cam-2 -> 2
        for sh in body["shards"]:
            expect = [f"cam-{i}" for i in range(5) if i % 3 == sh["index"]]
            assert sorted(sh["streams"]) == sorted(expect)
        # three distinct native children
        pids = [sh["runtime"]["pid"] for sh in body["shards"]]
        assert len(set(pids)) == 3
        for pid in pids:
            os.kill(pid, 0)          # still alive

        # client ids: <base>-0/-1/-2 publish sessions + <base>-ctl control
        wait_for(lambda: len([c for c in broker.connects
                              if c.get("client_id", "").endswith(
                                  ("-0", "-1", "-2"))]) >= 3,
                 what="shard publish sessions")
        cids = {c.get("client_id") for c in broker.connects}
        assert {"vb-test-01-0", "vb-test-01-1", "vb-test-01-2",
                "vb-test-01-ctl"} <= cids
        ctl = next(c for c in broker.connects if c.get("client_id") == "vb-test-01-ctl")
        assert ctl["will"]["topic"] == "site/vision/vb-test-01/status"
        assert ctl["will"]["payload"]  # LWT = status offline payload
        assert ctl["will_retain"] is True and ctl["will_qos"] == 1

        # shard publish sessions carry no LWT (§5.4 control-session row)
        for c in broker.connects:
            if c.get("client_id", "").endswith(("-0", "-1", "-2")):
                assert c["will"] is None

        # R/status retained payload validates against vb.status/1
        msg = broker.wait_publish("site/vision/vb-test-01/status")
        assert msg is not None and msg["retain"]
        status = json.loads(msg["payload"])
        jsonschema.validate(status, STATUS_SCHEMA)
        assert status["online"] is True and status["shards"] == 3
        assert len(status["streams"]) == 5

        # cmd/control add_stream -> ack via the control session
        broker.publish_down("site/vision/vb-test-01/cmd/control", json.dumps({
            "schema": "vb.command/1", "device_id": "vb-test-01",
            "request_id": "r-add-1", "command": "add_stream",
            "params": {"stream_id": "cam-99", "url": "fake://99"}}).encode())
        ack_msg = broker.wait_publish("site/vision/vb-test-01/cmd/ack")
        ack = json.loads(ack_msg["payload"])
        assert ack["request_id"] == "r-add-1" and ack["ok"] is True
        assert ack["applied"]["shard"] == 2   # fewest streams
    finally:
        sup.stop()


def test_kill_shard_restarted_after_backoff(tmp_path, broker):
    cfg = make_config(tmp_path, broker, n_streams=2, per_worker=1,
                      backoff=5.0, device="vb-test-02")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port

        def shard_up():
            code, body = get_healthz(port)
            return code == 200 and body["shards"][0]["alive"]
        wait_for(shard_up, timeout_s=30, what="initial shards")
        _, body = get_healthz(port)
        old_pid = body["shards"][0]["pid"]
        assert body["shards"][0]["restarts"] == 0

        kill_pid(old_pid)
        # §6.9 rule 1: alive:false, then restart after restart_backoff_s=5s
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["restarts"] == 1,
                 timeout_s=25, what="shard restarts=1")
        wait_for(shard_up, timeout_s=25, what="shard back alive")
        _, body = get_healthz(port)
        jsonschema.validate(body, HEALTHZ_SCHEMA)
        assert body["shards"][0]["pid"] != old_pid
        # streams re-added from the supervisor's record
        assert body["shards"][0]["streams"] == ["cam-0"]
        assert any(s["stream_id"] == "cam-0" for s in body["streams"])
    finally:
        sup.stop()


def test_kill_runtime_keeps_control_session(tmp_path, broker):
    cfg = make_config(tmp_path, broker, n_streams=1, per_worker=0,
                      backoff=0.5, device="vb-test-03")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port

        def up():
            code, body = get_healthz(port)
            return (code == 200 and body["shards"][0]["alive"]
                    and body["shards"][0]["runtime"]["alive"])
        wait_for(up, timeout_s=30, what="shard + runtime up")
        _, body = get_healthz(port)
        rt_pid = body["shards"][0]["runtime"]["pid"]
        assert body["shards"][0]["runtime"]["restarts"] == 0

        kill_pid(rt_pid)
        # §6.9 rule 2: shard survives, restarts its vb-runtime
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["runtime"]["restarts"] == 1,
                 timeout_s=20, what="runtime restarts=1")
        wait_for(up, timeout_s=20, what="runtime back alive")
        _, body = get_healthz(port)
        assert body["shards"][0]["runtime"]["pid"] != rt_pid
        # the supervisor control session never dropped
        assert body["mqtt"]["control_connected"] is True
        assert body["shards"][0]["alive"] is True

        # and it still answers commands end-to-end
        broker.publish_down("site/vision/vb-test-03/cmd/control", json.dumps({
            "schema": "vb.command/1", "device_id": "vb-test-03",
            "request_id": "r-list-1", "command": "list_streams",
            "params": {}}).encode())
        ack = json.loads(broker.wait_publish(
            "site/vision/vb-test-03/cmd/ack")["payload"])
        assert ack["ok"] is True
        ids = {s["stream_id"] for s in ack["applied"]["streams"]}
        assert ids == {"cam-0"}
    finally:
        sup.stop()


def test_stop_publishes_offline_status(tmp_path, broker):
    cfg = make_config(tmp_path, broker, n_streams=1, per_worker=0,
                      backoff=0.5, device="vb-test-04")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    sup.start()
    port = sup.health_port
    wait_for(lambda: get_healthz(port)[1]["shards"][0]["alive"],
             timeout_s=30, what="shard up")
    sup.stop()
    msg = broker.wait_publish("site/vision/vb-test-04/status")
    status = json.loads(msg["payload"])
    jsonschema.validate(status, STATUS_SCHEMA)
    assert status["online"] is False and "streams" not in status


@pytest.mark.native
@pytest.mark.skipif(not os.environ.get("VB_RUNTIME_BIN"),
                    reason="needs VB_RUNTIME_BIN")
def test_real_runtime_two_shards(tmp_path, broker):
    """Same acceptance path against the real vb-runtime (synthetic backend):
    2 streams / per_worker=1 -> 2 shards + 2 native children, /healthz ok."""
    cfg = make_config(tmp_path, broker, n_streams=2, per_worker=1,
                      backoff=0.5, device="vb-test-05")
    cfg.backend = {"name": "synthetic", "max_batch": 2, "model_w": 320,
                   "model_h": 320, "boxes": 2, "infer_ms": 0}
    for i, s in enumerate(cfg.streams):
        s["url"] = f"synthetic://?w=320&h=240&fps=15&boxes=2"
    # rewrite the config file so shard respawn would see the same streams
    argv = [os.environ["VB_RUNTIME_BIN"], "--ipc-fd", "{fd}"]
    sup = Supervisor(cfg, runtime_argv=argv)
    try:
        sup.start()
        port = sup.health_port

        def up():
            code, body = get_healthz(port)
            return (code == 200 and len(body["shards"]) == 2
                    and all(s["alive"] and s["runtime"]["alive"]
                            for s in body["shards"]))
        wait_for(up, timeout_s=30, what="2 native shards alive")
        _, body = get_healthz(port)
        jsonschema.validate(body, HEALTHZ_SCHEMA)
        assert all(sh["runtime"]["backend"] == "synthetic"
                   for sh in body["shards"])
        msg = broker.wait_publish("site/vision/vb-test-05/status")
        jsonschema.validate(json.loads(msg["payload"]), STATUS_SCHEMA)
        wait_for(lambda: any(
            m["topic"] == "site/vision/vb-test-05/results/cam-0"
            for m in broker.published), timeout_s=15, what="echo publications")
    finally:
        sup.stop()


# ------------------------------------------------------------- review items

BAD_APP = '''
class BadHealthApp:
    name = "bad"
    wants_frames = False
    def configure(self, options, device_id): pass
    def analyzers(self, spec): return []
    def plugins(self): return []
    def on_stream_added(self, ctx): pass
    def on_stream_removed(self, ctx): pass
    def on_event(self, ctx, ev): return []
    def on_frame(self, ctx, res): return []
    def health(self): return {"bad": {1, 2, 3}}       # not JSON serializable
'''

PLUGIN_APP = '''
class PluginApp:
    name = "plugin"
    wants_frames = False
    def configure(self, options, device_id): pass
    def analyzers(self, spec): return []
    def plugins(self): return ["/opt/vb/lib/app_extra.so"]
    def on_stream_added(self, ctx): pass
    def on_stream_removed(self, ctx): pass
    def on_event(self, ctx, ev): return []
    def on_frame(self, ctx, res): return []
'''

BAD_PLUGIN_APP = '''
class BadPluginApp:
    name = "badplugin"
    wants_frames = False
    def configure(self, options, device_id): pass
    def analyzers(self, spec): return []
    def plugins(self): return ["relative.so"]
    def on_stream_added(self, ctx): pass
    def on_stream_removed(self, ctx): pass
    def on_event(self, ctx, ev): return []
    def on_frame(self, ctx, res): return []
'''


def test_validate_app_health_boundaries():
    """Review item 14: only a JSON-serializable dict of <= 4096 bytes may ride
    the heartbeat; anything else becomes the specified error object."""
    from vision_base.supervisor import (APP_HEALTH_INVALID, APP_HEALTH_MAX_BYTES,
                                        validate_app_health)
    ok = "x" * (APP_HEALTH_MAX_BYTES - len('{"b":""}'))
    assert validate_app_health({"b": ok}) == {"b": ok}
    too_big = "x" * (APP_HEALTH_MAX_BYTES - len('{"b":""}') + 1)
    for bad in [None, [1], "s", {"s": {1, 2, 3}}, {"b": too_big},
                {"d": object()}]:
        assert validate_app_health(bad) == APP_HEALTH_INVALID


def test_bad_app_health_is_replaced_not_forwarded(tmp_path, broker):
    """Review item 14: an app returning an unserializable dict must not break
    the heartbeat or the /healthz serialization."""
    (tmp_path / "badhealth_app.py").write_text(BAD_APP)
    cfg = make_config(tmp_path, broker, n_streams=0, per_worker=0,
                      app_module="badhealth_app:BadHealthApp",
                      device="vb-test-14")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port
        # `alive` alone is true from the first heartbeat, before the runtime
        # and the publish session are up: wait for the 200
        wait_for(lambda: get_healthz(port)[0] == 200, timeout_s=30,
                 what="healthy supervisor")
        code, body = get_healthz(port)
        assert code == 200
        assert body["shards"][0]["app"] == {"error": "app health invalid"}
        assert body["app"] == {"error": "app health invalid"}
        jsonschema.validate(body, HEALTHZ_SCHEMA)
    finally:
        sup.stop()


def test_app_plugins_join_config_plugins(tmp_path, broker):
    """Review item 15: AppHooks.plugins() must reach the runtime config that
    the native child actually reads."""
    (tmp_path / "plugin_app.py").write_text(PLUGIN_APP)
    cfg = make_config(tmp_path, broker, n_streams=0, per_worker=0,
                      app_module="plugin_app:PluginApp",
                      plugins=["/opt/vb/lib/base.so"], device="vb-test-15")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    expect = ["/opt/vb/lib/base.so", "/opt/vb/lib/app_extra.so"]
    try:
        assert sup.cfg.analyzers["plugins"] == expect
        assert sup._shard_args(0, [])["runtime_cfg"]["analyzers"]["plugins"] == expect
        sup.start()
        port = sup.health_port
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["runtime"]["alive"],
                 timeout_s=30)
        written = json.loads(
            (tmp_path / "state" / "shard-0" / "rt-0.json").read_text())
        assert written["analyzers"]["plugins"] == expect
    finally:
        sup.stop()


def test_app_plugin_path_is_validated(tmp_path, broker):
    """Review item 15: merging is not enough — the merged list is checked the
    same way as analyzers.plugins."""
    from vision_base.config import ConfigError
    (tmp_path / "badplugin_app.py").write_text(BAD_PLUGIN_APP)
    cfg = make_config(tmp_path, broker, n_streams=0, per_worker=0,
                      app_module="badplugin_app:BadPluginApp",
                      device="vb-test-15b")
    with pytest.raises(ConfigError) as excinfo:
        Supervisor(cfg, runtime_argv=FAKE_ARGV)
    assert "app.plugins[0]" in str(excinfo.value)


def test_single_process_shard_is_supervised_from_startup(tmp_path, broker):
    """Review item 1: with workers=1 the shard is a supervisor thread; one that
    dies before sending `started` must be restarted, not ignored."""
    cfg = make_config(tmp_path, broker, n_streams=0, per_worker=0, workers=1,
                      backoff=0.4, device="vb-test-01b")
    state = tmp_path / "state" / "shard-0"
    state.mkdir(parents=True)
    victim = tmp_path / "victim.json"
    victim.write_text("untouched")
    (state / "rt-0.json").symlink_to(victim)     # every start attempt fails
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        wait_for(lambda: sup.shards and sup.shards[0].restarts >= 1,
                 timeout_s=25, what="dead shard thread restarted")
        assert sup.shards[0].in_process
        assert victim.read_text() == "untouched"
    finally:
        sup.stop()


def test_add_stream_while_runtime_is_restarting_is_acked_not_ok(tmp_path, broker):
    """Review item 8: Shard.add_stream answers ok:false while the runtime is
    restarting; that verdict has to reach the ack."""
    cfg = make_config(tmp_path, broker, n_streams=1, per_worker=0, backoff=5.0,
                      device="vb-test-08")
    root = "site/vision/vb-test-08"
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["runtime"]["alive"],
                 timeout_s=30)
        rt_pid = get_healthz(port)[1]["shards"][0]["runtime"]["pid"]
        kill_pid(rt_pid)
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["runtime"]["restarts"] == 1,
                 timeout_s=20, what="runtime restarts=1")
        # the shard now sits in its 5 s backoff with no runtime client
        send_command(broker, root, "vb-test-08", "r-add-restarting", "add_stream",
                     {"stream_id": "cam-9", "url": "fake://9"})
        ack = wait_ack(broker, f"{root}/cmd/ack", "r-add-restarting")
        assert ack["ok"] is False
        assert "restarting" in ack["error"]
        assert [s["stream_id"] for s in sup.shards[0].streams] == ["cam-0"]
    finally:
        sup.stop()


def test_new_shard_add_failure_is_acked_not_ok(tmp_path, broker, monkeypatch):
    """Review item 8: the supervisor used to report a spawned shard as a
    success without waiting for its add result."""
    monkeypatch.setenv("FAKE_REJECT_STREAM", "cam-9")
    cfg = make_config(tmp_path, broker, n_streams=1, per_worker=1, backoff=0.5,
                      device="vb-test-08b")
    root = "site/vision/vb-test-08b"
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["alive"], timeout_s=30)
        # per_worker=1 with one stream already: this needs a whole new shard
        send_command(broker, root, "vb-test-08b", "r-add-new", "add_stream",
                     {"stream_id": "cam-9", "url": "fake://9"})
        ack = wait_ack(broker, f"{root}/cmd/ack", "r-add-new")
        assert ack["ok"] is False
        assert "synthetic rejection" in ack["error"]
        assert len(sup.shards) == 1                 # the new shard was discarded
        assert [s["stream_id"] for s in sup.shards[0].streams] == ["cam-0"]
    finally:
        sup.stop()


def test_restart_reaps_a_stalled_shard(tmp_path, broker):
    """Review item 10: the previous shard (and its vb-runtime) must be
    reclaimed before its index is reused."""
    cfg = make_config(tmp_path, broker, n_streams=1, per_worker=1, backoff=0.5,
                      device="vb-test-10")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    old_pid = None
    try:
        sup.start()
        port = sup.health_port
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["runtime"]["alive"],
                 timeout_s=30)
        old_pid = get_healthz(port)[1]["shards"][0]["pid"]
        os.kill(old_pid, signal.SIGSTOP)     # alive, but no longer heartbeating

        def replaced():
            body = get_healthz(port)[1]
            return (body["shards"][0]["pid"] not in (None, old_pid)
                    and body["shards"][0]["alive"])
        wait_for(replaced, timeout_s=30, what="replacement shard alive")
        wait_for(lambda: not pid_alive(old_pid), timeout_s=15,
                 what="stalled shard reclaimed")
    finally:
        if old_pid:
            for sig in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.kill(old_pid, sig)
                except OSError:
                    pass
        sup.stop()


def test_restart_uses_the_current_stream_list(tmp_path, broker, monkeypatch):
    """Review item 12: a stream removed while the restart backs off must not be
    replayed by the new instance."""
    cfg = make_config(tmp_path, broker, n_streams=1, per_worker=1, backoff=1.0,
                      device="vb-test-12")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["runtime"]["alive"],
                 timeout_s=30)
        handle = sup.shards[0]
        entered = threading.Event()
        release = threading.Event()
        orig_reclaim = sup._reclaim_shard

        def reclaim(h):
            orig_reclaim(h)
            entered.set()
            release.wait(10.0)      # hold the restart thread while the test
                                    # removes the stream, as a control op would
        monkeypatch.setattr(sup, "_reclaim_shard", reclaim)
        t = threading.Thread(target=sup._restart_shard, args=(handle,))
        t.start()
        assert entered.wait(10.0), "restart never reached the backoff"
        with sup._mu:
            handle.streams = []
        release.set()
        t.join(15.0)
        assert not t.is_alive()

        new = sup.shards[0]
        assert new is not handle and new.streams == []
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["alive"], timeout_s=20)
        assert get_healthz(port)[1]["streams"] == []
    finally:
        sup.stop()


def test_runtime_failure_marks_healthz_degraded(tmp_path, broker, monkeypatch):
    """Review item 16: three pre-hello exits while the Python shard and the
    MQTT session stay healthy must still report degraded."""
    monkeypatch.setenv("FAKE_EXIT_BEFORE_HELLO", "1")
    cfg = make_config(tmp_path, broker, n_streams=1, per_worker=0, backoff=0.3,
                      device="vb-test-16")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port
        def gave_up():
            body = get_healthz(port)[1]
            # "error" is delivered by the shard's hook thread after its own
            # backoff, so runtime.failed alone is not enough to read /healthz
            states = {s["stream_id"]: s["state"] for s in body["streams"]}
            return (body["shards"][0]["runtime"]["failed"]
                    and states == {"cam-0": "error"})
        wait_for(gave_up, timeout_s=30, what="runtime gave up, streams errored")
        code, body = get_healthz(port)
        assert code == 503 and body["status"] == "degraded"
        assert body["shards"][0]["alive"] is True
        assert body["mqtt"]["control_connected"] is True
        assert {s["stream_id"]: s["state"] for s in body["streams"]} == \
            {"cam-0": "error"}
        jsonschema.validate(body, HEALTHZ_SCHEMA)
    finally:
        sup.stop()


def test_healthz_control_connected_follows_the_live_session(tmp_path, broker):
    """Review item 16: the health report used to keep the value sampled at
    start() even after the control session dropped."""
    cfg = make_config(tmp_path, broker, n_streams=0, per_worker=0,
                      device="vb-test-16b")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    assert sup.healthz_snapshot()["mqtt"]["control_connected"] is False
    sup.start()
    try:
        wait_for(lambda: sup.healthz_snapshot()["mqtt"]["control_connected"],
                 timeout_s=15, what="control session up")
        sup._control.close()
        assert sup.healthz_snapshot()["mqtt"]["control_connected"] is False
    finally:
        sup.stop()


def test_add_stream_ack_redacts_credentials(tmp_path, broker):
    """Review item 21: the ack leaves the device on an unauthenticated topic,
    so the stream URL is echoed with its credentials masked."""
    cfg = make_config(tmp_path, broker, n_streams=1, per_worker=0, backoff=0.5,
                      device="vb-test-21")
    root = "site/vision/vb-test-21"
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    try:
        sup.start()
        port = sup.health_port
        wait_for(lambda: get_healthz(port)[1]["shards"][0]["alive"], timeout_s=30)
        send_command(broker, root, "vb-test-21", "r-cred", "add_stream",
                     {"stream_id": "cam-2",
                      "url": "rtsp://admin:s3cret@cam.local/s1?token=abc"})
        ack = wait_ack(broker, f"{root}/cmd/ack", "r-cred")
        assert ack["ok"] is True
        url = ack["applied"]["url"]
        assert "s3cret" not in url and "token=abc" not in url
        assert "***" in url
        # the stored stream keeps the real URL: the runtime needs it
        assert any("s3cret" in s["url"] for h in sup.shards for s in h.streams)
    finally:
        sup.stop()


def test_control_queue_is_bounded(tmp_path, broker):
    """Review item 22: the cmd/control queue needs a message count, a byte
    budget and a per-message cap before it takes a slot."""
    from vision_base.supervisor import (CMD_MESSAGE_MAX_BYTES,
                                        CMD_QUEUE_MAX_MESSAGES)
    cfg = make_config(tmp_path, broker, n_streams=0, per_worker=0,
                      device="vb-test-22")
    sup = Supervisor(cfg, runtime_argv=FAKE_ARGV)
    payload = b'{"schema":"vb.command/1"}'
    for _ in range(CMD_QUEUE_MAX_MESSAGES * 3):
        sup._on_control_message("t", payload, False)
    with sup._cmd_cv:
        assert len(sup._cmd_q) == CMD_QUEUE_MAX_MESSAGES
        assert sup._cmd_q_bytes == CMD_QUEUE_MAX_MESSAGES * len(payload)
    assert sup._cmd_dropped == CMD_QUEUE_MAX_MESSAGES * 2

    before = sup._cmd_dropped
    sup._on_control_message("t", b"x" * (CMD_MESSAGE_MAX_BYTES + 1), False)
    sup._on_control_message("t", b"not json", False)
    sup._on_control_message("t", b"", False)
    sup._on_control_message("t", b'["array"]', False)
    assert sup._cmd_dropped == before + 4
    with sup._cmd_cv:
        assert len(sup._cmd_q) == CMD_QUEUE_MAX_MESSAGES

    # retained payloads are ignored (broker replay), and draining frees bytes
    sup._on_control_message("t", payload, True)
    assert sup._cmd_dropped == before + 4
    with sup._cmd_cv:
        raw = sup._cmd_q.pop(0)
        sup._cmd_q_bytes -= len(raw)
    assert sup._cmd_q_bytes == (CMD_QUEUE_MAX_MESSAGES - 1) * len(payload)
