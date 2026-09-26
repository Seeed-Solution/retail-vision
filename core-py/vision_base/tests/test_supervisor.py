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
                backoff=5.0, device="vb-test-01"):
    streams = [{"stream_id": f"cam-{i}", "url": f"fake://{i}"}
               for i in range(n_streams)]
    cfg = {
        "schema": "vb.config/1",
        "device_id": device,
        "backend": {"name": "fake", "model_path": "/tmp/fake.onnx",
                    "input_size": [320, 320]},
        "runtime": {"workers": "auto", "max_streams_per_worker": per_worker,
                    "restart_backoff_s": backoff, "open_timeout_s": 5.0},
        "native": {"binary": sys.executable, "hello_timeout_s": 10.0},
        "state_dir": str(tmp_path / "state"),
        "mqtt": {"host": broker.host, "port": broker.port,
                 "topic_root": f"site/vision/{device}",
                 "status_interval_s": 1.0},
        "health": {"host": "127.0.0.1", "port": free_port()},
        "app": {"module": "vision_base.hooks:EchoApp"},
        "streams": streams,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return load(str(path))


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
