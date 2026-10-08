from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from vision_base.health import HealthServer, build_healthz
from vision_base.shard import Shard
from vision_base.hooks import StreamContext
from vision_base.types import StreamSpec
import vision_base.supervisor as supervisor_module


class _DelayedStopConn:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        time.sleep(1.2)
        return {"op": "stop"}


def _worker_args():
    return {
        "index": 0, "app_module": "fake:App", "config_dir": "",
        "app_options": {}, "device_id": "test", "mqtt_host": "host",
        "mqtt_port": 1883, "client_id": "client", "runtime_argv": [],
        "runtime_cfg": {}, "state_dir": "/tmp/state", "topic_root": "root",
        "streams": [],
    }


def test_configure_error_is_reported_in_started_and_every_heartbeat(monkeypatch):
    class App:
        def configure(self, options, device_id):
            raise ValueError("site_id must be <= 32 bytes")

        def health(self):
            return {"healthy": True}

        def close(self):
            pass

    class Client:
        connected = True

        def __init__(self, *args, **kwargs):
            pass

        def connect(self, **kwargs):
            pass

        def close(self):
            pass

    class Publisher:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stats(self):
            return {"queue_depth": 0, "dropped": 0}

        def close(self):
            pass

    class Shard:
        hello = None

        def __init__(self, *args, **kwargs):
            self.last_stats = {}

        def start(self):
            pass

        def status_streams(self, stats=None):
            return []

        def runtime_status(self, stats=None):
            return {"pid": 1235, "alive": True, "restarts": 0,
                    "rss_kb": None, "cpu_s": None, "version": "0",
                    "backend": "synthetic", "failed": False}

        def hook_budget_status(self):
            return {}

        def stop(self):
            pass

    monkeypatch.setattr(supervisor_module, "load_app", lambda *args: App())
    monkeypatch.setattr(supervisor_module, "MqttClient", Client)
    monkeypatch.setattr(supervisor_module, "PublishWorker", Publisher)
    monkeypatch.setattr(supervisor_module, "Shard", Shard)

    conn = _DelayedStopConn()
    supervisor_module.ShardWorker(_worker_args()).run(conn)

    started = next(message for message in conn.sent if message["op"] == "started")
    heartbeats = [message for message in conn.sent
                  if message["op"] == "heartbeat"]
    assert heartbeats
    expected = {"error": "ValueError: site_id must be <= 32 bytes"}
    assert started["app_error"] == expected
    assert all(message["app_error"] == expected for message in heartbeats)
    assert all(message["app"] == {"healthy": True} for message in heartbeats)


def test_heartbeat_pairs_native_stats_snapshot_across_runtime_and_streams(monkeypatch):
    class TwoHeartbeatConn:
        def __init__(self):
            self.sent = []
            self.ready = threading.Event()

        def send(self, message):
            self.sent.append(message)
            if message.get("op") == "heartbeat" and len(
                    [item for item in self.sent if item.get("op") == "heartbeat"]
            ) == 2:
                self.ready.set()

        def recv(self):
            self.ready.wait(timeout=5.0)
            return {"op": "stop"}

    class App:
        def close(self):
            pass

    class Client:
        connected = True

        def __init__(self, *args, **kwargs):
            pass

        def connect(self, **kwargs):
            pass

        def close(self):
            pass

    class Publisher:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stats(self):
            return {"queue_depth": 0, "dropped": 0}

        def close(self):
            pass

    class PairingShard(Shard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._client = type("Client", (), {
                "pid": os.getpid(),
                "proc": type("Proc", (), {"poll": lambda self: None})(),
            })()
            self._on_stats({
                "stats_wall_ms": 101, "stats_pid": 111,
                "streams": [{"stream_index": 0, "processed_frames": 0}],
            })
            ctx = StreamContext(0, StreamSpec("cam-0", "fake://"), runtime=None)
            ctx.last_state = "running"
            self.contexts[0] = ctx
            self._active.add(0)
            self._stats_updates = 0

        def start(self):
            pass

        def runtime_status(self, stats=None):
            assert stats is self.last_stats
            out = super().runtime_status(stats)
            if self._stats_updates == 0:
                self._on_stats({
                    "stats_wall_ms": 202, "stats_pid": 222,
                    "streams": [{"stream_index": 0, "processed_frames": 7}],
                })
                self._stats_updates += 1
            return out

        def hook_budget_status(self):
            return super().hook_budget_status()

        def stop(self):
            pass

    monkeypatch.setattr(supervisor_module, "load_app", lambda *args: App())
    monkeypatch.setattr(supervisor_module, "MqttClient", Client)
    monkeypatch.setattr(supervisor_module, "PublishWorker", Publisher)
    monkeypatch.setattr(supervisor_module, "Shard", PairingShard)

    conn = TwoHeartbeatConn()
    supervisor_module.ShardWorker(_worker_args()).run(conn)
    heartbeats = [message for message in conn.sent
                  if message["op"] == "heartbeat"]
    assert len(heartbeats) == 2
    assert heartbeats[0]["runtime"]["pid"] == os.getpid()
    assert heartbeats[0]["runtime"]["pid"] != heartbeats[0]["runtime"]["stats_pid"]
    assert heartbeats[0]["runtime"]["stats_wall_ms"] == 101
    assert heartbeats[0]["runtime"]["stats_pid"] == 111
    assert heartbeats[0]["streams"][0]["processed_frames"] == 0
    assert heartbeats[1]["runtime"]["stats_wall_ms"] == 202
    assert heartbeats[1]["runtime"]["stats_pid"] == 222
    assert heartbeats[1]["streams"][0]["processed_frames"] == 7


def _healthy_shard(app_error=None):
    shard = {
        "index": 0, "pid": 1234, "alive": True, "restarts": 0,
        "rss_kb": None, "cpu_s": None, "contexts": 1,
        "stream_ids": [],
        "runtime": {"pid": 1235, "alive": True, "restarts": 0,
                    "rss_kb": None, "cpu_s": None, "version": "0",
                    "backend": "synthetic", "failed": False,
                    "effective_contexts": 2, "stats_monotonic_ms": 1234,
                    "stats_wall_ms": 1700000000123, "stats_pid": 1235},
        "mqtt": {"connected": True, "queue_depth": 0, "dropped": 0},
    }
    if app_error is not None:
        shard["app_error"] = app_error
    return shard


def test_application_error_degrades_healthz_and_returns_http_503():
    error = {"error": "ConfigError: site_id must be <= 32 bytes"}
    body = build_healthz(device_id="dev-1", backend="synthetic", uptime_s=1.0,
                         control_connected=True, supervisor_pid=os.getpid(),
                         shards=[_healthy_shard(error)], streams=[])
    assert body["status"] == "degraded"
    assert body["app_error"] == error
    assert body["shards"][0]["app_error"] == error

    server = HealthServer("127.0.0.1", 0, lambda: body)
    server.start()
    try:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{server.actual_port}/healthz")
        except urllib.error.HTTPError as response:
            assert response.code == 503
            assert json.loads(response.read())["app_error"] == error
        else:
            raise AssertionError("degraded healthz unexpectedly returned 200")
    finally:
        server.stop()


def test_configured_app_without_error_remains_healthy():
    body = build_healthz(device_id="dev-1", backend="synthetic", uptime_s=1.0,
                         control_connected=True, supervisor_pid=os.getpid(),
                         shards=[{**_healthy_shard(), "app": {"healthy": True}}],
                         streams=[])
    assert body["status"] == "ok"
    assert body["app"] == {"healthy": True}
    runtime = body["shards"][0]["runtime"]
    assert runtime["effective_contexts"] == 2
    assert runtime["stats_monotonic_ms"] == 1234
    assert runtime["stats_wall_ms"] == 1700000000123
    assert runtime["stats_pid"] == 1235
