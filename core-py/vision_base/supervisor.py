"""Supervisor: process model, shard IPC and supervision (spec BASE-1 §5.4,
§6.7 status/control session, §6.9 IPC + restart rules, §6.8 /healthz).

One Python shard process per shard (``spawn`` start method, duplex Pipe),
each owning exactly one vb-runtime child; the supervisor alone owns the
LWT control session ``<client_id>-ctl`` and the /healthz server. Shards only
publish (client_id ``<client_id>-<i>``). Standard library only.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import multiprocessing
import os
import sys
import threading
import time

from . import __version__, procstat
from .config import BaseConfig, runtime_config
from .control import CommandError, ControlPlane
from .health import HealthServer, build_healthz
from .hooks import AppHooks
from .mqtt import MqttClient, Will
from .publish import PublishWorker
from .shard import Shard
from .types import StreamSpec

log = logging.getLogger("vision_base.supervisor")

__all__ = ["Supervisor", "load_app"]


# ----------------------------------------------------------------- app loader

def load_app(spec: str, extra_path: str = "") -> AppHooks:
    """§5.5.2 / M1.20 instantiation rule: ``"<module>:<class>"``, ``cls()``
    then ``configure(options, device_id)`` when defined."""
    mod_name, _, cls_name = spec.partition(":")
    if not mod_name or not cls_name:
        raise ValueError(f"app.module must be '<module>:<class>', got {spec!r}")
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls()


# ------------------------------------------------------------- shard process

def _jsonb(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def _shard_send(conn, obj: dict) -> None:
    try:
        conn.send(obj)
    except (OSError, ValueError, BrokenPipeError):
        pass


class ShardWorker:
    """Runs inside one shard process (or, when ``workers == 1``, in a
    supervisor thread): owns the publish MQTT session, the Shard and the
    supervisor Pipe protocol (§6.9)."""

    def __init__(self, args: dict):
        self.args = args

    def run(self, conn) -> None:
        a = self.args
        try:
            app = load_app(a["app_module"], a["config_dir"]) \
                if a.get("app_module") else None
            if app is None:
                from .hooks import EchoApp
                app = EchoApp()
            if hasattr(app, "configure"):
                app.configure(a.get("app_options") or {}, a.get("device_id", ""))
        except Exception:
            log.exception("app load failed in shard %s", a["index"])

        client = MqttClient(a["mqtt_host"], a["mqtt_port"],
                            client_id=a["client_id"],
                            username=a.get("mqtt_username", ""),
                            password=a.get("mqtt_password", ""),
                            tls=bool(a.get("mqtt_tls", False)),
                            ca_file=a.get("mqtt_ca_file", ""))
        client.connect(timeout_s=5.0)      # no LWT on publish sessions (§5.4)
        publisher = PublishWorker(client, queue_size=a.get("publish_queue", 256),
                                  name=f"vb-publish-{a['index']}")
        publisher.start()

        shard = Shard(a["index"], app, publisher,
                      runtime_argv=a["runtime_argv"],
                      runtime_cfg=a["runtime_cfg"], state_dir=a["state_dir"],
                      topic_root=a["topic_root"],
                      restart_backoff_s=a.get("restart_backoff_s", 5.0),
                      hello_timeout_s=a.get("hello_timeout_s", 10.0),
                      open_timeout_s=a.get("open_timeout_s", 8.0))
        shard.start()
        for s in a.get("streams", []):
            shard.add_stream(StreamSpec(**s))
        _shard_send(conn, {"op": "started", "index": a["index"],
                           "pid": os.getpid()})

        stop = threading.Event()

        def heartbeat():
            while not stop.wait(1.0):
                own = procstat.sample(os.getpid())
                hb = {"op": "heartbeat", "index": a["index"],
                      "pid": os.getpid(), "rss_kb": own["rss_kb"],
                      "cpu_s": own["cpu_s"],
                      "runtime": shard.runtime_status(),
                      "mqtt": {"connected": client.connected,
                               **publisher.stats()},
                      "contexts": a.get("contexts", 1),
                      "streams": shard.status_streams(),
                      "hook_budget": shard.hook_budget_status()}
                if shard.hello is not None:
                    hb["hello"] = {"backend": shard.hello.backend,
                                    "model_sha256": shard.hello.model_sha256,
                                    "runtime_version": shard.hello.runtime_version,
                                    "exclusive_device": bool(
                                        shard.hello.caps.get("exclusive_device"))}
                health_fn = getattr(app, "health", None)
                if callable(health_fn):
                    try:
                        hb["app"] = health_fn()
                    except Exception as exc:
                        hb["app"] = {"error": f"{type(exc).__name__}: {exc}"}
                _shard_send(conn, hb)

        threading.Thread(target=heartbeat, name=f"vb-hb-{a['index']}",
                         daemon=True).start()

        try:
            while not stop.is_set():
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                if not isinstance(msg, dict):
                    continue
                op = msg.get("op")
                if op == "stop":
                    break
                try:
                    if op == "add":
                        shard.add_stream(StreamSpec(**msg["stream"]))
                        reply = {"ok": True}
                    elif op == "remove":
                        shard.remove_stream(msg["stream_id"])
                        reply = {"ok": True}
                    elif op == "set_threshold":
                        shard.set_threshold(msg["stream_id"], msg["value"])
                        reply = {"ok": True}
                    else:
                        reply = {"ok": False, "error": f"unknown op {op}"}
                except Exception as exc:
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                _shard_send(conn, {"op": "reply", "req": msg.get("req", ""),
                                   **reply})
        finally:
            stop.set()
            shard.stop()
            publisher.close()
            client.close()


def shard_process_main(conn, args: dict) -> None:
    ShardWorker(args).run(conn)


# ----------------------------------------------------------------- supervisor

class _ShardHandle:
    def __init__(self, index: int):
        self.index = index
        self.proc = None            # multiprocessing.Process | threading.Thread
        self.conn = None            # supervisor end of the Pipe
        self.alive = False
        self.restarts = 0
        self.last_hb = 0.0
        self.hb: dict = {}
        self.streams: list[dict] = []       # authoritative stream dicts
        self.in_process = False
        self.restarting = False


class Supervisor:
    """Owns shard processes, the control MQTT session, status publishing,
    /healthz and the §6.9 restart rules."""

    def __init__(self, cfg: BaseConfig, *, runtime_argv: list[str] | None = None,
                 heartbeat_timeout_s: float = 5.0):
        self.cfg = cfg
        self.workers = cfg.runtime["workers"]
        self.per_worker = cfg.runtime["max_streams_per_worker"]
        self.backoff_s = float(cfg.runtime["restart_backoff_s"])
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.state_dir = cfg.state_dir
        self.topic_root = cfg.mqtt["topic_root"].rstrip("/")
        self.client_id = cfg.mqtt.get("client_id") or cfg.device_id
        self.session_id = str(int(time.time() * 1000))
        self.runtime_argv = runtime_argv or [cfg.native["binary"],
                                             "--ipc-fd", "{fd}"]
        self.hello: dict = {}          # first shard hello (backend/version)
        self.shards: list[_ShardHandle] = []
        self._mu = threading.Lock()
        self._req_cv = threading.Condition()
        self._pending: dict[str, dict] = {}
        self._req_counter = 0
        self._control: MqttClient | None = None
        self._control_ok = False
        self._plane = ControlPlane(cfg.device_id)
        self._plane_ops = _ControlOps(self)
        self._health: HealthServer | None = None
        self._status_thread = threading.Thread(target=self._status_loop,
                                               name="vb-status", daemon=True)
        self._watchdog = threading.Thread(target=self._watchdog_loop,
                                          name="vb-watchdog", daemon=True)
        self._cmd_q: list[bytes] = []
        self._cmd_cv = threading.Condition()
        self._stop = threading.Event()
        self._t0 = time.monotonic()

    # ---------------------------------------------------------------- helpers

    def _shard_count_for(self, n_streams: int) -> int:
        if self.per_worker <= 0:
            desired = 1
        else:
            desired = max(1, math.ceil(n_streams / self.per_worker))
        if isinstance(self.workers, int):
            desired = min(desired, self.workers)
        return desired

    def _shard_args(self, index: int, streams: list[dict]) -> dict:
        mqtt = self.cfg.mqtt
        return {
            "index": index, "client_id": f"{self.client_id}-{index}",
            "mqtt_host": mqtt["host"], "mqtt_port": mqtt["port"],
            "mqtt_username": mqtt.get("username", ""),
            "mqtt_password": mqtt.get("password", ""),
            "mqtt_tls": mqtt.get("tls", False),
            "mqtt_ca_file": mqtt.get("ca_file", ""),
            "publish_queue": self.cfg.runtime["publish_queue"],
            "runtime_argv": list(self.runtime_argv),
            "runtime_cfg": runtime_config(self.cfg, index),
            "state_dir": os.path.join(self.state_dir, f"shard-{index}"),
            "topic_root": self.topic_root,
            "restart_backoff_s": self.backoff_s,
            "hello_timeout_s": self.cfg.native["hello_timeout_s"],
            "open_timeout_s": self.cfg.runtime["open_timeout_s"],
            "contexts": self.cfg.runtime["contexts_per_worker"],
            "streams": streams,
            "app_module": self.cfg.app.get("module", ""),
            "config_dir": os.path.dirname(os.path.abspath(
                self.cfg.source_path)) if self.cfg.source_path else "",
            "app_options": self.cfg.app.get("options", {}),
            "device_id": self.cfg.device_id,
        }

    def _spawn_shard(self, index: int, streams: list[dict]) -> _ShardHandle:
        handle = _ShardHandle(index)
        handle.streams = [dict(s) for s in streams]
        parent, child = multiprocessing.get_context("spawn").Pipe(duplex=True)
        args = self._shard_args(index, handle.streams)
        in_process = isinstance(self.workers, int) and self.workers == 1
        handle.in_process = in_process
        if in_process:
            handle.proc = threading.Thread(
                target=shard_process_main, args=(child, args),
                name=f"vb-shard-{index}", daemon=True)
        else:
            handle.proc = multiprocessing.get_context("spawn").Process(
                target=shard_process_main, args=(child, args),
                name=f"vb-shard-{index}", daemon=True)
        handle.conn = parent
        handle.proc.start()
        threading.Thread(target=self._reader_loop, args=(handle,),
                         name=f"vb-ipc-{index}", daemon=True).start()
        return handle

    def _reader_loop(self, handle: _ShardHandle) -> None:
        conn = handle.conn
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            if not isinstance(msg, dict):
                continue
            op = msg.get("op")
            if op == "heartbeat" or op == "started":
                with self._mu:
                    handle.last_hb = time.monotonic()
                    handle.hb = msg
                    handle.alive = True
                    hl = msg.get("hello")
                    if hl and not self.hello:
                        self.hello = dict(hl)
            elif op == "reply":
                with self._req_cv:
                    self._pending[msg.get("req", "")] = msg
                    self._req_cv.notify_all()

    def _send_op(self, handle: _ShardHandle, msg: dict, timeout_s: float) -> dict:
        with self._req_cv:
            self._req_counter += 1
            req = f"s-{self._req_counter}"
            msg["req"] = req
        try:
            handle.conn.send(msg)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": f"shard unreachable: {exc}"}
        deadline = time.monotonic() + timeout_s
        with self._req_cv:
            while req not in self._pending:
                left = deadline - time.monotonic()
                if left <= 0:
                    return {"ok": False, "error": "shard request timed out"}
                self._req_cv.wait(left)
            reply = self._pending.pop(req)
        return reply

    # ------------------------------------------------------------- supervision

    def _watchdog_loop(self) -> None:
        while not self._stop.wait(0.5):
            now = time.monotonic()
            for handle in list(self.shards):
                if handle.restarting:
                    continue
                exited = (not handle.in_process
                          and handle.proc is not None
                          and handle.proc.exitcode is not None)
                timed_out = now - handle.last_hb > self.heartbeat_timeout_s
                if handle.last_hb == 0.0:
                    timed_out = now - self._t0 > max(self.heartbeat_timeout_s, 15.0)
                if exited or (handle.alive and timed_out):
                    with self._mu:
                        handle.alive = False
                        handle.restarts += 1
                        handle.restarting = True
                    log.warning("shard %d died (exit=%s); restarting",
                                handle.index,
                                handle.proc.exitcode if not handle.in_process
                                else "?")
                    threading.Thread(target=self._restart_shard,
                                     args=(handle,), daemon=True).start()

    def _restart_shard(self, handle: _ShardHandle) -> None:
        streams = list(handle.streams)
        if handle.in_process:
            try:
                handle.conn.send({"op": "stop"})
            except Exception:
                pass
        time.sleep(self.backoff_s)
        with self._mu:
            old_conn = handle.conn
        try:
            old_conn.close()
        except Exception:
            pass
        new = self._spawn_shard(handle.index, streams)
        new.restarts = handle.restarts      # restart count survives respawns
        with self._mu:
            try:
                pos = self.shards.index(handle)
                self.shards[pos] = new     # reader thread updates `new`
            except ValueError:
                self.shards.append(new)
        self._publish_status()

    # ------------------------------------------------------------ control ops

    def _pick_shard_for_add(self) -> _ShardHandle | None:
        """Fewest streams, still below max_streams_per_worker; None = full."""
        with self._mu:
            live = [h for h in self.shards if not h.restarting]
        if self.per_worker <= 0:
            return live[0] if live else None
        candidates = [h for h in live if len(h.streams) < self.per_worker]
        if not candidates:
            return None
        return min(candidates, key=lambda h: len(h.streams))

    def ops_add_stream(self, params: dict) -> dict:
        stream = {"stream_id": params["stream_id"], "url": params["url"],
                  "name": params.get("name", ""),
                  "transport": params.get("transport", "tcp"),
                  "score_threshold": params.get("score_threshold"),
                  "options": params.get("options", {})}
        with self._mu:
            total = sum(len(h.streams) for h in self.shards)
        if total >= self.cfg.runtime["max_streams"]:
            raise CommandError(f"at most {self.cfg.runtime['max_streams']} streams")
        handle = self._pick_shard_for_add()
        if handle is None:
            if self.hello.get("exclusive_device"):
                raise CommandError("exclusive device allows a single shard")
            if isinstance(self.workers, int) and len(self.shards) >= self.workers:
                raise CommandError(f"workers limit {self.workers} reached")
            index = len(self.shards)
            with self._mu:
                handle = self._spawn_shard(index, [stream])
                self.shards.append(handle)
            persisted = self._persist_streams()
            self._publish_status()
            return {"stream_id": stream["stream_id"], "url": stream["url"],
                    "shard": index,
                    "score_threshold": stream["score_threshold"],
                    "persisted": persisted}
        reply = self._send_op(handle, {"op": "add", "stream": stream},
                              self.cfg.runtime["open_timeout_s"])
        if not reply.get("ok"):
            raise CommandError(str(reply.get("error", "add failed")))
        with self._mu:
            handle.streams.append(stream)
        persisted = self._persist_streams()
        self._publish_status()
        return {"stream_id": stream["stream_id"], "url": stream["url"],
                "shard": handle.index, "score_threshold": stream["score_threshold"],
                "persisted": persisted}

    def ops_remove_stream(self, params: dict) -> dict:
        sid = params["stream_id"]
        with self._mu:
            handle = next((h for h in self.shards
                           if any(s["stream_id"] == sid for s in h.streams)), None)
        if handle is None:
            raise CommandError(f"unknown stream {sid}")
        reply = self._send_op(handle, {"op": "remove", "stream_id": sid},
                              self.cfg.runtime["open_timeout_s"])
        if not reply.get("ok"):
            raise CommandError(str(reply.get("error", "remove failed")))
        with self._mu:
            handle.streams = [s for s in handle.streams if s["stream_id"] != sid]
        persisted = self._persist_streams()
        self._publish_status()
        return {"stream_id": sid, "removed": True, "persisted": persisted}

    def ops_set_threshold(self, params: dict) -> dict:
        sid = params["stream_id"]
        with self._mu:
            handle = next((h for h in self.shards
                           if any(s["stream_id"] == sid for s in h.streams)), None)
        if handle is None:
            raise CommandError(f"unknown stream {sid}")
        reply = self._send_op(handle, {"op": "set_threshold", "stream_id": sid,
                                       "value": params["score_threshold"]},
                              self.cfg.runtime["open_timeout_s"])
        if not reply.get("ok"):
            raise CommandError(str(reply.get("error", "set_threshold failed")))
        with self._mu:
            for s in handle.streams:
                if s["stream_id"] == sid:
                    s["score_threshold"] = params["score_threshold"]
        return {"stream_id": sid, "score_threshold": params["score_threshold"]}

    def ops_list_streams(self, params: dict) -> dict:
        return {"streams": self._all_stream_status()}

    def _persist_streams(self) -> bool:
        path = self.cfg.streams_file
        if not path:
            return False
        with self._mu:
            streams = [s for h in self.shards for s in h.streams]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(streams, f)
        os.replace(tmp, path)
        return True

    def _all_stream_status(self) -> list[dict]:
        with self._mu:
            handles = list(self.shards)
        out: list[dict] = []
        for h in handles:
            hb = h.hb or {}
            for entry in hb.get("streams", []):
                out.append(dict(entry))
        # streams not yet in any heartbeat (just added / restarting)
        seen = {e["stream_id"] for e in out}
        for h in handles:
            for s in h.streams:
                if s["stream_id"] not in seen:
                    out.append({"stream_id": s["stream_id"],
                                "name": s.get("name", ""), "state": "starting",
                                "fps": 0.0, "decode": "",
                                "fallback_active": False,
                                "score_threshold": s.get("score_threshold"),
                                "shard": h.index, "hook_frames_dropped": 0})
        return out

    # ----------------------------------------------------------------- status

    def _status_payload(self, online: bool = True) -> dict:
        hello = self.hello or {}
        payload = {"schema": "vb.status/1",
                   "timestamp": int(time.time() * 1000),
                   "session_id": self.session_id,
                   "device_id": self.cfg.device_id, "online": online,
                   "versions": {"base": __version__,
                                "app": self.cfg.app.get("name", ""),
                                "backend": hello.get("backend", ""),
                                "model_sha256": hello.get("model_sha256", "")},
                   "uptime_s": round(time.monotonic() - self._t0, 3),
                   "shards": len(self.shards),
                   "streams": self._all_stream_status() if online else []}
        return payload

    def _publish_status(self) -> None:
        if self._control is None or not self._control.connected:
            return
        try:
            self._control.publish(f"{self.topic_root}/status",
                                  _jsonb(self._status_payload()),
                                  qos=1, retain=True)
        except Exception:
            log.exception("status publish failed")

    def _status_loop(self) -> None:
        interval = max(0.5, float(self.cfg.mqtt.get("status_interval_s", 10)))
        while not self._stop.wait(interval):
            self._publish_status()

    # ---------------------------------------------------------------- healthz

    def healthz_snapshot(self) -> dict:
        with self._mu:
            handles = list(self.shards)
        shards_body = []
        for h in handles:
            hb = dict(h.hb or {})
            rt = dict(hb.get("runtime") or {})
            shards_body.append({
                "index": h.index,
                "pid": hb.get("pid") if hb else (h.proc.pid if not h.in_process
                                                 and h.proc else None),
                "alive": h.alive, "restarts": h.restarts,
                "rss_kb": hb.get("rss_kb"), "cpu_s": hb.get("cpu_s"),
                "contexts": hb.get("contexts", self.cfg.runtime["contexts_per_worker"]),
                "stream_ids": [s["stream_id"] for s in h.streams],
                "runtime": rt, "mqtt": hb.get("mqtt") or {},
                "app": hb.get("app"),
                "hook_budget": hb.get("hook_budget"),
            })
        hello = self.hello or {}
        return build_healthz(
            device_id=self.cfg.device_id,
            backend=hello.get("backend", self.cfg.backend.get("name", "")),
            uptime_s=time.monotonic() - self._t0,
            control_connected=self._control_ok,
            supervisor_pid=os.getpid(), shards=shards_body,
            streams=self._all_stream_status())

    @property
    def health_port(self) -> int:
        return self._health.actual_port if self._health else 0

    # ------------------------------------------------------------- mqtt setup

    def _offline_payload(self) -> bytes:
        return _jsonb({"schema": "vb.status/1", "timestamp": int(time.time() * 1000),
                       "session_id": self.session_id,
                       "device_id": self.cfg.device_id, "online": False})

    def _on_control_message(self, topic: str, payload: bytes, retain: bool) -> None:
        if retain:
            return
        with self._cmd_cv:
            self._cmd_q.append(payload)
            self._cmd_cv.notify_all()

    def _cmd_loop(self) -> None:
        while not self._stop.is_set():
            with self._cmd_cv:
                while not self._cmd_q and not self._stop.is_set():
                    self._cmd_cv.wait(0.5)
                if self._stop.is_set():
                    return
                raw = self._cmd_q.pop(0)
            try:
                req = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            ack = self._plane.handle(req, self.session_id,
                                     int(time.time() * 1000), self._plane_ops)
            if ack is None:
                continue
            try:
                self._control.publish(f"{self.topic_root}/cmd/ack",
                                      _jsonb(ack), qos=1)
            except Exception:
                log.exception("ack publish failed")

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        mqtt = self.cfg.mqtt
        self._control = MqttClient(
            mqtt["host"], mqtt["port"], client_id=f"{self.client_id}-ctl",
            username=mqtt.get("username", ""), password=mqtt.get("password", ""),
            tls=bool(mqtt.get("tls", False)), ca_file=mqtt.get("ca_file", ""),
            keepalive_s=mqtt.get("keepalive_s", 30),
            will=Will(f"{self.topic_root}/status", self._offline_payload(),
                      qos=1, retain=True))
        self._control.connect(timeout_s=10.0)
        self._control_ok = self._control.connected
        self._control.subscribe(f"{self.topic_root}/cmd/control", 1,
                                self._on_control_message)
        threading.Thread(target=self._cmd_loop, name="vb-cmd",
                         daemon=True).start()
        self._watchdog.start()
        self._status_thread.start()

        n = len(self.cfg.streams)
        count = self._shard_count_for(n)
        with self._mu:
            for i in range(count):
                streams = [dict(s) for j, s in enumerate(self.cfg.streams)
                           if j % count == i]
                handle = self._spawn_shard(i, streams)
                self.shards.append(handle)

        self._health = HealthServer(self.cfg.health.get("host", "0.0.0.0"),
                                    int(self.cfg.health.get("port", 8099)),
                                    self.healthz_snapshot)
        self._health.start()

    def wait(self) -> None:
        """Block until stop() (used by main.py; SIGINT handler calls stop)."""
        while not self._stop.wait(0.5):
            pass

    def stop(self, timeout_s: float = 10.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        with self._cmd_cv:
            self._cmd_cv.notify_all()
        # graceful shard stop (they exit their run loop, stopping vb-runtime)
        with self._mu:
            handles = list(self.shards)
        for h in handles:
            try:
                h.conn.send({"op": "stop"})
            except Exception:
                pass
        deadline = time.monotonic() + 3.0
        for h in handles:
            if h.in_process or h.proc is None:
                continue
            h.proc.join(timeout=max(0.1, deadline - time.monotonic()))
            if h.proc.is_alive():
                h.proc.terminate()
                h.proc.join(timeout=3.0)
        for h in handles:
            try:
                h.conn.close()
            except Exception:
                pass
        # clean offline status (no LWT race: we still hold the session)
        if self._control is not None and self._control.connected:
            try:
                self._control.publish(f"{self.topic_root}/status",
                                      self._offline_payload(), qos=1,
                                      retain=True, timeout_s=2.0)
            except Exception:
                pass
            self._control.close()
        if self._health is not None:
            self._health.stop()


class _ControlOps:
    """Adapter object handed to ControlPlane.handle (§6.7 decision table)."""

    def __init__(self, supervisor: Supervisor):
        self.sup = supervisor

    def add_stream(self, params: dict) -> dict:
        return self.sup.ops_add_stream(params)

    def remove_stream(self, params: dict) -> dict:
        return self.sup.ops_remove_stream(params)

    def set_threshold(self, params: dict) -> dict:
        return self.sup.ops_set_threshold(params)

    def list_streams(self, params: dict) -> dict:
        return self.sup.ops_list_streams(params)
