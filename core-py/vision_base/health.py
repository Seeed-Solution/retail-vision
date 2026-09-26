"""`/healthz` server and snapshot builder (spec BASE-1 §5.4 `/healthz` row,
§6.8). A `ThreadingHTTPServer` in the supervisor process aggregates the
per-second shard heartbeats (§6.9). Standard library only.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, procstat

__all__ = ["build_healthz", "HealthServer"]


def build_healthz(*, device_id: str, backend: str, uptime_s: float,
                  control_connected: bool,
                  supervisor_pid: int, shards: list[dict],
                  streams: list[dict]) -> dict:
    """Build the §6.8 body from supervisor state.

    ``shards`` entries already carry per-shard heartbeat data
    (pid/alive/restarts/rss_kb/cpu_s/runtime/mqtt/app); ``streams`` holds the
    merged per-stream status entries.
    """
    sup = procstat.sample(supervisor_pid)
    rss_total = 0
    publish_connected: list[bool] = []
    queue_depth: list[int] = []
    dropped: list[int] = []
    app_parts: list[dict] = []
    for sh in shards:
        if sh.get("rss_kb") is not None:
            rss_total += sh["rss_kb"]
        rt = sh.get("runtime") or {}
        if rt.get("rss_kb") is not None:
            rss_total += rt["rss_kb"]
        mqtt = sh.get("mqtt") or {}
        publish_connected.append(bool(mqtt.get("connected")))
        queue_depth.append(int(mqtt.get("queue_depth", 0)))
        dropped.append(int(mqtt.get("dropped", 0)))
        if isinstance(sh.get("app"), dict):
            app_parts.append(sh["app"])
    if sup["rss_kb"] is not None:
        rss_total += sup["rss_kb"]

    body = {
        "status": "ok",
        "device_id": device_id,
        "backend": backend,
        "base_version": __version__,
        "uptime_s": round(uptime_s, 3),
        "mqtt": {"control_connected": control_connected,
                 "publish_connected": publish_connected,
                 "queue_depth": queue_depth, "dropped": dropped},
        "supervisor": {"pid": supervisor_pid, "rss_kb": sup["rss_kb"],
                       "cpu_s": sup["cpu_s"]},
        "shards": [{
            "index": sh["index"], "pid": sh["pid"], "alive": sh["alive"],
            "restarts": sh["restarts"], "rss_kb": sh.get("rss_kb"),
            "cpu_s": sh.get("cpu_s"),
            "contexts": int(sh.get("contexts", 1)),
            "streams": list(sh.get("stream_ids", [])),
            "runtime": {
                "pid": rt.get("pid"), "alive": bool(rt.get("alive")),
                "restarts": int(rt.get("restarts", 0)),
                "rss_kb": rt.get("rss_kb"), "cpu_s": rt.get("cpu_s"),
                "version": rt.get("version", ""),
                "backend": rt.get("backend", "")},
            **({"app": sh["app"]} if isinstance(sh.get("app"), dict) else {}),
            **({"hook_budget": sh["hook_budget"]}
               if isinstance(sh.get("hook_budget"), dict) else {}),
        } for sh in shards for rt in [sh.get("runtime") or {}]],
        "rss_kb_total": rss_total,
        "streams": streams,
    }
    merged = _merge_app(app_parts)
    if merged is not None:
        body["app"] = merged

    healthy = bool(shards) and all(sh["alive"] for sh in shards) \
        and all(publish_connected) and control_connected
    if not healthy:
        body["status"] = "degraded"
    return body


def _merge_app(parts: list[dict]) -> dict | None:
    """§6.8 `app` merge: max for `*_ts` numbers, or for bools, sum for other
    numbers, else the lowest-index shard's value; first `error` wins."""
    if not parts:
        return None
    if len(parts) == 1:
        return dict(parts[0])
    out: dict = {}
    keys: list[str] = []
    for p in parts:
        for k in p:
            if k not in keys:
                keys.append(k)
    for k in keys:
        vals = [p[k] for p in parts if k in p]
        if k == "error":
            out[k] = vals[0]
        elif k.endswith("_ts") and all(isinstance(v, (int, float))
                                       and not isinstance(v, bool) for v in vals):
            out[k] = max(vals)
        elif all(isinstance(v, bool) for v in vals):
            out[k] = any(vals)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool)
                 for v in vals):
            out[k] = sum(vals)
        else:
            out[k] = vals[0]
    return out


class HealthServer:
    """GET /healthz -> 200 (ok) or 503 (degraded); body from ``snapshot_fn``."""

    def __init__(self, host: str, port: int, snapshot_fn):
        self.snapshot_fn = snapshot_fn
        self.port = port
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path.split("?")[0] != "/healthz":
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    body = server.snapshot_fn()
                    code = 200 if body.get("status") == "ok" else 503
                    data = json.dumps(body, separators=(",", ":")).encode()
                except Exception as exc:  # never crash the server
                    code = 503
                    data = json.dumps({"status": "degraded",
                                       "error": f"{type(exc).__name__}: {exc}"}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # silence
                pass

        self._srv = ThreadingHTTPServer((host, port), Handler)
        if port == 0:
            self.port = self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        name="vb-healthz", daemon=True)

    @property
    def actual_port(self) -> int:
        return self.port

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=3.0)
