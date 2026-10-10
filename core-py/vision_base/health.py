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

# The endpoint is unauthenticated, so it is bounded on both axes (review
# item 20): a request may not hold a thread open indefinitely, and the number
# of concurrent connections is capped. Both are per-supervisor-process limits;
# the binding address — not these — is what keeps the endpoint off the
# network by default (`health.host`, config.py).
REQUEST_TIMEOUT_S = 5.0
MAX_CONNECTIONS = 32


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
    app_error_parts: list[dict] = []
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
        if isinstance(sh.get("app_error"), dict):
            app_error_parts.append(sh["app_error"])
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
                "backend": rt.get("backend", ""),
                **({key: rt[key] for key in
                    ("effective_contexts", "stats_monotonic_ms",
                     "stats_wall_ms", "stats_pid") if key in rt}),
                # §6.9 rule 2: the shard gave up restarting its vb-runtime.
                # The Python shard is still alive, so without this the report
                # says "ok" while no stream can run (review item 16).
                "failed": bool(rt.get("failed"))},
            **({"app": sh["app"]} if isinstance(sh.get("app"), dict) else {}),
            **({"app_error": sh["app_error"]}
               if isinstance(sh.get("app_error"), dict) else {}),
            **({"hook_budget": sh["hook_budget"]}
               if isinstance(sh.get("hook_budget"), dict) else {}),
        } for sh in shards for rt in [sh.get("runtime") or {}]],
        "rss_kb_total": rss_total,
        "streams": streams,
    }
    merged = _merge_app(app_parts)
    if merged is not None:
        body["app"] = merged
    merged_app_error = _merge_app(app_error_parts)
    if merged_app_error is not None:
        body["app_error"] = merged_app_error

    runtime_failed = any((sh.get("runtime") or {}).get("failed")
                         for sh in shards)
    healthy = bool(shards) and all(sh["alive"] for sh in shards) \
        and all(publish_connected) and control_connected \
        and not runtime_failed and not app_error_parts
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


class _BoundedHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` with a cap on live connections.

    One thread (and one file descriptor) per connection is fine when the
    clients are probes; an unauthenticated endpoint must not let a handful of
    half-open connections exhaust the supervisor's threads and FDs (review
    item 20). Connections beyond the cap are shed with 503.
    """

    daemon_threads = True
    request_queue_size = 16

    def __init__(self, addr, handler, *, max_connections: int = MAX_CONNECTIONS):
        super().__init__(addr, handler)
        self._slots = threading.BoundedSemaphore(max_connections)
        self.shed = 0        # connections refused over the cap
        self.served = 0      # connections that got a worker thread

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            self.shed += 1
            self._shed(request)
            return
        self.served += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def _shed(self, request) -> None:
        try:
            request.sendall(b"HTTP/1.1 503 Service Unavailable\r\n"
                            b"Content-Length: 0\r\nConnection: close\r\n\r\n")
        except OSError:
            pass
        self.shutdown_request(request)


class HealthServer:
    """GET /healthz -> 200 (ok) or 503 (degraded); body from ``snapshot_fn``."""

    def __init__(self, host: str, port: int, snapshot_fn,
                 *, max_connections: int = MAX_CONNECTIONS):
        self.snapshot_fn = snapshot_fn
        self.port = port
        server = self

        class Handler(BaseHTTPRequestHandler):
            # a slow or half-open client must not occupy a thread forever
            timeout = REQUEST_TIMEOUT_S

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

        self._srv = _BoundedHTTPServer((host, port), Handler,
                                       max_connections=max_connections)
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
