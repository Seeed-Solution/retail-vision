"""Shard: one Python shard owning one vb-runtime child (spec BASE-1 §5.2.3,
§6.5.1 hook thread, §6.9 rule 2 restart supervision). Standard library only.

M1.12 scope: RuntimeClient + per-stream StreamContext + AppHooks dispatch +
PublishWorker-style publisher + crash restart. Supervisor IPC (§6.9 pipes,
heartbeat, status) arrives in M1.13.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque

from .hooks import AppHooks, StreamContext
from .runtime_client import RuntimeClient, RuntimeGone
from .types import FrameResult, StreamSpec

__all__ = ["Shard"]

log = logging.getLogger("vb.shard")

HOOK_BUDGET_WINDOW_S = 10.0
HOOK_BUDGET_DEFAULT_CORE = 0.10
HOOK_BUDGET_ADVICE = ("per-frame hook exceeds budget; move it to a C ABI "
                      "analyzer plugin (docs/extending.md)")


class Shard:
    def __init__(self, index: int, hooks: AppHooks, publisher, *,
                 runtime_argv: list[str], runtime_cfg: dict,
                 state_dir: str, topic_root: str = "",
                 restart_backoff_s: float = 5.0,
                 hello_timeout_s: float = 10.0,
                 open_timeout_s: float = 8.0,
                 max_start_failures: int = 3):
        self.index = index
        self.hooks = hooks
        self.publisher = publisher          # PublishWorker-like: submit(topic, payload, qos, retain)
        self.runtime_argv = list(runtime_argv)
        self.runtime_cfg = runtime_cfg
        self.state_dir = state_dir
        self.topic_root = topic_root.rstrip("/")
        self.restart_backoff_s = restart_backoff_s
        self.hello_timeout_s = hello_timeout_s
        self.open_timeout_s = open_timeout_s
        self.max_start_failures = max_start_failures

        self.contexts: dict[int, StreamContext] = {}   # stream_index -> ctx (indices never reused)
        self._by_stream_id: dict[str, int] = {}
        self._next_index = 0
        self._active: set[int] = set()

        self.last_stats: dict = {}
        self.runtime_restarts = 0
        self.hook_frames_dropped: dict[int, int] = {}
        self.hello = None
        self.last_error = ""

        self._client: RuntimeClient | None = None
        self._cfg_path = ""
        self._mu = threading.Lock()          # client lifecycle transitions
        self._stopping = False
        self._start_failures = 0
        self._restart_thread: threading.Thread | None = None
        self._restart_scheduled = False

        # hook thread state (§6.5.1): unbounded FIFO for events/states/lifecycle,
        # length-1 slot per stream for frames (new overwrites old).
        self._fifo: deque = deque()
        self._frame_slots: dict[int, FrameResult] = {}
        self._frame_seen: dict[int, int] = {}

        # hook CPU metering (§6.5.5 / §10.5④): (end_monotonic, method, dur)
        # samples from the hook thread's time.thread_time().
        self._hook_samples: deque = deque()
        self._hook_budget_exceeded = False
        self._hook_warn_monotonic = 0.0
        self._cv = threading.Condition()
        self._hook_thread: threading.Thread | None = None
        self._hook_stop = False

    # ------------------------------------------------------------------ start

    def start(self) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        self._cfg_path = os.path.join(self.state_dir, f"rt-{self.index}.json")
        tmp = self._cfg_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.runtime_cfg, f)
        os.replace(tmp, self._cfg_path)      # atomic write (§6.9 rule 4)
        self._hook_thread = threading.Thread(target=self._hook_loop,
                                             name=f"vb-hook-{self.index}",
                                             daemon=True)
        self._hook_thread.start()
        self._start_runtime()

    def stop(self, timeout_s: float = 5.0) -> None:
        with self._mu:
            self._stopping = True
            client = self._client
        if client is not None:
            client.stop(timeout_s)
        with self._cv:
            self._hook_stop = True
            self._cv.notify_all()
        if self._hook_thread is not None:
            self._hook_thread.join(timeout_s)

    # ------------------------------------------------------------ runtime mgmt

    def _start_runtime(self) -> None:
        """Spawn vb-runtime and re-add all active streams (ascending index)."""
        with self._mu:
            if self._stopping:
                return
        client = RuntimeClient(self.runtime_argv, self._cfg_path,
                               on_frame=self._on_frame,
                               on_event=self._on_event,
                               on_stats=self._on_stats,
                               on_state=self._on_state,
                               on_exit=self._on_exit)
        try:
            hello = client.start(self.hello_timeout_s)
        except (TimeoutError, RuntimeGone):
            # Pre-hello failures are counted and reported only here; the
            # client's on_exit fires only after hello (M1.12 review 2026-09-26).
            client.kill()
            self._handle_start_failure()
            return
        with self._mu:
            self._client = client
            self._start_failures = 0
        self.hello = hello
        for idx in sorted(i for i in self._active):
            ctx = self.contexts.get(idx)
            if ctx is None:
                continue
            ctx._runtime = client     # snapshot/configure go to the live client
            self._send_add(client, ctx, first=False)

    def _send_add(self, client: RuntimeClient, ctx: StreamContext, *, first: bool) -> None:
        spec = ctx.spec
        stream = {"index": ctx.index, "id": spec.stream_id, "url": spec.url,
                  "name": spec.name, "transport": spec.transport,
                  "score_threshold": spec.score_threshold,
                  "options": spec.options}
        analyzers = ctx.analyzer_configs(self.hooks)
        client.request("add", self.open_timeout_s, stream=stream,
                       analyzers=analyzers)
        if first:
            with self._cv:
                self._fifo.append(("added", ctx.index))
                self._cv.notify()

    def _handle_start_failure(self) -> None:
        with self._mu:
            if self._stopping:
                return
            self._start_failures += 1
            active = sorted(self._active)
        with self._cv:
            for idx in active:
                self._fifo.append(("state", idx, "reconnecting", "runtime exited before hello"))
            self._cv.notify()
        with self._mu:
            if self._start_failures >= self.max_start_failures:
                # §6.9 rule 2: 3 pre-hello exits -> all streams error, stop restarting.
                self._schedule_error_all()
                return
        self._schedule_restart()

    def _schedule_restart(self) -> None:
        with self._mu:
            if self._stopping or self._restart_scheduled:
                return
            self._restart_scheduled = True
            self._restart_thread = threading.Thread(
                target=self._restart_after_backoff, daemon=True,
                name=f"vb-restart-{self.index}")
            self._restart_thread.start()

    def _schedule_error_all(self) -> None:
        threading.Thread(target=self._deliver_error_all, daemon=True).start()

    def _restart_after_backoff(self) -> None:
        time.sleep(self.restart_backoff_s)
        with self._mu:
            if self._stopping:
                return
            self._restart_scheduled = False
        self._start_runtime()

    def _deliver_error_all(self) -> None:
        time.sleep(self.restart_backoff_s)
        with self._cv:
            for idx in sorted(self._active):
                self._fifo.append(("state", idx, "error", "runtime failed to start"))
            self._cv.notify()

    # --------------------------------------------------- runtime client events

    def _on_frame(self, idx: int, res: FrameResult) -> None:
        ctx = self.contexts.get(idx)
        if ctx is None:
            return
        res.stream_id = ctx.stream_id
        with self._cv:
            if idx in self._frame_slots:
                self.hook_frames_dropped[idx] = self.hook_frames_dropped.get(idx, 0) + 1
            self._frame_slots[idx] = res
            self._cv.notify()

    def _on_event(self, idx: int, ev) -> None:
        ctx = self.contexts.get(idx)
        if ctx is None:
            return
        ev.stream_id = ctx.stream_id
        with self._cv:
            self._fifo.append(("event", idx, ev))
            self._cv.notify()

    def _on_stats(self, stats: dict) -> None:
        self.last_stats = stats

    def _on_state(self, idx: int, state: str, error: str) -> None:
        with self._cv:
            self._fifo.append(("state", idx, state, error))
            self._cv.notify()

    def _on_exit(self, code: int) -> None:
        with self._mu:
            if self._stopping:
                return
            self._client = None
            self.runtime_restarts += 1
            had_hello = self.hello is not None
            if not had_hello:
                self._start_failures += 1
        with self._cv:
            for idx in sorted(self._active):
                self._fifo.append(("state", idx, "reconnecting", f"runtime exit {code}"))
            self._cv.notify()
        with self._mu:
            if not had_hello and self._start_failures >= self.max_start_failures:
                self._schedule_error_all()
                return
        self._schedule_restart()

    # ------------------------------------------------------------- stream ops

    def add_stream(self, spec: StreamSpec) -> dict:
        """Add one stream; returns the native reply (ok:true)."""
        with self._mu:
            if spec.stream_id in self._by_stream_id:
                raise ValueError(f"duplicate stream_id {spec.stream_id!r}")
            client = self._client
            idx = self._next_index
            self._next_index += 1
            ctx = StreamContext(idx, spec, client,
                                self.hello.attr_names if self.hello else ())
            self.contexts[idx] = ctx
            self._by_stream_id[spec.stream_id] = idx
            self._active.add(idx)
        if client is None:
            # runtime down (restarting or failed to start): register the
            # stream; the restart path re-adds it once hello arrives.
            self._schedule_restart()
            return {"ok": False, "error": "runtime restarting",
                    "stream_index": idx}
        self._send_add(client, ctx, first=True)
        return {"ok": True, "stream_index": idx}

    def remove_stream(self, stream_id: str) -> dict:
        idx = self._by_stream_id.get(stream_id)
        if idx is None:
            raise KeyError(f"unknown stream {stream_id!r}")
        client = self._client
        if client is not None:
            client.request("remove", self.open_timeout_s, stream_index=idx)
        self._active.discard(idx)
        with self._cv:
            self._fifo.append(("state", idx, "stopped", ""))
            self._fifo.append(("removed", idx))
            self._cv.notify()
        return {"ok": True}

    def set_threshold(self, stream_id: str, value: float) -> dict:
        idx = self._by_stream_id.get(stream_id)
        if idx is None:
            raise KeyError(f"unknown stream {stream_id!r}")
        client = self._client
        if client is None:
            raise RuntimeGone("runtime not running")
        return client.request("set_threshold", self.open_timeout_s,
                              stream_index=idx, value=value)

    # ------------------------------------------------------------- hook thread

    def _hook_loop(self) -> None:
        while True:
            with self._cv:
                while (not self._fifo and not self._frame_slots
                       and not self._hook_stop):
                    self._cv.wait(0.5)
                if self._hook_stop and not self._fifo:
                    return
                items = []
                while self._fifo:
                    items.append(self._fifo.popleft())
                frames = dict(self._frame_slots)
                self._frame_slots.clear()
            for item in items:
                self._run_hook_item(item)
            if self.hooks.wants_frames:
                stride = int(getattr(self.hooks, "frame_stride", 1) or 1)
                for idx in sorted(frames):
                    seen = self._frame_seen.get(idx, 0) + 1
                    self._frame_seen[idx] = seen
                    if (seen - 1) % max(1, stride) == 0:
                        self._deliver_frame(self.contexts.get(idx), frames[idx])

    def _run_hook_item(self, item) -> None:
        kind = item[0]
        ctx = self.contexts.get(item[1])
        if ctx is None:
            return
        if kind == "state":
            self._deliver_state(ctx, item[2], item[3])
        elif kind == "added":
            self._call_hook(self.hooks.on_stream_added, ctx)
        elif kind == "removed":
            self._call_hook(self.hooks.on_stream_removed, ctx)
        elif kind == "event":
            outs = self._call_hook(self.hooks.on_event, ctx, item[2]) or []
            for o in outs:
                self._publish(ctx, o)

    def _deliver_frame(self, ctx, res) -> None:
        if ctx is None:
            return
        outs = self._call_hook(self.hooks.on_frame, ctx, res) or []
        for o in outs:
            self._publish(ctx, o)

    def _deliver_state(self, ctx: StreamContext, state: str, error: str) -> None:
        """§6.5.2 deliver(): only on state change; error text alone doesn't count."""
        if state == ctx.last_state:
            return
        ctx.last_state = state
        if hasattr(self.hooks, "on_stream_state"):
            outs = self._call_hook(self.hooks.on_stream_state, ctx, state, error) or []
            for o in outs:
                self._publish(ctx, o)

    def _publish(self, ctx, outgoing) -> None:
        topic = (self.topic_root + "/" + outgoing.topic_suffix) if self.topic_root \
            else outgoing.topic_suffix
        self.publisher.submit(topic, outgoing.payload, outgoing.qos,
                              outgoing.retain)

    def _call_hook(self, fn, *args):
        name = getattr(fn, "__name__", "")
        method = name if name in ("on_frame", "on_event") else "other"
        t0 = time.thread_time()
        try:
            return fn(*args)
        except Exception:
            return None
        finally:
            dur = time.thread_time() - t0
            now = time.monotonic()
            with self._cv:
                self._hook_samples.append((now, method, dur))
                # prune samples older than the window (+ slack)
                while self._hook_samples and \
                        now - self._hook_samples[0][0] > HOOK_BUDGET_WINDOW_S + 1.0:
                    self._hook_samples.popleft()

    # -------------------------------------------------------- hook budget

    def _hook_budget_core(self) -> float:
        try:
            opts = getattr(self.hooks, "options", None) or {}
            value = float(opts.get("hook_budget_core", HOOK_BUDGET_DEFAULT_CORE))
        except (TypeError, ValueError):
            value = HOOK_BUDGET_DEFAULT_CORE
        return value if value > 0 else HOOK_BUDGET_DEFAULT_CORE

    def hook_budget_status(self) -> dict:
        """§6.5.5: 10 s rolling window of hook-thread CPU (thread_time)."""
        budget = self._hook_budget_core()
        now = time.monotonic()
        per = {"on_frame": 0.0, "on_event": 0.0, "other": 0.0}
        span = 0.0
        with self._cv:
            samples = list(self._hook_samples)
        samples = [s for s in samples if now - s[0] <= HOOK_BUDGET_WINDOW_S]
        if samples:
            span = min(HOOK_BUDGET_WINDOW_S, now - samples[0][0])
            for _t, method, dur in samples:
                per[method if method in per else "other"] += dur
        core = (sum(per.values()) / span) if span > 0 else 0.0
        exceeded = core > budget
        if exceeded and not self._hook_budget_exceeded and \
                now - self._hook_warn_monotonic >= 60.0:
            log.warning("shard %s hook budget exceeded: core=%.3f limit=%.3f",
                        self.index, core, budget)
            self._hook_warn_monotonic = now
        self._hook_budget_exceeded = exceeded
        return {
            "core": round(core, 6), "limit": budget, "exceeded": exceeded,
            "by_method": {k: round(v, 6) for k, v in per.items()},
            "advice": HOOK_BUDGET_ADVICE if exceeded else "",
        }

    # ------------------------------------------------------------------ status

    def runtime_status(self) -> dict:
        """Native child process status for heartbeat/healthz (§6.9)."""
        from . import procstat
        client = self._client
        alive = client is not None and client.proc is not None \
            and client.proc.poll() is None
        pid = client.pid if client is not None else None
        st = procstat.sample(pid) if pid else {"rss_kb": None, "cpu_s": None}
        return {"pid": pid, "alive": bool(alive),
                "restarts": self.runtime_restarts,
                "rss_kb": st["rss_kb"], "cpu_s": st["cpu_s"],
                "version": self.hello.runtime_version if self.hello else "",
                "backend": self.hello.backend if self.hello else ""}

    def status_streams(self) -> list[dict]:
        """Full per-stream status entries (vb.status/1 stream items)."""
        native: dict[int, dict] = {}
        if isinstance(self.last_stats, dict):
            for s in self.last_stats.get("streams") or []:
                try:
                    native[int(s.get("stream_index", -1))] = s
                except (TypeError, ValueError):
                    pass
        out = []
        for idx in sorted(self._active):
            ctx = self.contexts.get(idx)
            if ctx is None:
                continue
            st = native.get(idx, {})
            out.append({
                "stream_id": ctx.stream_id,
                "name": ctx.spec.name,
                "state": ctx.last_state or "starting",
                "fps": float(st.get("fps", 0.0) or 0.0),
                "decode": str(st.get("decode", "") or ""),
                "fallback_active": bool(st.get("fallback_active", False)),
                "score_threshold": ctx.spec.score_threshold,
                "shard": self.index,
                "hook_frames_dropped": self.hook_frames_dropped.get(idx, 0),
            })
        return out

    def stream_status(self) -> list[dict]:
        out = []
        for idx in sorted(self._active):
            ctx = self.contexts.get(idx)
            if ctx is None:
                continue
            out.append({
                "stream_id": ctx.stream_id,
                "state": ctx.last_state or "starting",
                "hook_frames_dropped": self.hook_frames_dropped.get(idx, 0),
            })
        return out
