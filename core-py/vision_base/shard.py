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

from . import atomicio
from .hooks import AppHooks, StreamContext
from .runtime_client import RuntimeClient, RuntimeGone
from .types import FrameResult, StreamSpec

__all__ = ["Shard"]

log = logging.getLogger("vb.shard")

HOOK_BUDGET_WINDOW_S = 10.0
HOOK_BUDGET_DEFAULT_CORE = 0.10
HOOK_BUDGET_ADVICE = ("per-frame hook exceeds budget; move it to a C ABI "
                      "analyzer plugin (docs/extending.md)")
HOOK_SLOW_S = 1.0   # §6.5.1: a single hook call longer than this is `hook_slow`


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
        # §6.9 rule 2: set when restarting the vb-runtime has been given up.
        # It rides the heartbeat into /healthz so the supervisor can report
        # `degraded` even though this Python shard is still alive (item 16).
        self.runtime_failed = False
        self.hook_frames_dropped: dict[int, int] = {}
        self.hello = None
        self.last_error = ""

        # §6.12 dev mode: VBT1 records are accepted only when the runtime was
        # started with --dev (main.py --dev) and the app defines on_tensors;
        # otherwise the client treats a VBT1 record as a protocol error.
        self.has_tensors_hook = ("--dev" in self.runtime_argv and
                                 getattr(hooks, "on_tensors", None) is not None)

        self._client: RuntimeClient | None = None
        self._starting: RuntimeClient | None = None   # hello in flight (item 11)
        self._start_lock = threading.Lock()  # one start/restart attempt at a time
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

        # hook CPU metering (§6.5.5 / §10.5④): CPU seconds accumulated per
        # method inside the current 10 s window, plus the slowest single call.
        self._hook_win_start = time.monotonic()
        self._hook_win_cpu = {"on_frame": 0.0, "on_event": 0.0, "other": 0.0}
        self.hook_slow = 0
        self._hook_budget_exceeded = False
        self._hook_warn_monotonic = 0.0
        self._cv = threading.Condition()
        self._hook_thread: threading.Thread | None = None
        self._hook_stop = False

    # ------------------------------------------------------------------ start

    def start(self) -> None:
        atomicio.ensure_private_dir(self.state_dir)
        self._cfg_path = os.path.join(self.state_dir, f"rt-{self.index}.json")
        # atomic + exclusive temp file (§6.9 rule 4, review item 23)
        atomicio.write_json_atomic(self._cfg_path, self.runtime_cfg)
        self._hook_thread = threading.Thread(target=self._hook_loop,
                                             name=f"vb-hook-{self.index}",
                                             daemon=True)
        self._hook_thread.start()
        self._start_runtime()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the runtime and the hook thread.

        A start attempt may be in flight (item 11): killing the client it is
        waiting on makes ``start()`` return at once, and the ``_start_lock``
        join guarantees no restart thread is still about to install one.
        """
        with self._mu:
            self._stopping = True
            client = self._client
            starting = self._starting
        if starting is not None:
            starting.kill()          # unblocks the in-flight hello wait
        if client is not None:
            client.stop(timeout_s)
        if self._start_lock.acquire(timeout=max(0.1, timeout_s)):
            self._start_lock.release()
        with self._mu:
            self._client = None
        with self._cv:
            self._hook_stop = True
            self._cv.notify_all()
        if self._hook_thread is not None:
            self._hook_thread.join(timeout_s)

    # ------------------------------------------------------------ runtime mgmt

    def _start_runtime(self) -> None:
        """Spawn vb-runtime and re-add all active streams (ascending index).

        Serialised by ``_start_lock`` so a watchdog restart and a control
        operation cannot both spawn a child, and so ``stop()`` can wait for
        the attempt to converge (report item 11).
        """
        with self._start_lock:
            with self._mu:
                if self._stopping:
                    return
                client = RuntimeClient(self.runtime_argv, self._cfg_path,
                                       on_frame=self._on_frame,
                                       on_event=self._on_event,
                                       on_stats=self._on_stats,
                                       on_state=self._on_state,
                                       on_exit=self._on_exit,
                                       on_tensors=self._on_tensors
                                       if self.has_tensors_hook else None)
                self._starting = client
            try:
                hello = client.start(self.hello_timeout_s)
            except (TimeoutError, RuntimeGone):
                # Pre-hello failures are counted and reported only here; the
                # client's on_exit fires only after hello (M1.12 review
                # 2026-09-26).
                client.kill()
                with self._mu:
                    if self._starting is client:
                        self._starting = None
                self._handle_start_failure()
                return
            with self._mu:
                stopping = self._stopping
                if self._starting is client:
                    self._starting = None
                if not stopping:
                    self._client = client
                    self._start_failures = 0
            if stopping:
                # stop() ran while hello was in flight: never install the
                # client and never replay the streams (report item 11).
                client.stop(timeout_s=2.0)
                return
            self.hello = hello
            self.runtime_failed = False   # recovered (§6.9 rule 2)
            for idx in sorted(i for i in self._active):
                ctx = self.contexts.get(idx)
                if ctx is None:
                    continue
                ctx._runtime = client  # snapshot/configure go to the live client
                try:
                    self._send_add(client, ctx, first=False)
                except RuntimeGone:
                    # The runtime died again mid-replay: leave the streams to
                    # the client's on_exit path (one uniform restart) instead
                    # of reporting per-stream errors on top of it.
                    log.warning("shard %s: replay stopped, runtime gone at "
                                "stream %s", self.index, idx)
                    return
                except Exception as exc:      # noqa: BLE001 - per stream
                    # One rejected stream must not abandon the rest (report
                    # item 2): report it and keep replaying the others.
                    log.warning("shard %s: replay failed for stream %s: %s",
                                self.index, idx, exc)
                    with self._cv:
                        self._fifo.append(("state", idx, "error",
                                           f"replay failed: {exc}"))
                        self._cv.notify()

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
        # §6.9 rule 2: restarting stopped for good; say so in every status
        # (report item 16 — the shard stays alive, so `alive` alone lies).
        self.runtime_failed = True
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

    def _on_tensors(self, idx: int, tf) -> None:
        """§6.12: fill stream_id, queue for the hook thread."""
        ctx = self.contexts.get(idx)
        if ctx is None:
            return
        tf.stream_id = ctx.stream_id
        with self._cv:
            self._fifo.append(("tensors", idx, tf))
            self._cv.notify()

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
        """Add one stream; returns ``{"ok": bool, ...}``.

        The stream is registered before the control line goes out — frames
        and states for that index must not be dropped while ``add`` is in
        flight — and the registration is rolled back when the native side
        rejects the add, so a failed add no longer leaks the stream_id
        forever (report item 7). The stream index itself is never reused.
        """
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
            # stream; the restart path re-adds it once hello arrives. The
            # caller must pass this ok:false through (report item 8).
            self._schedule_restart()
            return {"ok": False, "error": "runtime restarting",
                    "stream_index": idx}
        try:
            self._send_add(client, ctx, first=True)
        except BaseException:
            self._rollback_add(spec.stream_id, idx)
            raise
        return {"ok": True, "stream_index": idx}

    def _rollback_add(self, stream_id: str, idx: int) -> None:
        with self._mu:
            if self._by_stream_id.get(stream_id) == idx:
                del self._by_stream_id[stream_id]
            self.contexts.pop(idx, None)
            self._active.discard(idx)
        with self._cv:
            self._frame_slots.pop(idx, None)

    def remove_stream(self, stream_id: str) -> dict:
        idx = self._by_stream_id.get(stream_id)
        if idx is None:
            raise KeyError(f"unknown stream {stream_id!r}")
        client = self._client
        if client is not None:
            client.request("remove", self.open_timeout_s, stream_index=idx)
        # Stop reporting the stream immediately, but keep the context alive
        # until the hook thread has delivered "stopped" and called
        # on_stream_removed; the context travels with the queue item so the
        # stream_id is free for a re-add at once and the mapping cannot leak
        # (report item 7).
        ctx = self.contexts.get(idx)
        with self._mu:
            self._active.discard(idx)
            if self._by_stream_id.get(stream_id) == idx:
                del self._by_stream_id[stream_id]
        with self._cv:
            self._fifo.append(("state", idx, "stopped", ""))
            self._fifo.append(("removed", idx, ctx))
            self._cv.notify()
        return {"ok": True}

    def _forget(self, idx: int) -> None:
        """Drop every trace of a removed stream (report item 7)."""
        with self._mu:
            self.contexts.pop(idx, None)
            self._active.discard(idx)
        with self._cv:
            self._frame_slots.pop(idx, None)
            self._frame_seen.pop(idx, None)
            self.hook_frames_dropped.pop(idx, None)

    def set_threshold(self, stream_id: str, value: float) -> dict:
        idx = self._by_stream_id.get(stream_id)
        if idx is None:
            raise KeyError(f"unknown stream {stream_id!r}")
        client = self._client
        if client is None:
            raise RuntimeGone("runtime not running")
        reply = client.request("set_threshold", self.open_timeout_s,
                               stream_index=idx, value=value)
        # Keep the shard's own copy in step: it is what a vb-runtime restart
        # replays and what status reports (report item 6).
        ctx = self.contexts.get(idx)
        if ctx is not None:
            ctx.spec.score_threshold = value
        return reply

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
        if kind == "removed":
            ctx = item[2]            # carried by remove_stream: see _forget()
        else:
            ctx = self.contexts.get(item[1])
        if ctx is None:
            return
        if kind == "state":
            self._deliver_state(ctx, item[2], item[3])
        elif kind == "added":
            self._call_hook(self.hooks.on_stream_added, ctx)
        elif kind == "removed":
            self._call_hook(self.hooks.on_stream_removed, ctx)
            self._forget(item[1])
        elif kind == "event":
            outs = self._call_hook(self.hooks.on_event, ctx, item[2]) or []
            for o in outs:
                self._publish(ctx, o)
        elif kind == "tensors":
            fn = getattr(self.hooks, "on_tensors", None)
            if fn is not None:
                outs = self._call_hook(fn, ctx, item[2]) or []
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
            if dur > HOOK_SLOW_S:
                # §6.5.1: one call over a second is worth a warning and the
                # `hook_slow` count (report item 13).
                self.hook_slow += 1
                log.warning("shard %s: hook %s took %.3f s", self.index,
                            method, dur)
            now = time.monotonic()
            with self._cv:
                self._roll_hook_window_locked(now)
                self._hook_win_cpu[method] += dur

    # -------------------------------------------------------- hook budget

    def _roll_hook_window_locked(self, now: float) -> None:
        """Open a fresh 10 s window once the current one has run its course."""
        if now - self._hook_win_start < HOOK_BUDGET_WINDOW_S:
            return
        self._hook_win_start = now
        self._hook_win_cpu = {"on_frame": 0.0, "on_event": 0.0, "other": 0.0}

    def _hook_budget_core(self) -> float:
        try:
            opts = getattr(self.hooks, "options", None) or {}
            value = float(opts.get("hook_budget_core", HOOK_BUDGET_DEFAULT_CORE))
        except (TypeError, ValueError):
            value = HOOK_BUDGET_DEFAULT_CORE
        return value if value > 0 else HOOK_BUDGET_DEFAULT_CORE

    def hook_budget_status(self) -> dict:
        """§6.5.5: CPU of the hook thread accumulated over a 10 s window.

        The denominator is the window (``÷ 10``), never the age of the oldest
        sample: dividing by "time since the earliest sample" turned a single
        0.02 s call sampled 0.1 s later into 0.2 core instead of 0.002
        (report item 13).
        """
        budget = self._hook_budget_core()
        now = time.monotonic()
        with self._cv:
            self._roll_hook_window_locked(now)
            per = dict(self._hook_win_cpu)
        core = sum(per.values()) / HOOK_BUDGET_WINDOW_S
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
            "slow": self.hook_slow,
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
                "failed": bool(self.runtime_failed and not alive),
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
