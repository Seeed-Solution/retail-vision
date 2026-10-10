"""Embedded runtime client (spec BASE-1 §6.13.1, M1.21).

``Runtime`` runs one ``vb-runtime`` child directly — no supervisor, shards,
MQTT or ``/healthz``, no automatic restart. Results are consumed from a
single iterator: frame results live in a bounded queue (oldest dropped when
full), events are never dropped.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time

from . import config as _config
from .runtime_client import RuntimeClient, RuntimeGone
from .types import Event, FrameResult

__all__ = ["Runtime", "RuntimeGone"]

DEFAULT_BINARY = "/opt/vb/bin/vb-runtime"


class Runtime:
    def __init__(self, *, runtime_argv: list[str], runtime_cfg: dict,
                 cfg_path: str, queue_size: int, hello_timeout_s: float,
                 open_timeout_s: float):
        self._runtime_argv = list(runtime_argv)
        self._runtime_cfg = runtime_cfg
        self._cfg_path = cfg_path            # temp file, removed on stop()
        self._queue_size = max(1, int(queue_size))
        self._hello_timeout_s = hello_timeout_s
        self._open_timeout_s = open_timeout_s

        self._client: RuntimeClient | None = None
        self._exited = False
        self.frames_dropped = 0

        self._contexts: dict[str, int] = {}   # stream_id -> index (never reused)
        self._ids: dict[int, str] = {}
        self._next_index = 0
        self._analyzers: dict[int, list[dict]] = {}

        # results queue: ("frame", res) / ("event", ev); frames bounded,
        # oldest frame dropped when full; events unbounded.
        self._mu = threading.Lock()
        self._cv = threading.Condition(self._mu)
        self._items: list = []
        self._frame_count = 0

    # ------------------------------------------------------------- factory

    @classmethod
    def from_config(cls, path: str, *, binary: str | None = None,
                    queue_size: int = 1024) -> "Runtime":
        cfg = _config.load(path, partial=True)
        bin_path = binary or os.environ.get("VB_RUNTIME_BIN") \
            or cfg.native.get("binary") or DEFAULT_BINARY
        rt_cfg = _config.runtime_config(cfg, 0)
        fd, cfg_path = tempfile.mkstemp(prefix="vb-embed-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rt_cfg, f)
        return cls(runtime_argv=[bin_path, "--ipc-fd", "{fd}"],
                   runtime_cfg=rt_cfg, cfg_path=cfg_path,
                   queue_size=queue_size,
                   hello_timeout_s=float(cfg.native.get("hello_timeout_s", 10.0)),
                   open_timeout_s=float(cfg.runtime.get("open_timeout_s", 8.0)))

    # ----------------------------------------------------------- lifecycle

    def start(self):
        if self._client is not None:
            raise RuntimeError("already started")
        client = RuntimeClient(self._runtime_argv, self._cfg_path,
                               on_frame=self._on_frame,
                               on_event=self._on_event,
                               on_stats=lambda stats: None,
                               on_state=lambda i, s, e: None,
                               on_exit=self._on_exit)
        hello = client.start(self._hello_timeout_s)
        self._client = client
        return hello

    def stop(self) -> None:
        client = self._client
        if client is not None:
            client.stop()
        self._client = None
        self._exited = True
        with self._cv:
            self._cv.notify_all()
        try:
            os.unlink(self._cfg_path)
        except OSError:
            pass

    def __enter__(self) -> "Runtime":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -------------------------------------------------------------- streams

    def add_stream(self, stream_id: str, url: str, *,
                   analyzers: list[dict] | None = None,
                   options: dict | None = None, name: str = "",
                   transport: str = "tcp",
                   score_threshold: float | None = None,
                   timeout_s: float = 10.0) -> dict:
        client = self._require_client()
        if stream_id in self._contexts:
            raise ValueError(f"duplicate stream_id {stream_id!r}")
        opts = dict(options or {})
        analyzer_list = analyzers if analyzers is not None \
            else opts.get("analyzers", [])
        idx = self._next_index
        self._next_index += 1        # never reused, even if the add fails
        self._contexts[stream_id] = idx
        self._ids[idx] = stream_id
        self._analyzers[idx] = list(analyzer_list)
        try:
            reply = client.request("add", timeout_s,
                                   stream={"index": idx, "id": stream_id,
                                           "url": url, "name": name,
                                           "transport": transport,
                                           "score_threshold": score_threshold,
                                           "options": opts},
                                   analyzers=analyzer_list)
        except BaseException:
            # A failed add must not reserve the stream_id forever (report
            # item 7): roll the registration back before re-raising.
            self._forget(stream_id)
            raise
        return reply.get("applied", reply)

    def _forget(self, stream_id: str) -> None:
        """Drop the host-side mapping for one stream (no control line)."""
        idx = self._contexts.pop(stream_id, None)
        if idx is None:
            return
        self._ids.pop(idx, None)
        self._analyzers.pop(idx, None)

    def remove_stream(self, stream_id: str, timeout_s: float = 5.0) -> None:
        client = self._require_client()
        idx = self._contexts.get(stream_id)
        if idx is None:
            raise KeyError(f"unknown stream {stream_id!r}")
        client.request("remove", timeout_s, stream_index=idx)
        # Only a successful remove releases the mapping (report item 7); a
        # raised request leaves it in place so the caller can retry.
        self._forget(stream_id)

    def configure_analyzer(self, stream_id: str, name: str, config: dict,
                           timeout_s: float = 3.0) -> dict:
        client = self._require_client()
        idx = self._contexts.get(stream_id)
        if idx is None:
            raise KeyError(f"unknown stream {stream_id!r}")
        reply = client.request("configure_analyzer", timeout_s,
                               stream_index=idx, name=name, config=config)
        self._analyzers.setdefault(idx, []).append({"name": name,
                                                    "config": config})
        return reply.get("applied", reply)

    def snapshot(self, stream_id: str, seq: int = 0, track_id: int = 0,
                 max_side: int = 640) -> bytes:
        client = self._require_client()
        idx = self._contexts.get(stream_id)
        if idx is None:
            raise KeyError(f"unknown stream {stream_id!r}")
        _meta, jpeg = client.snapshot(idx, seq=seq, track_id=track_id,
                                      max_side=max_side)
        return jpeg

    # -------------------------------------------------------------- results

    def results(self, timeout_s: float | None = None):
        """Yield FrameResult / Event items; ends on idle timeout; raises
        RuntimeGone once the child process has exited (§6.13.1)."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            with self._cv:
                while not self._items and not self._exited:
                    remaining = None if deadline is None \
                        else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return
                    self._cv.wait(remaining if remaining is not None else 0.5)
                if self._items:
                    kind, item = self._items.pop(0)
                    if kind == "frame":
                        self._frame_count -= 1
                    # idle timeout: restart the window after every item
                    if timeout_s is not None:
                        deadline = time.monotonic() + timeout_s
                else:
                    if self._exited:
                        raise RuntimeGone("runtime process exited")
                    continue
            yield item

    # ------------------------------------------------------- client events

    def _on_frame(self, idx: int, res: FrameResult) -> None:
        res.stream_id = self._ids.get(idx, res.stream_id)
        with self._cv:
            if self._frame_count >= self._queue_size:
                for i, (k, _) in enumerate(self._items):
                    if k == "frame":
                        del self._items[i]          # drop oldest frame
                        self.frames_dropped += 1
                        self._frame_count -= 1
                        break
            self._items.append(("frame", res))
            self._frame_count += 1
            self._cv.notify()

    def _on_event(self, idx: int, ev: Event) -> None:
        ev.stream_id = self._ids.get(idx, ev.stream_id)
        with self._cv:
            self._items.append(("event", ev))
            self._cv.notify()

    def _on_exit(self, code: int) -> None:
        self._exited = True
        with self._cv:
            self._cv.notify_all()

    def _require_client(self) -> RuntimeClient:
        if self._client is None:
            raise RuntimeGone("runtime not started" if not self._exited
                              else "runtime process exited")
        return self._client
