"""Business hook protocol + EchoApp reference implementation (spec BASE-1
§6.5, M1.12). StreamContext is created by the base per stream; per-stream app
state lives in ``ctx.state`` and is never shared across streams.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .runtime_client import RuntimeGone, RuntimeError_, RuntimeClient
from .types import Event, FrameResult, StreamSpec

__all__ = ["Outgoing", "ConfigureError", "StreamContext", "AppHooks", "EchoApp"]


@dataclass
class Outgoing:
    topic_suffix: str   # relative to mqtt.topic_root, e.g. f"results/{stream_id}"
    payload: dict | bytes
    qos: int = 0
    retain: bool = False


class ConfigureError(Exception):
    """configure_analyzer failed on the native side."""


class StreamContext:
    """Per-stream context owned by the shard (spec §6.5)."""

    def __init__(self, index: int, spec: StreamSpec, runtime: RuntimeClient,
                 attr_names: tuple[str, ...] = ()):
        self.index = index
        self.stream_id = spec.stream_id
        self.spec = spec
        self.state: dict = {}
        self.attr_names = attr_names
        self.last_state: str | None = None
        self._runtime = runtime
        self._analyzers: dict[str, dict] = {}   # latest successful configs

    def request_snapshot(self, seq: int = 0, track_id: int = 0, crop: bool = True,
                         max_side: int = 640, timeout_s: float = 3.0) -> tuple[dict, bytes]:
        return self._runtime.snapshot(self.index, seq=seq, track_id=track_id,
                                      crop=crop, max_side=max_side,
                                      timeout_s=timeout_s)

    def configure_analyzer(self, name: str, config: dict,
                           timeout_s: float = 3.0) -> dict:
        try:
            reply = self._runtime.request("configure_analyzer", timeout_s,
                                          stream_index=self.index, name=name,
                                          config=config)
        except RuntimeError_ as e:
            raise ConfigureError(str(e)) from e
        except RuntimeGone as e:
            raise ConfigureError("runtime restarting") from e
        except TimeoutError:
            raise
        self._analyzers[name] = config
        return reply.get("applied", reply)

    def analyzer_configs(self, hooks: "AppHooks") -> list[dict]:
        """Configs to (re)send on add: latest configure_analyzer wins (§6.5.3)."""
        if self._analyzers:
            return [{"name": n, "config": c} for n, c in self._analyzers.items()]
        return list(hooks.analyzers(self.spec))


@runtime_checkable
class AppHooks(Protocol):
    name: str
    wants_frames: bool

    def analyzers(self, spec: StreamSpec) -> list[dict]: ...
    def plugins(self) -> list[str]: ...
    def on_stream_added(self, ctx: StreamContext) -> None: ...
    def on_frame(self, ctx: StreamContext, res: FrameResult) -> list[Outgoing]: ...
    def on_event(self, ctx: StreamContext, ev: Event) -> list[Outgoing]: ...
    def on_stream_removed(self, ctx: StreamContext) -> None: ...
    # Optional (probed with getattr): on_stream_state, health.


class EchoApp:
    """Reference app (§6.5): republishes the latest frame result at
    ``publish_hz`` to ``results/<stream_id>``.

    Instantiation rule (M1.20): ``cls()`` then, if defined,
    ``configure(options, device_id)``; ``publish_hz`` / ``frame_stride`` come
    from ``options``.
    """

    name = "echo"
    wants_frames = True

    def __init__(self) -> None:
        self.options: dict = {}
        self.device_id = ""

    def configure(self, options: dict, device_id: str) -> None:
        self.options = dict(options or {})
        self.device_id = device_id

    @property
    def publish_hz(self) -> float:
        try:
            hz = float(self.options.get("publish_hz", 1.0))
        except (TypeError, ValueError):
            hz = 1.0
        return hz if hz > 0 else 1.0

    @property
    def frame_stride(self) -> int:
        try:
            s = int(self.options.get("frame_stride", 1))
        except (TypeError, ValueError):
            s = 1
        return max(1, s)

    def analyzers(self, spec: StreamSpec) -> list[dict]:
        return []

    def plugins(self) -> list[str]:
        return []

    def on_stream_added(self, ctx: StreamContext) -> None:
        pass

    def on_stream_removed(self, ctx: StreamContext) -> None:
        pass

    def on_frame(self, ctx: StreamContext, res: FrameResult) -> list[Outgoing]:
        ctx.state["last_frame"] = res
        now = time.monotonic()
        interval = 1.0 / self.publish_hz
        last = ctx.state.get("last_publish")
        if last is not None and now - last < interval:
            return []
        ctx.state["last_publish"] = now
        payload = {
            "stream_id": ctx.stream_id,
            "seq": res.seq,
            "wall_ms": res.wall_ms,
            "inference_ms": res.inference_ms,
            "queue_delay_ms": res.queue_delay_ms,
            "detections": [
                {"cx": d.cx, "cy": d.cy, "w": d.w, "h": d.h, "score": d.score,
                 "class_id": d.class_id, "track_id": d.track_id}
                for d in res.detections
            ],
        }
        return [Outgoing(f"results/{ctx.stream_id}", payload)]

    def on_event(self, ctx: StreamContext, ev: Event) -> list[Outgoing]:
        return []
