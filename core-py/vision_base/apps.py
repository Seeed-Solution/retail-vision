"""Zero-code config-mode app (spec BASE-1 §5.5.2 / §6.10.2, M1.20).

``ConfigApp`` publishes the same ``vb.event/1`` / ``vb.frame/1`` payloads as
``vb-runtime --standalone`` (verified by the M1.20 cross-check); analyzers
come from the config file, no application code is required.

Instantiation rule (§6.5): ``cls()`` then ``configure(options, device_id)``.
"""
from __future__ import annotations

from .hooks import Outgoing, StreamContext
from .letterbox import box_to_source_norm, to_source_norm
from .types import Event, FrameResult, StreamSpec

__all__ = ["ConfigApp"]


def _num6(v: float) -> float:
    """§6.10.2 consistency: Python side rounds to 6 decimals to match the
    C++ ``%.6g`` JSON construction."""
    return round(float(v), 6)


def _round6(obj):
    if isinstance(obj, bool) or isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return _num6(obj)
    if isinstance(obj, dict):
        return {k: _round6(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round6(v) for v in obj]
    return obj


class ConfigApp:
    """§5.5.2: analyzers from ``streams[].options.analyzers``; default
    publishing of ``events/<stream_id>`` (qos 1) and, when
    ``app.options.frame_every > 0``, ``frames/<stream_id>`` (qos 0)."""

    name = "config"

    def __init__(self) -> None:
        self.options: dict = {}
        self.device_id = ""

    def configure(self, options: dict, device_id: str) -> None:
        self.options = dict(options or {})
        self.device_id = device_id

    @property
    def frame_every(self) -> int:
        try:
            n = int(self.options.get("frame_every", 0))
        except (TypeError, ValueError):
            n = 0
        return max(0, n)

    @property
    def frame_stride(self) -> int:
        # The shard samples frames by this stride (§6.5); ConfigApp then
        # publishes every delivered frame, i.e. one per frame_every.
        return max(1, self.frame_every)

    @property
    def wants_frames(self) -> bool:
        return self.frame_every > 0

    def analyzers(self, spec: StreamSpec) -> list[dict]:
        return list(spec.options.get("analyzers", []))

    def plugins(self) -> list[str]:
        return []

    def on_stream_added(self, ctx: StreamContext) -> None:
        pass

    def on_stream_removed(self, ctx: StreamContext) -> None:
        pass

    # -------------------------------------------------------------- payloads

    def event_payload(self, ev: Event) -> dict:
        """vb.event/1, field-for-field identical to the C++ JSONL output
        (§6.10.2) except ts_ms, which both sides take from wall clock."""
        return {
            "schema": "vb.event/1",
            "device_id": self.device_id,
            "stream_id": ev.stream_id,
            "seq": ev.seq,
            "ts_ms": int(round(ev.wall_ms)),
            "analyzer": ev.analyzer,
            "type": ev.type,
            "track_id": ev.track_id,
            "fields": _round6(ev.fields),
        }

    def frame_payload(self, res: FrameResult, attr_names: tuple[str, ...] = ()) -> dict:
        """vb.frame/1, mirroring src/out/event_json.cpp frame_json()."""
        geom = res.geom
        dets = []
        for d in res.detections:
            scx, scy, sw, sh = box_to_source_norm(geom, d.cx, d.cy, d.w, d.h)
            dj = {
                "track_id": d.track_id,
                "class_id": d.class_id,
                "score": _num6(d.score),
                "box": [_num6(scx - sw / 2.0), _num6(scy - sh / 2.0),
                        _num6(scx + sw / 2.0), _num6(scy + sh / 2.0)],
            }
            if d.keypoints:
                kps = []
                for i in range(0, len(d.keypoints) - 2, 3):
                    sx, sy = to_source_norm(geom, d.keypoints[i], d.keypoints[i + 1])
                    kps.append([_num6(sx), _num6(sy), _num6(d.keypoints[i + 2])])
                dj["keypoints"] = kps
            if d.attrs:
                names = attr_names or tuple(f"attr_{i}" for i in range(len(d.attrs)))
                dj["attrs"] = {names[i]: _num6(v)
                               for i, v in enumerate(d.attrs) if i < len(names)}
            dets.append(dj)
        return {
            "schema": "vb.frame/1",
            "device_id": self.device_id,
            "stream_id": res.stream_id,
            "seq": res.seq,
            "ts_ms": int(round(res.wall_ms)),
            "inference_ms": _num6(res.inference_ms),
            "detections": dets,
        }

    # --------------------------------------------------------------- hooks

    def on_event(self, ctx: StreamContext, ev: Event) -> list[Outgoing]:
        return [Outgoing(f"events/{ctx.stream_id}", self.event_payload(ev), qos=1)]

    def on_frame(self, ctx: StreamContext, res: FrameResult) -> list[Outgoing]:
        return [Outgoing(f"frames/{ctx.stream_id}",
                         self.frame_payload(res, getattr(ctx, "attr_names", ())),
                         qos=0)]
