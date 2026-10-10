"""Pure-dataclass types for the vision base (spec BASE-1 §6.4).

No numpy; coordinates are plain floats.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LetterboxGeom:
    src_w: int
    src_h: int
    model_w: int
    model_h: int
    scale: float
    pad_x: float
    pad_y: float
    align: str  # "center" | "top_left"


@dataclass(slots=True)
class Detection:
    cx: float  # model-canvas normalized [0, 1]
    cy: float
    w: float
    h: float
    score: float
    class_id: int
    track_id: int  # 0 = unassociated
    keypoints: tuple[float, ...] = ()  # flat x, y, conf × K; model-canvas normalized
    attrs: tuple[float, ...] = ()  # order matches Hello.attr_names


@dataclass(slots=True)
class FrameResult:
    stream_id: str
    seq: int
    wall_ms: float
    geom: LetterboxGeom
    detections: list[Detection]
    inference_ms: float
    queue_delay_ms: float


@dataclass(frozen=True, slots=True)
class RawTensor:
    """One raw output tensor, dev mode only (spec BASE-1 §6.12)."""
    name: str
    dtype: int            # 0 = f32 little-endian; 1-255 reserved
    dims: tuple[int, ...]
    scale: float          # dequant scale; 1.0 for f32
    zero_point: int
    nhwc: bool            # True = channel-last layout
    data: bytes


@dataclass(slots=False)  # dev-mode frames carry a transient stream_index
class TensorFrame:
    """Dev-mode raw tensor frame (VBT1), spec BASE-1 §6.12."""
    stream_id: str
    seq: int
    wall_ms: float
    geom: LetterboxGeom
    tensors: list[RawTensor]


@dataclass(slots=True)
class Event:
    stream_id: str
    seq: int
    wall_ms: float
    analyzer: str
    type: str
    track_id: int
    fields: dict


@dataclass
class StreamSpec:
    stream_id: str
    url: str
    name: str = ""
    transport: str = "tcp"
    score_threshold: float | None = None
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Hello:
    runtime_version: str
    abi: int
    backend: str
    caps: dict
    model_hw: tuple[int, int]
    model_sha256: str
    attr_names: tuple[str, ...]
    pid: int
