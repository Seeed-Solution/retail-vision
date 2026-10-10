"""Wire record decoding for VBR1/VBE1/VBC1/VBS1 (spec BASE-1 §6.3).

Standard library only; decoding uses ``struct.unpack_from`` only (no
per-byte loops over buffers — spec §10.5①).
"""
from __future__ import annotations

import json
import struct
from typing import Any

from .types import LetterboxGeom, RawTensor, TensorFrame

__all__ = ["WireError", "HEADER", "MAGIC_FRAME", "MAGIC_EVENT", "MAGIC_CONTROL",
           "MAGIC_SNAPSHOT", "MAGIC_TENSOR", "decode_header", "decode_result",
           "decode_event", "decode_snapshot", "decode_tensors",
           "encode_control_line"]


class WireError(Exception):
    """Malformed or truncated wire record."""


HEADER = struct.Struct("<4sI")  # char[4] magic + u32 body_len (excludes header)
# VBR1 fixed part: stream_index, seq, wall_ms, src_w/h, model_w/h, scale/pads,
# align, kpt_per_det, n_det, inference_ms, queue_delay_ms, attr_per_det,
# reserved u8, reserved u16.
_BODY = struct.Struct("<IQd4i3fBBHffBBH")
_DET = struct.Struct("<5fiI")          # cx, cy, w, h, score, class_id, track_id
_KPT = struct.Struct("<3f")
# VBT1 fixed part (spec §6.12): stream_index, seq, wall_ms, src/model w/h,
# scale/pads, u8 align + 3 reserved, u16 n_tensors + u16 reserved.
_TBODY = struct.Struct("<IQd4i3fB3xHH")
# VBT1 per-tensor fixed part: dtype, n_dims, nhwc, reserved, 4 dims, scale,
# zero_point.
_TTENSOR = struct.Struct("<BBBB4ifi")

MAGIC_FRAME = b"VBR1"
MAGIC_EVENT = b"VBE1"
MAGIC_CONTROL = b"VBC1"
MAGIC_SNAPSHOT = b"VBS1"
MAGIC_TENSOR = b"VBT1"   # dev mode only (§6.12)

_ALIGN_NAMES = {0: "center", 1: "top_left"}


def decode_header(hdr: bytes) -> tuple[bytes, int]:
    """Decode the 8-byte record header -> (magic, body_len)."""
    if len(hdr) < HEADER.size:
        raise WireError(f"truncated header: {len(hdr)} bytes")
    magic, body_len = HEADER.unpack_from(hdr, 0)
    if magic not in (MAGIC_FRAME, MAGIC_EVENT, MAGIC_CONTROL, MAGIC_SNAPSHOT,
                     MAGIC_TENSOR):
        raise WireError(f"bad magic {magic!r}")
    return magic, body_len


def decode_result(body: bytes, attr_names: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """Decode a VBR1 body into a plain dict (spec §6.3 layout)."""
    if len(body) < _BODY.size:
        raise WireError(f"truncated VBR1 body: {len(body)} < {_BODY.size}")
    (stream_index, seq, wall_ms, src_w, src_h, model_w, model_h,
     scale, pad_x, pad_y, align, kpt_per_det, n_det,
     inference_ms, queue_delay_ms, attr_per_det, _r1, _r2) = _BODY.unpack_from(body, 0)
    offset = _BODY.size
    need = n_det * (_DET.size + kpt_per_det * _KPT.size + attr_per_det * 4)
    if len(body) - offset < need:
        raise WireError(f"truncated VBR1 detections: have {len(body) - offset}, need {need}")
    detections = []
    for _ in range(n_det):
        cx, cy, w, h, score, class_id, track_id = _DET.unpack_from(body, offset)
        offset += _DET.size
        detections.append({
            "cx": cx, "cy": cy, "w": w, "h": h, "score": score,
            "class_id": class_id, "track_id": track_id,
            "keypoints": [], "attrs": [],
        })
    # layout: all detections, then all keypoints, then all attrs (spec §6.3)
    for d in detections:
        for _k in range(kpt_per_det):
            d["keypoints"].append(_KPT.unpack_from(body, offset))
            offset += _KPT.size
    for d in detections:
        if attr_per_det:
            d["attrs"] = list(struct.unpack_from(f"<{attr_per_det}f", body, offset))
            offset += attr_per_det * 4
    return {
        "stream_index": stream_index,
        "seq": seq,
        "wall_ms": wall_ms,
        "src_w": src_w, "src_h": src_h, "model_w": model_w, "model_h": model_h,
        "scale": scale, "pad_x": pad_x, "pad_y": pad_y, "align": align,
        "inference_ms": inference_ms,
        "queue_delay_ms": queue_delay_ms,
        "detections": detections,
    }


def decode_event(body: bytes) -> dict[str, Any]:
    """Decode a VBE1 body: UTF-8 JSON object."""
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise WireError(f"bad VBE1 body: {e}") from e
    if not isinstance(obj, dict):
        raise WireError("VBE1 body must be a JSON object")
    return obj


def decode_snapshot(body: bytes) -> tuple[dict[str, Any], bytes]:
    """Decode a VBS1 body -> (meta, payload bytes)."""
    if len(body) < 4:
        raise WireError("truncated VBS1 body")
    (json_len,) = struct.unpack_from("<I", body, 0)
    if len(body) < 4 + json_len:
        raise WireError("truncated VBS1 meta")
    try:
        meta = json.loads(body[4:4 + json_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise WireError(f"bad VBS1 meta: {e}") from e
    return meta, bytes(body[4 + json_len:])


def decode_tensors(body: bytes) -> TensorFrame:
    """Decode a VBT1 body into a TensorFrame (stream_id filled by the shard;
    spec §6.12 layout)."""
    if len(body) < _TBODY.size:
        raise WireError(f"truncated VBT1 body: {len(body)} < {_TBODY.size}")
    (stream_index, seq, wall_ms, src_w, src_h, model_w, model_h,
     scale, pad_x, pad_y, align, n_tensors, _r) = _TBODY.unpack_from(body, 0)
    geom = LetterboxGeom(
        src_w=src_w, src_h=src_h, model_w=model_w, model_h=model_h,
        scale=scale, pad_x=pad_x, pad_y=pad_y,
        align=_ALIGN_NAMES.get(align, "center"))
    offset = _TBODY.size
    tensors: list[RawTensor] = []
    for _ in range(n_tensors):
        if len(body) - offset < _TTENSOR.size:
            raise WireError("truncated VBT1 tensor header")
        dtype, n_dims, nhwc, _r, d0, d1, d2, d3, t_scale, zero_point = \
            _TTENSOR.unpack_from(body, offset)
        offset += _TTENSOR.size
        if n_dims > 4:
            raise WireError(f"bad VBT1 n_dims {n_dims}")
        (name_len,) = struct.unpack_from("<H", body, offset)
        offset += 2
        if len(body) - offset < name_len:
            raise WireError("truncated VBT1 tensor name")
        name = body[offset:offset + name_len].decode("utf-8", "replace")
        offset += name_len
        if len(body) - offset < 4:
            raise WireError("truncated VBT1 tensor data length")
        (data_len,) = struct.unpack_from("<I", body, offset)
        offset += 4
        if len(body) - offset < data_len:
            raise WireError("truncated VBT1 tensor data")
        data = bytes(body[offset:offset + data_len])
        offset += data_len
        dims_all = (d0, d1, d2, d3)
        tensors.append(RawTensor(
            name=name, dtype=dtype, dims=tuple(dims_all[:n_dims]),
            scale=t_scale, zero_point=zero_point, nhwc=bool(nhwc), data=data))
    tf = TensorFrame(stream_id="", seq=seq, wall_ms=wall_ms, geom=geom,
                     tensors=tensors)
    tf.stream_index = stream_index      # transient; shard maps it to stream_id
    return tf


def encode_control_line(msg: dict[str, Any]) -> bytes:
    """Encode a Python -> native control record: one UTF-8 JSON line."""
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
