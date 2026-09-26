"""Wire record decoding for VBR1/VBE1/VBC1/VBS1 (spec BASE-1 §6.3).

Standard library only; decoding uses ``struct.unpack_from`` only (no
per-byte loops over buffers — spec §10.5①).
"""
from __future__ import annotations

import json
import struct
from typing import Any

__all__ = ["WireError", "HEADER", "MAGIC_FRAME", "MAGIC_EVENT", "MAGIC_CONTROL",
           "MAGIC_SNAPSHOT", "decode_header", "decode_result", "decode_event",
           "decode_snapshot", "encode_control_line"]


class WireError(Exception):
    """Malformed or truncated wire record."""


HEADER = struct.Struct("<4sI")  # char[4] magic + u32 body_len (excludes header)
# VBR1 fixed part: stream_index, seq, wall_ms, src_w/h, model_w/h, scale/pads,
# align, kpt_per_det, n_det, inference_ms, queue_delay_ms, attr_per_det,
# reserved u8, reserved u16.
_BODY = struct.Struct("<IQd4i3fBBHffBBH")
_DET = struct.Struct("<5fiI")          # cx, cy, w, h, score, class_id, track_id
_KPT = struct.Struct("<3f")

MAGIC_FRAME = b"VBR1"
MAGIC_EVENT = b"VBE1"
MAGIC_CONTROL = b"VBC1"
MAGIC_SNAPSHOT = b"VBS1"


def decode_header(hdr: bytes) -> tuple[bytes, int]:
    """Decode the 8-byte record header -> (magic, body_len)."""
    if len(hdr) < HEADER.size:
        raise WireError(f"truncated header: {len(hdr)} bytes")
    magic, body_len = HEADER.unpack_from(hdr, 0)
    if magic not in (MAGIC_FRAME, MAGIC_EVENT, MAGIC_CONTROL, MAGIC_SNAPSHOT):
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


def encode_control_line(msg: dict[str, Any]) -> bytes:
    """Encode a Python -> native control record: one UTF-8 JSON line."""
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
