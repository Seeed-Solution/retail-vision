"""M1.4: Python wire decode must match wire_case{1,2,3} fixtures (spec BASE-1 §8 M1.4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vision_base import wire

FIXTURES = Path(__file__).resolve().parents[3] / "contracts" / "fixtures" / "vb"
CASES = [FIXTURES / f"wire_case{i}" for i in (1, 2, 3)]

TOL = 1e-6


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= TOL, f"{a} vs {b}"
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{a} vs {b}"
        for x, y in zip(a, b):
            _close(x, y)
    elif isinstance(a, tuple) and isinstance(b, list):
        assert len(a) == len(b), f"{a} vs {b}"
        for x, y in zip(a, b):
            _close(x, y)
    else:
        assert a == b, f"{a!r} vs {b!r}"


@pytest.mark.parametrize("base", CASES, ids=lambda p: p.name)
def test_decode_matches_json(base: Path):
    raw = (base.with_suffix(".bin")).read_bytes()
    magic, body_len = wire.decode_header(raw[:8])
    assert magic == b"VBR1"
    body = raw[8:]
    assert body_len == len(body)

    want = json.loads(base.with_suffix(".json").read_text())
    got = wire.decode_result(body, want.get("attr_names", []))
    _close(got["stream_index"], want["stream_index"])
    _close(got["seq"], want["seq"])
    _close(got["wall_ms"], want["wall_ms"])
    for k in ("src_w", "src_h", "model_w", "model_h"):
        _close(got[k], want[k])
    for k in ("scale", "pad_x", "pad_y"):
        _close(got[k], want[k])
    _close(got["align"], want["align"])
    _close(got["inference_ms"], want["inference_ms"])
    _close(got["queue_delay_ms"], want["queue_delay_ms"])
    assert len(got["detections"]) == len(want["detections"])
    for d, wd in zip(got["detections"], want["detections"]):
        for k in ("cx", "cy", "w", "h", "score"):
            _close(d[k], wd[k])
        for k in ("class_id", "track_id"):
            assert d[k] == wd[k]
        _close(d["keypoints"], wd["keypoints"])
        _close(d["attrs"], wd["attrs"])


def test_truncated_record_raises():
    raw = (FIXTURES / "wire_case2.bin").read_bytes()
    body = raw[8:]
    with pytest.raises(wire.WireError):
        wire.decode_result(body[:-3], [])
    with pytest.raises(wire.WireError):
        wire.decode_result(body[:20], [])
    with pytest.raises(wire.WireError):
        wire.decode_header(raw[:5])
    with pytest.raises(wire.WireError):
        wire.decode_header(b"XXXX" + raw[4:8])


def test_event_decode():
    ev = wire.decode_event(b'{"type":"line_cross","track_id":7,"fields":{}}')
    assert ev["type"] == "line_cross" and ev["track_id"] == 7
    with pytest.raises(wire.WireError):
        wire.decode_event(b"\xff\xfe")
    with pytest.raises(wire.WireError):
        wire.decode_event(b"[1,2]")


def test_snapshot_decode():
    import struct as _s

    meta = b'{"req":"r-9","w":8,"h":6}'
    payload = b"\xff\xd8\xff\xe0jpegdata"
    body = _s.pack("<I", len(meta)) + meta + payload
    m, p = wire.decode_snapshot(body)
    assert m == {"req": "r-9", "w": 8, "h": 6}
    assert p == payload
    with pytest.raises(wire.WireError):
        wire.decode_snapshot(body[:3])


def test_control_line():
    line = wire.encode_control_line({"op": "stop", "req": "r-1"})
    assert line.endswith(b"\n")
    assert json.loads(line.decode("utf-8")) == {"op": "stop", "req": "r-1"}
