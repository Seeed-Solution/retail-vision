"""vb-runtime --standalone --output jsonl tests (spec BASE-1 §8 M1.18).

Runs the real binary via VB_RUNTIME_BIN against the synthetic backend
fixture; every stdout line must be valid JSON passing the matching
contracts/vb-{event,frame,status}.schema.json.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import jsonschema  # noqa: E402

BIN = os.environ.get("VB_RUNTIME_BIN")
ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "contracts" / "fixtures" / "vb" / "standalone_synthetic.json"
SCHEMAS = {
    "vb.event/1": json.loads((ROOT / "contracts" / "vb-event.schema.json").read_text()),
    "vb.frame/1": json.loads((ROOT / "contracts" / "vb-frame.schema.json").read_text()),
    "vb.status/1": json.loads((ROOT / "contracts" / "vb-status.schema.json").read_text()),
}
STREAMS = {"cam-a", "cam-b"}

pytestmark = [
    pytest.mark.native,
    # Depth guard only: CI sets VB_RUNTIME_BIN to the freshly built binary so
    # these tests really execute. Without it there is no binary to drive, and
    # skipping beats a TypeError from a None executable path.
    pytest.mark.skipif(not BIN, reason="VB_RUNTIME_BIN is not set"),
]


def run_standalone(tmp_path, *extra, seconds=4.0, config=FIXTURE):
    proc = subprocess.Popen(
        [BIN, "--standalone", "--config", str(config), "--output", "jsonl", *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(seconds)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=10.0)
    return proc.returncode, out.decode(), err.decode()


def parse_all(out):
    lines = [l for l in out.splitlines() if l.strip()]
    objs = [json.loads(l) for l in lines]  # raises on any non-JSON line
    for o in objs:
        jsonschema.validate(o, SCHEMAS[o["schema"]])
    return objs


def test_standalone_jsonl_sigterm_exit0_and_schemas():
    rc, out, err = run_standalone(None)
    assert rc == 0, err
    objs = parse_all(out)
    events = [o for o in objs if o["schema"] == "vb.event/1"]
    assert any(o["type"] == "zone_enter" for o in events)
    assert {o["stream_id"] for o in events} <= STREAMS
    assert {o["stream_id"] for o in events} == STREAMS
    # last line: online:false status
    assert objs[-1]["schema"] == "vb.status/1" and objs[-1]["online"] is False
    # stdout only carries the JSON lines (logs go to stderr)
    assert objs


def test_standalone_frame_every_counts():
    rc, out, err = run_standalone(None, "--frame-every", "5")
    assert rc == 0, err
    objs = parse_all(out)
    per = {}
    for o in objs:
        if o["schema"] == "vb.frame/1":
            per.setdefault(o["stream_id"], 0)
            per[o["stream_id"]] += 1
    assert set(per) == STREAMS
    for n in per.values():
        assert 8 <= n <= 16, per


def test_standalone_float_rule_matches_python(tmp_path):
    """inference_ms is not normalised, so it exercises the rounding rule
    outside [0, 1): C++ output must equal Python apps._num6 of the same
    float32 value (§6.10.2)."""
    import struct
    from vision_base.apps import _num6
    infer = 12.3456789
    cfg = json.loads(FIXTURE.read_text())
    cfg["backend"]["infer_ms"] = infer
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    rc, out, err = run_standalone(tmp_path, "--frame-every", "1", config=p)
    assert rc == 0, err
    got = {o["inference_ms"] for o in parse_all(out) if o["schema"] == "vb.frame/1"}
    f32 = struct.unpack("f", struct.pack("f", infer))[0]
    assert got == {_num6(f32)}, (got, _num6(f32))
    assert _num6(f32) != float("%.6g" % f32)  # the old rule would fail here


def test_standalone_no_frames_by_default():
    rc, out, err = run_standalone(None, "--frame-every", "0")
    assert rc == 0, err
    objs = parse_all(out)
    assert not [o for o in objs if o["schema"] == "vb.frame/1"]


def test_standalone_mutex_with_listen(tmp_path):
    proc = subprocess.run(
        [BIN, "--standalone", "--listen", str(tmp_path / "x.sock"),
         "--config", str(FIXTURE), "--output", "jsonl"],
        capture_output=True, timeout=10.0)
    assert proc.returncode == 2


def test_standalone_empty_streams(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(
        {"schema": "vb.config/1", "backend": {"name": "synthetic"}, "streams": []}))
    proc = subprocess.run(
        [BIN, "--standalone", "--config", str(cfg), "--output", "jsonl"],
        capture_output=True, timeout=10.0)
    assert proc.returncode == 1


def test_standalone_dev_rejected():
    """§6.12: --dev exists to hand VBT1 raw tensors to Python. standalone has no
    tensor output, so the combination is refused at startup instead of silently
    dropping the VBT1 records the operator expects."""
    proc = subprocess.run(
        [BIN, "--standalone", "--config", str(FIXTURE), "--output", "jsonl", "--dev"],
        capture_output=True, timeout=10.0)
    assert proc.returncode == 2
    assert "--dev" in proc.stderr.decode()


def test_standalone_records_never_interleave():
    """§6.10.1: the status timer and the Writer thread share stdout, so every
    line must be exactly one JSON document (a record split into body + newline
    interleaves into "JSON_A JSON_B\\n\\n")."""
    rc, out, err = run_standalone(None, "--frame-every", "1", "--status-every", "1",
                                  seconds=5.0)
    assert rc == 0, err
    objs = parse_all(out)  # raises on any line that is not one JSON document
    assert [o for o in objs if o["schema"] == "vb.status/1"]
    assert [o for o in objs if o["schema"] == "vb.frame/1"]
    assert not [l for l in out.splitlines() if not l.strip()]


@pytest.mark.parametrize("section,key,value,expected", [
    ("backend", "align", "top_left",
     "backend.align: 'top_left' is not supported by this vb-runtime (only 'center')"),
    ("backend", "nms_threshold", 0.5,
     "backend.nms_threshold: 0.5 is not supported by this vb-runtime (only 0.45)"),
    ("native", "jpeg_quality", 90,
     "native.jpeg_quality: 90 is not supported by this vb-runtime (only 85)"),
])
def test_standalone_rejects_unsupported_value_at_startup(tmp_path, section, key,
                                                         value, expected):
    """Standalone mode does not go through vision_base.config.load, so the
    native config path must refuse these values itself (same message)."""
    cfg = json.loads(FIXTURE.read_text())
    cfg.setdefault(section, {})[key] = value
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    proc = subprocess.run(
        [BIN, "--standalone", "--config", str(p), "--output", "jsonl"],
        capture_output=True, timeout=10.0)
    assert proc.returncode != 0
    assert expected in proc.stderr.decode()
    assert proc.stdout == b""


def test_standalone_accepts_supported_defaults(tmp_path):
    cfg = json.loads(FIXTURE.read_text())
    cfg["backend"].update({"align": "center", "nms_threshold": 0.45})
    cfg.setdefault("native", {})["jpeg_quality"] = 85
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    rc, out, err = run_standalone(tmp_path, seconds=2.0, config=p)
    assert rc == 0, err
    assert parse_all(out)
