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

pytestmark = pytest.mark.native


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
