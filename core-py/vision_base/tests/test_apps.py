"""ConfigApp tests (spec BASE-1 §8 M1.20): zero-code config mode, payload
shape vs contracts/vb-event|frame.schema.json, config validation of
``streams[].options.analyzers`` / ``app.options.frame_every``, app-module
loading from the config directory, and the cross-check against
``vb-runtime --standalone`` (``native`` marker, needs VB_RUNTIME_BIN).
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

from fake_broker import FakeBroker  # noqa: E402
from vision_base.apps import ConfigApp  # noqa: E402
from vision_base.config import ConfigError, load  # noqa: E402
from vision_base.hooks import Outgoing, StreamContext  # noqa: E402
from vision_base.letterbox import fit, to_model_norm, to_source_norm  # noqa: E402
from vision_base.types import Detection, Event, FrameResult, StreamSpec  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMAS = {
    "vb.event/1": json.loads((ROOT / "contracts" / "vb-event.schema.json").read_text()),
    "vb.frame/1": json.loads((ROOT / "contracts" / "vb-frame.schema.json").read_text()),
}
FIXTURE = ROOT / "contracts" / "fixtures" / "vb" / "standalone_synthetic.json"
QUICKSTART = ROOT / "examples" / "quickstart"


# ------------------------------------------------------------------ helpers

def make_ctx(stream_id="cam-a", options=None):
    spec = StreamSpec(stream_id=stream_id, url="synthetic://", options=options or {})
    return StreamContext(index=0, spec=spec, runtime=None)


def make_app(**options) -> ConfigApp:
    app = ConfigApp()
    app.configure(options, "dev-1")
    return app


# --------------------------------------------------------------- unit tests

def test_analyzers_from_options():
    analyzers = [{"name": "zone", "config": {"zones": []}}]
    app = make_app()
    spec = StreamSpec("cam", "synthetic://", options={"analyzers": analyzers})
    assert app.analyzers(spec) == analyzers
    assert app.plugins() == []
    assert app.name == "config"


def test_defaults_no_frames():
    app = make_app()
    assert app.frame_every == 0
    assert app.wants_frames is False


def test_on_event_payload_matches_event_schema():
    app = make_app()
    ctx = make_ctx()
    ev = Event(stream_id="cam-a", seq=120, wall_ms=1790000000123.4,
               analyzer="line_cross", type="line_cross", track_id=7,
               fields={"line_id": "door", "direction": "forward",
                       "anchor": [0.52, 0.61], "class_id": 0, "score": 0.81})
    outs = app.on_event(ctx, ev)
    assert len(outs) == 1
    o = outs[0]
    assert o.topic_suffix == "events/cam-a" and o.qos == 1
    jsonschema.validate(o.payload, SCHEMAS["vb.event/1"])
    assert o.payload["device_id"] == "dev-1"
    assert o.payload["ts_ms"] == 1790000000123
    assert o.payload["fields"] == ev.fields


def test_on_frame_payload_matches_frame_schema():
    app = make_app(frame_every=3)
    geom = fit(320, 240, 640, 640)
    res = FrameResult(stream_id="cam-a", seq=5, wall_ms=1000.6, geom=geom,
                      inference_ms=12.4, queue_delay_ms=0.1,
                      detections=[Detection(cx=0.5, cy=0.5, w=0.1, h=0.2, score=0.81,
                                            class_id=0, track_id=7)])
    outs = app.on_frame(make_ctx(), res)
    assert len(outs) == 1
    o = outs[0]
    assert o.topic_suffix == "frames/cam-a" and o.qos == 0
    jsonschema.validate(o.payload, SCHEMAS["vb.frame/1"])
    det = o.payload["detections"][0]
    assert det["track_id"] == 7 and det["class_id"] == 0
    assert det["score"] == 0.81
    assert len(det["box"]) == 4 and det["box"][0] < det["box"][2]
    assert "keypoints" not in det and "attrs" not in det   # omitted when empty
    assert o.payload["inference_ms"] == 12.4


def test_frame_every_stride():
    app = make_app(frame_every=3)
    assert app.wants_frames is True
    assert app.frame_stride == 3
    geom = fit(320, 240, 640, 640)
    res = FrameResult(stream_id="s", seq=1, wall_ms=0.0, geom=geom,
                      inference_ms=1.0, queue_delay_ms=0.0, detections=[])
    # every delivered frame publishes (shard does the stride sampling)
    assert app.on_frame(make_ctx("s"), res)
    assert app.on_frame(make_ctx("s"), res)


def test_floats_rounded_to_6():
    app = make_app()
    ev = Event("s", 1, 1.0, "zone", "zone_enter", 3,
               fields={"x": 0.123456789, "y": [0.987654321]})
    assert app.event_payload(ev)["fields"] == {"x": 0.123457, "y": [0.987654]}


# ---------------------------------------------------------- config validation

def _write_cfg(tmp_path, data):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    return p


def test_config_rejects_unknown_analyzer(tmp_path):
    cfg = json.loads(FIXTURE.read_text())
    cfg["backend"]["model_path"] = "synthetic.onnx"  # required by config.load
    cfg["mqtt"] = {"host": "127.0.0.1", "topic_root": "t"}
    cfg["app"] = {"module": "vision_base.apps:ConfigApp"}
    cfg["streams"][0]["options"]["analyzers"] = [
        {"name": "no_such_analyzer", "config": {}}]
    with pytest.raises(ConfigError):
        load(str(_write_cfg(tmp_path, cfg)))


def test_config_accepts_builtin_and_plugin_names(tmp_path):
    cfg = json.loads(FIXTURE.read_text())
    cfg["backend"]["model_path"] = "synthetic.onnx"
    cfg["mqtt"] = {"host": "127.0.0.1", "topic_root": "t"}
    cfg["app"] = {"module": "vision_base.apps:ConfigApp"}
    cfg["streams"][0]["options"]["analyzers"] = [
        {"name": "zone", "config": {}},
        {"name": "plugin:myplug", "config": {}}]
    loaded = load(str(_write_cfg(tmp_path, cfg)))
    assert loaded.streams[0]["options"]["analyzers"][1]["name"] == "plugin:myplug"


def test_config_frame_every_validation(tmp_path):
    cfg = json.loads(FIXTURE.read_text())
    cfg["backend"]["model_path"] = "synthetic.onnx"
    cfg["mqtt"] = {"host": "127.0.0.1", "topic_root": "t"}
    cfg["app"] = {"module": "vision_base.apps:ConfigApp",
                  "options": {"frame_every": 3}}
    assert load(str(_write_cfg(tmp_path, cfg)))  # ok
    cfg["app"]["options"]["frame_every"] = -1
    with pytest.raises(ConfigError):
        load(str(_write_cfg(tmp_path, cfg)))


# ------------------------------------------------- app module from config dir

def test_app_module_imported_from_config_dir(tmp_path):
    cfg = {"schema": "vb.config/1", "device_id": "d",
           "backend": {"name": "synthetic", "model_w": 64, "model_h": 64,
                     "model_path": "synthetic.onnx"},
           "mqtt": {"host": "127.0.0.1", "topic_root": "t"},
           "app": {"module": "hooks:DoorCounter"},
           "streams": [{"stream_id": "door", "url": "synthetic://",
                        "options": {"analyzers": [
                            {"name": "line_cross", "config": {}}]}}]}
    # hooks:DoorCounter from examples/quickstart (§5.5.2), unmodified
    import shutil
    shutil.copy(QUICKSTART / "hooks.py", tmp_path / "hooks.py")
    from vision_base.supervisor import load_app
    app = load_app("hooks:DoorCounter", str(tmp_path))
    app.configure({"frame_every": 0}, "door-01")
    ctx = make_ctx("door", {"analyzers": [{"name": "line_cross", "config": {}}]})
    ctx.state = {}
    ev = Event("door", 1, 1.0, "line_cross", "line_cross", 7,
               fields={"direction": "forward"})
    outs = app.on_event(ctx, ev)
    assert [(o.topic_suffix, o.qos) for o in outs] == \
        [("events/door", 1), ("count/door", 0)]
    assert outs[0].payload["schema"] == "vb.event/1"
    assert outs[1].payload == {"inside": 1}
    ev2 = Event("door", 2, 1.0, "line_cross", "line_cross", 8,
                fields={"direction": "backward"})
    outs = app.on_event(ctx, ev2)
    assert outs[1].payload == {"inside": 0}


def test_hooks_example_is_12_lines():
    text = (QUICKSTART / "hooks.py").read_text().strip("\n")
    assert len(text.splitlines()) == 12


# ------------------------------------------------ cross-check (native marker)

BIN = os.environ.get("VB_RUNTIME_BIN")
pytestmark_native = pytest.mark.native


@pytest.mark.native
@pytest.mark.skipif(not BIN, reason="needs VB_RUNTIME_BIN")
def test_configapp_vs_standalone_crosscheck(tmp_path):
    """M1.20 acceptance: same config + synthetic input through (a) the Python
    ConfigApp path (fake broker collects MQTT) and (b) vb-runtime --standalone
    --output jsonl must yield identical event sets (ignoring ts_ms)."""
    broker = FakeBroker().start()
    cfg = json.loads(FIXTURE.read_text())
    cfg["backend"]["model_path"] = "synthetic.onnx"
    cfg["state_dir"] = str(tmp_path / "state")
    cfg["mqtt"] = {"host": broker.host, "port": broker.port, "topic_root": "X"}
    cfg["app"] = {"module": "vision_base.apps:ConfigApp"}
    cfg["native"] = {"binary": BIN}
    py_cfg = _write_cfg(tmp_path, cfg)

    py = subprocess.Popen([sys.executable, "-u", "-m", "vision_base.main",
                           "--config", str(py_cfg)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          cwd=str(ROOT / "core-py"))
    native = subprocess.Popen(
        [BIN, "--standalone", "--config", str(FIXTURE), "--output", "jsonl"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            evs = [json.loads(p["payload"]) for p in broker.published
                   if p["topic"].startswith("X/events/")
                   and json.loads(p["payload"])["type"] == "zone_enter"]
            if len(evs) >= 10:
                break
            time.sleep(0.2)
    finally:
        for proc in (py, native):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            if proc is py:
                proc.communicate(timeout=15.0)
        broker.stop()
    out, err = native.communicate(timeout=10.0)
    assert native.returncode == 0, err.decode()

    py_events = [json.loads(p["payload"]) for p in broker.published
                 if p["topic"].startswith("X/events/")]
    native_events = [json.loads(l) for l in out.decode().splitlines()
                     if l.strip() and json.loads(l).get("schema") == "vb.event/1"]
    # zone_exit.dwell_s is wall-clock dependent and cannot match across two
    # independent runs; the deterministic events (zone_enter) carry fixed
    # fields and are the cross-check set.
    py_events = [e for e in py_events if e["type"] == "zone_enter"]
    native_events = [e for e in native_events if e["type"] == "zone_enter"]
    assert len(py_events) >= 10 and len(native_events) >= 10, \
        f"not enough events: py={len(py_events)} native={len(native_events)}"

    def key(e):
        return (e["stream_id"], e["track_id"], e["type"])

    py_map = {}
    for e in py_events:
        py_map.setdefault(key(e), []).append(e)
    matched = 0
    for e in native_events:
        cands = py_map.get(key(e), [])
        if not cands:
            continue
        c = cands.pop(0)
        matched += 1
        for field in ("schema", "device_id", "stream_id", "seq", "analyzer",
                      "type", "track_id"):
            assert c[field] == e[field], f"{field}: {c[field]} != {e[field]}"
        assert set(c["fields"]) == set(e["fields"])
        for k in e["fields"]:
            a, b = c["fields"][k], e["fields"][k]
            # §6.10.2: the two implementations must agree *exactly* — the
            # earlier 1e-6 tolerance hid a genuine field-path divergence
            # (§6.10.2 rules 1-3), and a tolerance is not what the spec
            # requires ("逐字段精确相等，不是近似相等").
            if isinstance(a, list) and isinstance(b, list):
                assert len(a) == len(b), f"fields[{k}]: {a} != {b}"
                for x, y in zip(a, b):
                    assert x == y, f"fields[{k}]: {a!r} != {b!r}"
            else:
                assert a == b, f"fields[{k}]: {a!r} != {b!r}"
        if matched >= 10:
            break
    assert matched >= 10, f"only {matched} matched events"


# --------------------------------------------- §6.10.2 numeric consistency

def test_rule1_round6_leaves_int_bool_and_str_alone():
    """§6.10.2 rule 1: `fields` is rounded recursively, but only floats — an
    integer must not become a float, a bool must not become a number."""
    app = make_app()
    ev = Event("s", 1, 1.0, "zone", "zone_enter", 3, fields={
        "i": 3, "b": True, "s": "x", "f": 0.123456789,
        "nested": {"i": 4, "f": 1.9999999, "l": [1, 2.5]},
    })
    got = app.event_payload(ev)["fields"]
    assert got == {"i": 3, "b": True, "s": "x", "f": 0.123457,
                   "nested": {"i": 4, "f": 2.0, "l": [1, 2.5]}}
    assert isinstance(got["i"], int) and not isinstance(got["i"], bool)
    assert isinstance(got["b"], bool)
    assert isinstance(got["s"], str)
    assert isinstance(got["nested"]["i"], int)
    assert isinstance(got["nested"]["l"][0], int)


def test_rule1_round6_is_six_decimals_not_six_significant_digits():
    """§6.10.2 rule 1: `round(x, 6)`, not `%.6g` (which diverges outside
    [0.1, 1) — `inference_ms` was the measured case)."""
    app = make_app()
    res = FrameResult(stream_id="s", seq=1, wall_ms=0.0, geom=fit(320, 240, 640, 640),
                      inference_ms=12.3456789, queue_delay_ms=0.0, detections=[])
    assert app.frame_payload(res)["inference_ms"] == 12.345679
    res.inference_ms = 1234.5678901
    assert app.frame_payload(res)["inference_ms"] == 1234.56789


def test_rule3_box_and_keypoints_are_clipped_to_unit_range():
    """§6.10.2 rule 3: letterbox padding back-computes a full-canvas box out of
    range; the output must be clamped to [0, 1]."""
    # 1280x720 into 640x640: scale 0.5, pad_y = (640 - 360)/2 = 140
    geom = fit(1280, 720, 640, 640)
    assert geom.pad_y == 140.0
    app = make_app(frame_every=1)
    # a box covering the whole model canvas -> source y = [-0.389, 1.389]
    det = Detection(cx=0.5, cy=0.5, w=1.0, h=1.0, score=0.5, class_id=0,
                    track_id=1, keypoints=(0.5, 0.0, 0.9, 0.5, 1.0, 0.9))
    res = FrameResult(stream_id="s", seq=1, wall_ms=0.0, geom=geom,
                      inference_ms=0.0, queue_delay_ms=0.0, detections=[det])
    payload = app.frame_payload(res)
    box = payload["detections"][0]["box"]
    assert box[0] == 0.0 and box[2] == 1.0
    assert box[1] == 0.0 and box[3] == 1.0, box       # was -0.389 / 1.389
    kps = payload["detections"][0]["keypoints"]
    assert kps[0] == [0.5, 0.0, 0.9]
    assert kps[1] == [0.5, 1.0, 0.9]
    for x, y, conf in kps:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    jsonschema.validate(payload, SCHEMAS["vb.frame/1"])


def test_rule4_ts_ms_uses_banker_rounding():
    """§6.10.2 rule 4: half-to-even, as Python's round() does (C++ must use
    nearbyint, not llround)."""
    app = make_app()
    for wall_ms, expect in [(1000.5, 1000), (1001.5, 1002), (1002.5, 1002),
                            (1790000000123.4, 1790000000123),
                            (1790000000123.6, 1790000000124)]:
        ev = Event("s", 1, wall_ms, "zone", "zone_enter", 0, fields={})
        assert app.event_payload(ev)["ts_ms"] == expect
        res = FrameResult(stream_id="s", seq=1, wall_ms=wall_ms,
                          geom=fit(320, 240, 640, 640), inference_ms=0.0,
                          queue_delay_ms=0.0, detections=[])
        assert app.frame_payload(res)["ts_ms"] == expect


def test_rule2_source_transform_stays_double_precision():
    """§6.10.2 rule 2: the inverse transform has no float32 round trip — the
    sixth decimal of a known case must match the double computation."""
    geom = fit(1920, 1080, 640, 640)
    sx, sy = to_source_norm(geom, 0.5, 0.5)
    assert (sx, sy) == (0.5, 0.5)                 # exact for a centred point
    px = 0.1234567
    mx, my = to_model_norm(geom, px, 0.7654321)
    bx, by = to_source_norm(geom, mx, my)
    # double round trip is exact to well under 1e-12; float32 would land ~1e-8 off
    assert abs(bx - px) < 1e-12 and abs(by - 0.7654321) < 1e-12
