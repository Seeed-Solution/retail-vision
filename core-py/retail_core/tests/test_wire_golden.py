"""End-to-end wire goldens for the retail service.

Runs ``retail_core.app.main`` with a stub detector, a stub video source and a
per-thread fake clock against a byte-recording broker, then compares every
packet the service sent (CONNECT, retained status, results, DISCONNECT) with
``fixtures/golden/wire_*.json``. The goldens were recorded on the pre-
vision_base implementation (``MqttPublisher``); any change to topic, payload
bytes, retain flag, QoS, field rounding or publish cadence fails here.

PINGREQ is excluded from the comparison: the legacy client never sent one,
the vision_base client sends one after ``keepalive_sec`` of silence, and the
runs below are far shorter than the keepalive.

Regenerate (only when a wire change is intended):
    RETAIL_GOLDEN_REGEN=1 uv run pytest core-py/retail_core/tests/test_wire_golden.py
"""
from __future__ import annotations

import json
import os
import pathlib
import struct
import sys
import threading
import time
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "contracts"))

from retail_recording_broker import RecordingBroker, packet_type  # noqa: E402
from validate_payload import check as check_payload  # noqa: E402

from retail_core import app as app_mod  # noqa: E402
from retail_core import video as video_mod  # noqa: E402
from retail_core.tracker import DetectionBox  # noqa: E402

GOLDEN_DIR = HERE / "fixtures" / "golden"
SEQUENCE = json.loads((HERE / "fixtures" / "tracker_sequence.json").read_text())
REGEN = os.environ.get("RETAIL_GOLDEN_REGEN") == "1"
FPS = 15.0


# ---------------------------------------------------------------- stubs
class FakeClock:
    """Per-thread monotonic clock: each stream advances its own by 1/FPS per
    frame so cadence and velocities are independent of scheduling."""

    def __init__(self):
        self._local = threading.local()

    def monotonic(self):
        return getattr(self._local, "now", 1000.0)

    def advance(self, dt):
        self._local.now = self.monotonic() + dt

    def time(self):
        return 1_760_000_000.0 + self.monotonic()

    @staticmethod
    def sleep(seconds):
        time.sleep(min(seconds, 0.01))


def frames_for(stream_id):
    frames = SEQUENCE["frames"]
    if stream_id == "cam-b":  # mirrored copy so the two topics differ
        return [[[1.0 - b[0], b[1], b[2], b[3], b[4]] for b in f] for f in frames]
    return frames


class Harness:
    def __init__(self, stream_ids):
        self.clock = FakeClock()
        self.remaining = set(stream_ids)
        self.lock = threading.Lock()


HARNESS: Harness | None = None


class StubSource:
    backend_name = "golden_stub"

    def __init__(self, url, size=640, **_):
        self.stream_id = url.rsplit("/", 1)[-1]
        self.index = 0

    def read(self):
        frames = frames_for(self.stream_id)
        if self.index >= len(frames):
            with HARNESS.lock:
                HARNESS.remaining.discard(self.stream_id)
                if not HARNESS.remaining:
                    app_mod.RUNNING = False
            return None
        HARNESS.clock.advance(1.0 / FPS)
        idx = self.index
        self.index += 1
        return (self.stream_id, idx)

    def close(self):
        pass


class StubDetector:
    def detect(self, frame, score_threshold, nms_threshold):
        stream_id, idx = frame
        boxes = [DetectionBox(*b) for b in frames_for(stream_id)[idx]]
        return boxes, 10.0 + (idx % 7) * 0.35

    def close(self):
        pass


STUB_BACKEND = types.SimpleNamespace(create_detector=lambda cfg, stream: StubDetector())
video_mod.register_source("golden_stub", StubSource)


# ------------------------------------------------------------- packets
def parse_publish(raw):
    """(topic, payload, qos, retain) of one PUBLISH packet."""
    i = 1
    while raw[i] & 0x80:
        i += 1
    i += 1
    flags = raw[0] & 0x0F
    tlen = struct.unpack_from(">H", raw, i)[0]
    topic = raw[i + 2:i + 2 + tlen].decode()
    pos = i + 2 + tlen
    qos = (flags >> 1) & 3
    if qos:
        pos += 2
    return topic, raw[pos:], qos, bool(flags & 1)


def normalize(packets):
    """Split one connection's packets into control sequence + per-topic results."""
    sequence, results = [], {}
    for _, raw in packets:
        kind = packet_type(raw)
        if kind == 12:  # PINGREQ, see module docstring
            continue
        if kind == 3:
            topic = parse_publish(raw)[0]
            if "/results/" in topic:
                results.setdefault(topic, []).append(raw.hex())
                sequence.append(f"<result {topic}>")
                continue
        sequence.append(raw.hex())
    return sequence, results


def collapse(sequence):
    """Collapse runs of result placeholders; per-topic order is checked
    separately and cross-topic interleaving is scheduler-dependent."""
    out = []
    for item in sequence:
        if item.startswith("<result") and out and out[-1] == "<results>":
            continue
        out.append("<results>" if item.startswith("<result") else item)
    return out


# ------------------------------------------------------------ scenario
def run_app(tmp_path, monkeypatch, broker, cfg_extra, streams):
    global HARNESS
    HARNESS = Harness([s["id"] for s in streams if s.get("enabled", True)])
    cfg = {
        "installation": "golden-site",
        "backend": "golden",
        "model_path": "/models/none",
        "input_size": 640,
        "mqtt": {"host": "127.0.0.1", "port": broker.port},
        "video": {"backend": "golden_stub", "fallback": "none",
                  "failure_limit": 1_000_000},
        "streams": streams,
    }
    for key, value in cfg_extra.items():
        if key == "mqtt":
            cfg["mqtt"].update(value)
        else:
            cfg[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(app_mod, "load_backend", lambda name: STUB_BACKEND)
    monkeypatch.setattr(app_mod, "time", HARNESS.clock)
    monkeypatch.setattr(app_mod.signal, "signal", lambda *a: None)
    monkeypatch.setattr(app_mod, "RUNNING", True)
    app_mod.main(["--config", str(path)])
    assert broker.wait_for(lambda p: p and packet_type(p[-1][1]) == 14), \
        "no DISCONNECT recorded"
    return broker.snapshot()


SCENARIOS = {
    "single": dict(
        cfg_extra={
            "mqtt": {"client_id": "golden-single", "username": "u1", "password": "p1"},
            "publish_hz": 1.0,
            "frame_width": 1280, "frame_height": 720,
            "count_zone": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]],
            "entry_line": {"a": [0.6, 0.0], "b": [0.6, 1.0], "ab_in": True},
        },
        streams=[{"id": "cam-a", "rtsp_url": "rtsp://127.0.0.1/cam-a"}],
    ),
    "dual": dict(
        cfg_extra={
            "mqtt": {"client_id": "golden-dual", "keepalive_sec": 15},
            "window_duration": 10.0,
            "tracker": {"edge_margin": 0.1},
        },
        streams=[
            {"id": "cam-a", "rtsp_url": "rtsp://127.0.0.1/cam-a"},
            {"id": "cam-b", "rtsp_url": "rtsp://127.0.0.1/cam-b", "publish_hz": 2.0,
             "frame_width": 1920, "frame_height": 1080},
            {"id": "cam-off", "rtsp_url": "rtsp://127.0.0.1/cam-off", "enabled": False},
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_wire_golden(name, tmp_path, monkeypatch):
    broker = RecordingBroker().start()
    try:
        packets = run_app(tmp_path, monkeypatch, broker, **SCENARIOS[name])
    finally:
        broker.stop()
    assert {conn for conn, _ in packets} == {0}, "service opened more than one session"
    sequence, results = normalize(packets)
    observed = {"sequence": collapse(sequence), "results": results}

    # contract sanity on every result payload
    for topic, items in results.items():
        for hexed in items:
            _, payload, qos, retain = parse_publish(bytes.fromhex(hexed))
            assert qos == 0 and retain is False
            assert check_payload(json.loads(payload)) == [], topic

    golden_path = GOLDEN_DIR / f"wire_{name}.json"
    if REGEN:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(observed, indent=1, sort_keys=True) + "\n")
    golden = json.loads(golden_path.read_text())
    assert observed["sequence"] == golden["sequence"]
    assert sorted(observed["results"]) == sorted(golden["results"])
    for topic in golden["results"]:
        assert observed["results"][topic] == golden["results"][topic], topic


def test_default_client_id(tmp_path, monkeypatch):
    """Without mqtt.client_id the CONNECT carries retail-vision-<pid>."""
    broker = RecordingBroker().start()
    try:
        packets = run_app(tmp_path, monkeypatch, broker, {},
                          [{"id": "cam-a", "rtsp_url": "rtsp://127.0.0.1/cam-a"}])
    finally:
        broker.stop()
    connect = packets[0][1]
    assert packet_type(connect) == 1
    expected = f"retail-vision-{os.getpid()}".encode()
    assert struct.pack(">H", len(expected)) + expected in connect
