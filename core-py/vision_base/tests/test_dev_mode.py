"""Dev-mode raw tensor passthrough tests (spec BASE-1 §6.12, M1.23b).

Covers every M1.23b acceptance bullet:
- decode_tensors(wire_tensor_case1.bin) equals wire_tensor_case1.json
- raw_tensors=true without allow_dev raises ConfigError
- VB_PRODUCTION=1 -> main --dev exits 2
- fake_runtime sends VBT1 -> hook on_tensors receives it
- VBT1 received outside dev mode -> connection closed + error recorded
"""
from __future__ import annotations

import json
import os
import pathlib
import struct
import sys
import threading
import time

import pytest

from vision_base.runtime_client import RuntimeGone

from vision_base.config import ConfigError, load, runtime_config
from vision_base.main import main
from vision_base.shard import Shard
from vision_base.types import StreamSpec
from vision_base import wire

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / "contracts" / "fixtures" / "vb"
FAKE = os.path.join(os.path.dirname(__file__), "fake_runtime.py")
FAKE_ARGV = [sys.executable, FAKE, "--ipc-fd", "{fd}"]

RT_CFG = {"backend": {"name": "synthetic", "model_w": 320, "model_h": 320},
          "contexts_per_worker": 1, "tracker": {},
          "analyzers": {"plugins": []}, "snapshot_ring": 2,
          "dev": {"raw_tensors": True, "max_fps": 1.0, "max_streams": 1}}


def wait_for(predicate, timeout_s=6.0, what="condition"):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


# ------------------------------------------------------------- decode_tensors

def test_decode_tensors_fixture():
    """decode_tensors(wire_tensor_case1.bin) == wire_tensor_case1.json."""
    blob = (FIXTURES / "wire_tensor_case1.bin").read_bytes()
    assert blob[:4] == b"VBT1"
    (body_len,) = struct.unpack_from("<I", blob, 4)
    tf = wire.decode_tensors(blob[8:8 + body_len])
    ref = json.loads((FIXTURES / "wire_tensor_case1.json").read_text())

    assert tf.stream_index == ref["stream_index"]
    assert tf.seq == ref["seq"]
    assert tf.wall_ms == ref["wall_ms"]
    assert tf.geom.src_w == ref["src_w"] and tf.geom.src_h == ref["src_h"]
    assert tf.geom.model_w == ref["model_w"] and tf.geom.model_h == ref["model_h"]
    assert tf.geom.scale == pytest.approx(ref["scale"], rel=1e-6)
    assert tf.geom.pad_x == pytest.approx(ref["pad_x"], abs=1e-6)
    assert tf.geom.pad_y == pytest.approx(ref["pad_y"], abs=1e-6)
    assert tf.geom.align == "center"
    assert len(tf.tensors) == len(ref["tensors"]) == 2
    for got, want in zip(tf.tensors, ref["tensors"]):
        assert got.name == want["name"]
        assert got.dtype == want["dtype"]
        assert got.dims == tuple(want["dims"])
        assert got.scale == pytest.approx(want["scale"], rel=1e-6)
        assert got.zero_point == want["zero_point"]
        assert got.nhwc == want["nhwc"]
        assert got.data.hex() == want["data_hex"]


def test_decode_tensors_truncated():
    blob = (FIXTURES / "wire_tensor_case1.bin").read_bytes()
    with pytest.raises(wire.WireError):
        wire.decode_tensors(blob[8:20])
    with pytest.raises(wire.WireError):
        wire.decode_tensors(blob[8:-4])


# ------------------------------------------------------------- config: allow_dev

def _dev_cfg(tmp_path, **dev):
    base = json.loads((FIXTURES / "config_valid" / "synthetic.json").read_text())
    base["dev"] = dev or {"raw_tensors": True, "max_fps": 1.0, "max_streams": 1}
    p = tmp_path / "dev.json"
    p.write_text(json.dumps(base))
    return str(p)


def test_raw_tensors_rejected_without_allow_dev(tmp_path):
    p = _dev_cfg(tmp_path)
    with pytest.raises(ConfigError) as excinfo:
        load(p)
    assert str(excinfo.value) == \
        "dev.raw_tensors is not allowed in production configs"


def test_raw_tensors_allowed_with_allow_dev(tmp_path):
    p = _dev_cfg(tmp_path)
    cfg = load(p, allow_dev=True)
    assert cfg.dev["raw_tensors"] is True
    assert cfg.dev["max_fps"] == 1.0
    assert cfg.dev["max_streams"] == 1
    # runtime config forwards `dev` to the native child in dev mode (§6.12)
    rt = runtime_config(cfg, 0)
    assert rt["dev"] == {"raw_tensors": True, "max_fps": 1.0, "max_streams": 1}


def test_dev_defaults_omitted(tmp_path):
    base = json.loads((FIXTURES / "config_valid" / "synthetic.json").read_text())
    p = tmp_path / "nodev.json"
    p.write_text(json.dumps(base))
    cfg = load(str(p))
    assert cfg.dev == {"raw_tensors": False, "max_fps": 1.0, "max_streams": 1}
    assert "dev" not in runtime_config(cfg, 0)


@pytest.mark.parametrize("dev, path", [
    ({"raw_tensors": "yes"}, "dev.raw_tensors"),
    ({"raw_tensors": False, "max_fps": 0}, "dev.max_fps"),
    ({"raw_tensors": False, "max_fps": 2.5}, "dev.max_fps"),
    ({"raw_tensors": False, "max_fps": -1}, "dev.max_fps"),
    ({"raw_tensors": False, "max_streams": 2}, "dev.max_streams"),
    ({"raw_tensors": False, "bogus": 1}, "dev.bogus"),
    ("nope", "dev"),
])
def test_dev_validation_errors(tmp_path, dev, path):
    p = _dev_cfg(tmp_path, **dev) if isinstance(dev, dict) else None
    if p is None:  # non-dict dev value
        base = json.loads((FIXTURES / "config_valid" / "synthetic.json").read_text())
        base["dev"] = dev
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(base))
        p = str(p)
    with pytest.raises(ConfigError) as excinfo:
        load(p, allow_dev=True)
    assert path in str(excinfo.value)


def test_raw_decoder_requires_dev_raw_tensors(tmp_path):
    """§6.11: decoder.type=raw only valid with dev.raw_tensors=true."""
    base = json.loads((FIXTURES / "config_valid" / "synthetic.json").read_text())
    base["backend"]["decoder"] = {"type": "raw"}
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(base))
    with pytest.raises(ConfigError) as excinfo:
        load(str(p), allow_dev=True)
    assert "raw decoder requires dev.raw_tensors" in str(excinfo.value)


# ------------------------------------------------------------- main: --dev

def test_main_dev_production_env(tmp_path, monkeypatch):
    """VB_PRODUCTION=1 + --dev -> exit code 2."""
    monkeypatch.setenv("VB_PRODUCTION", "1")
    p = _dev_cfg(tmp_path)
    rc = main(["--config", p, "--dev", "--validate"])
    assert rc == 2


def test_main_dev_allows_dev_config(tmp_path, monkeypatch):
    monkeypatch.delenv("VB_PRODUCTION", raising=False)
    p = _dev_cfg(tmp_path)
    rc = main(["--config", p, "--dev", "--validate"])
    assert rc == 0


def test_main_without_dev_rejects_raw_tensors(tmp_path, monkeypatch):
    monkeypatch.delenv("VB_PRODUCTION", raising=False)
    p = _dev_cfg(tmp_path)
    rc = main(["--config", p, "--validate"])
    assert rc == 2


# ------------------------------------------------------------- shard dispatch

class FakePublisher:
    def __init__(self):
        self.lock = threading.Lock()
        self.messages = []

    def submit(self, topic, payload, qos=0, retain=False):
        with self.lock:
            self.messages.append((topic, payload, qos, retain))


class TensorApp:
    """Dev-mode app with the optional on_tensors hook (§6.12)."""

    name = "tensorapp"
    wants_frames = False

    def __init__(self):
        self.lock = threading.Lock()
        self.tensors = []
        self.states = []

    def analyzers(self, spec):
        return []

    def plugins(self):
        return []

    def on_stream_added(self, ctx):
        pass

    def on_stream_removed(self, ctx):
        pass

    def on_event(self, ctx, ev):
        return []

    def on_frame(self, ctx, res):
        return []

    def on_stream_state(self, ctx, state, error):
        with self.lock:
            self.states.append((ctx.stream_id, state, error))
        return []

    def on_tensors(self, ctx, tf):
        with self.lock:
            self.tensors.append((ctx.stream_id, tf))
        return []


def _make_shard(app, env, dev=True):
    old = {}
    for k, v in env.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    argv = FAKE_ARGV + ["--dev"] if dev else FAKE_ARGV
    shard = Shard(0, app, FakePublisher(), runtime_argv=argv,
                  runtime_cfg=RT_CFG, state_dir="/tmp/vb-test-dev",
                  topic_root="vb", restart_backoff_s=0.3, hello_timeout_s=5.0)
    return shard, old


def _cleanup_env(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_fake_runtime_vbt1_reaches_on_tensors_hook():
    """fake_runtime sends VBT1 -> hook on_tensors receives the TensorFrame."""
    app = TensorApp()
    shard, old = _make_shard(app, {"FAKE_DEV_TENSORS": "1", "FAKE_FPS": "5"})
    try:
        shard.start()
        shard.add_stream(StreamSpec("cam-0", "fake://"))
        wait_for(lambda: app.tensors, what="on_tensors delivery")
        sid, tf = app.tensors[0]
        assert sid == "cam-0"
        assert tf.stream_id == "cam-0"
        assert tf.seq == 1
        assert tf.geom.src_w == 320 and tf.geom.model_w == 416
        assert tf.geom.align == "center"
        assert len(tf.tensors) == 1
        t = tf.tensors[0]
        assert (t.name, t.dtype, t.dims, t.nhwc) == ("out0", 0, (2, 2), False)
        assert t.data == struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    finally:
        shard.stop()
        _cleanup_env(old)


@pytest.mark.parametrize("case", ["no_dev_flag", "no_hook"])
def test_vbt1_outside_dev_mode_closes_connection(case):
    """VBT1 received outside dev mode (runtime started without --dev, even if
    the app defines on_tensors; or no on_tensors hook) -> connection closed
    and error recorded (§6.12 'unknown magic' rule)."""
    class PlainApp(TensorApp):
        on_tensors = None

    app = TensorApp() if case == "no_dev_flag" else PlainApp()
    shard, old = _make_shard(app, {"FAKE_DEV_TENSORS": "1", "FAKE_FPS": "5"},
                             dev=(case != "no_dev_flag"))
    try:
        shard.start()
        client = shard._client
        added = True
        try:
            shard.add_stream(StreamSpec("cam-0", "fake://"))
        except RuntimeGone:
            # the record closes the socket, and whether the add reply was
            # consumed before the reader tears the waiters down is a race
            added = False
        assert client is not None
        wait_for(lambda: client.last_wire_error != "",
                 what="wire error recorded")
        assert client.last_wire_error == \
            "unexpected VBT1 record outside dev mode"
        if added:
            # the add completed, so the runtime's error is that stream's state
            wait_for(lambda: any(s[1] == "error" for s in app.states),
                     what="error state delivered")
        else:
            # the add failed along with the connection: the stream is rolled
            # back (review item 7), so it holds no slot in the shard
            assert shard.stream_status() == []
            assert shard.contexts == {}
        # connection closed: reader loop hits EOF, client reports exit
        wait_for(lambda: shard._client is None or shard.runtime_restarts >= 1,
                 what="connection closed after VBT1 outside dev mode")
    finally:
        shard.stop()
        _cleanup_env(old)
