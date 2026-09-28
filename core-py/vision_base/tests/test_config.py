"""Config loading/validation tests (M1.2).

- every invalid fixture raises ConfigError whose message contains the field path
- every valid fixture loads and also passes the JSON Schema
- runtime_config output has no `streams` key
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import jsonschema

from vision_base.config import BaseConfig, ConfigError, load, runtime_config

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / "contracts" / "fixtures" / "vb"
SCHEMA_PATH = FIXTURES.parent.parent / "vb-config.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())

# expected field path per invalid fixture (message must contain it)
INVALID_EXPECTED_PATH = {
    "missing_required.json": "mqtt.host",
    "bad_stream_id.json": "streams[0].stream_id",
    "dup_stream_id.json": "streams[1].stream_id",
    "threshold_range.json": "backend.score_threshold",
    "bad_workers.json": "runtime.workers",
    "unknown_key.json": "verbosity",
    "bad_transport.json": "streams[0].transport",
    "relative_binary.json": "native.binary",
    "bad_plugin.json": "analyzers.plugins[0]",
    "bad_anchor.json": "tracker.anchor",
    "bad_roi_crop.json": "streams[0].options.roi_crop",
    "bad_max_fps.json": "streams[0].options.max_fps",
    "missing_topic_root.json": "mqtt.topic_root",
    "workers_capacity.json": "runtime.workers",
    "bad_decoder_type.json": "backend.decoder",
    "classify_with_tracker.json": "backend.decoder",
    "ctc_primary.json": "backend.decoder",
}


def test_all_invalid_fixtures_raise_config_error():
    invalid_dir = FIXTURES / "config_invalid"
    files = sorted(p.name for p in invalid_dir.glob("*.json"))
    assert set(files) == set(INVALID_EXPECTED_PATH), files
    for name, field_path in INVALID_EXPECTED_PATH.items():
        with pytest.raises(ConfigError) as excinfo:
            load(str(invalid_dir / name))
        assert field_path in str(excinfo.value), (name, str(excinfo.value))


def test_valid_fixtures_load_and_pass_schema():
    for path in sorted((FIXTURES / "config_valid").glob("*.json")):
        raw = json.loads(path.read_text())
        jsonschema.validate(instance=raw, schema=SCHEMA)
        cfg = load(str(path))
        assert isinstance(cfg, BaseConfig)
        assert cfg.device_id


def test_runtime_config_has_no_streams():
    cfg = load(str(FIXTURES / "config_valid" / "full.json"))
    rt = runtime_config(cfg, 0)
    assert "streams" not in rt
    # §6.9 rule 4 content
    assert rt["backend"] == cfg.backend
    assert rt["contexts_per_worker"] == cfg.runtime["contexts_per_worker"]
    assert rt["tracker"] == cfg.tracker
    assert rt["analyzers"] == {"plugins": []}
    assert rt["snapshot_ring"] == 2
    rt1 = runtime_config(cfg, 1)
    assert "streams" not in rt1
    with pytest.raises(ConfigError):
        runtime_config(cfg, -1)


def test_decoder_config_validation():
    """M1.15b: exact spec error strings for classify/ctc."""
    invalid = FIXTURES / "config_invalid"
    for name, needle in (("classify_with_tracker.json",
                          "classify decoder requires tracker.enabled=false"),
                         ("ctc_primary.json",
                          "ctc is only valid in stage2")):
        with pytest.raises(ConfigError) as excinfo:
            load(str(invalid / name))
        assert "backend.decoder" in str(excinfo.value)
        assert needle in str(excinfo.value)
    cfg = load(str(FIXTURES / "config_valid" / "decoder_yolov8.json"))
    assert cfg.backend["decoder"] == {"type": "yolov8", "num_classes": 2}


def test_load_partial_allows_missing_mqtt_app(tmp_path):
    """M1.21: partial=True skips mqtt/app requirements (§6.13.1)."""
    p = tmp_path / "embed.json"
    p.write_text(json.dumps({
        "schema": "vb.config/1", "device_id": "dev-1",
        "backend": {"name": "synthetic", "model_path": "/models/m.onnx"}}))
    cfg = load(str(p), partial=True)
    assert cfg.device_id == "dev-1"
    with pytest.raises(ConfigError) as excinfo:
        load(str(p))
    assert "mqtt" in str(excinfo.value)


def test_runtime_config_deep_copy():
    cfg = load(str(FIXTURES / "config_valid" / "full.json"))
    rt = runtime_config(cfg, 0)
    rt["backend"]["score_threshold"] = 0.9
    assert cfg.backend["score_threshold"] == 0.35


def test_streams_file_overrides_streams(tmp_path):
    cfg_raw = json.loads((FIXTURES / "config_valid" / "synthetic.json").read_text())
    streams_file = tmp_path / "streams.json"
    streams_file.write_text(json.dumps(
        [{"stream_id": "file-01", "url": "rtsp://f/1"},
         {"stream_id": "file-02", "url": "rtsp://f/2"}]))
    cfg_raw["streams_file"] = str(streams_file)
    cfg_raw["streams"] = [{"stream_id": "ignored", "url": "rtsp://x"}]
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg_raw))
    cfg = load(str(p))
    assert [s["stream_id"] for s in cfg.streams] == ["file-01", "file-02"]


def test_health_defaults_to_loopback(tmp_path):
    """Review item 20: /healthz is unauthenticated and must not be reachable
    from the network unless the configuration asks for it."""
    from vision_base.config import DEFAULTS
    assert DEFAULTS["health"]["host"] == "127.0.0.1"
    cfg = {"schema": "vb.config/1", "device_id": "d",
           "backend": {"name": "cpu", "model_path": "/m.onnx"},
           "mqtt": {"host": "127.0.0.1", "topic_root": "t"},
           "app": {"module": "vision_base.hooks:EchoApp"},
           "streams": []}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    assert load(str(path)).health["host"] == "127.0.0.1"
    # an explicit host is still honoured (opt-in exposure)
    cfg["health"] = {"host": "0.0.0.0", "port": 8099}
    path.write_text(json.dumps(cfg))
    assert load(str(path)).health["host"] == "0.0.0.0"

# ------------------------------------------------- backend.input (§6.11)

FULL_FIXTURE = (pathlib.Path(__file__).resolve().parents[3] / "contracts" / "fixtures" / "vb" / "config_valid" / "full.json")


def _write_with_input(tmp_path, inp, drop=False):
    cfg = json.loads(FULL_FIXTURE.read_text(encoding="utf-8"))
    if drop:
        cfg["backend"].pop("input", None)
    else:
        cfg["backend"]["input"] = inp
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


@pytest.mark.parametrize("inp", [
    {"color_order": "bgr"}, {"color_order": "rgb"},
    {"divide": 1.0}, {"divide": 255.0},
    {"color_order": "rgb", "divide": 255.0},
])
def test_backend_input_accepts_valid(tmp_path, inp):
    from vision_base.config import load
    load(str(_write_with_input(tmp_path, inp)))


@pytest.mark.parametrize("inp", [
    {"color_order": "gray"},
    {"divide": 0},
    {"divide": -1},
    {"unknown_key": 1},
])
def test_backend_input_rejects_invalid(tmp_path, inp):
    from vision_base.config import load, ConfigError
    with pytest.raises(ConfigError):
        load(str(_write_with_input(tmp_path, inp)))


def test_backend_input_absent_is_fine(tmp_path):
    """It is optional: the backend falls back to the decoder family."""
    from vision_base.config import load
    load(str(_write_with_input(tmp_path, {}, drop=True)))
