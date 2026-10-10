"""Base config parsing/validation and per-shard runtime config generation.

Spec BASE-1 §6.6 (schema = ``contracts/vb-config.schema.json``) and §6.9
rule 4 (per-shard runtime config content). Standard library only.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass, field

SCHEMA_ID = "vb.config/1"
STREAM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DECODER_TYPES = {"yolox", "yolov8", "yolov8_dfl", "yolo_pose", "classify", "raw"}
DECODER_KEYS = {"type", "num_classes", "reg_max", "strides", "keypoints",
                "top_k", "softmax"}
TOP_LEVEL_KEYS = {
    "schema", "device_id", "backend", "runtime", "native", "state_dir",
    "tracker", "analyzers", "mqtt", "health", "streams_file", "streams", "app",
    "dev",
}

DEFAULTS: dict = {
    "backend": {"input_size": [640, 640], "align": "center",
                "score_threshold": 0.35, "nms_threshold": 0.45, "options": {}},
    "runtime": {"workers": "auto", "max_streams_per_worker": 0,
                "contexts_per_worker": 1, "publishers_per_process": 1,
                "publish_queue": 256, "max_streams": 16, "open_timeout_s": 8.0,
                "reconnect_delay_s": 1.0, "restart_backoff_s": 5.0},
    "native": {"binary": "/opt/vb/bin/vb-runtime", "hello_timeout_s": 10.0,
               "snapshot_ring": 2, "jpeg_quality": 85},
    "state_dir": "/data/vb",
    "tracker": {"enabled": True, "iou_threshold": 0.2, "dist_threshold": 0.15,
                "max_misses": 15, "max_lost_s": 0.0, "min_hits": 1,
                "class_aware": True, "anchor": "center"},
    "analyzers": {"plugins": []},
    "mqtt": {"port": 1883, "username": "", "password": "", "tls": False,
             "ca_file": "", "cert_file": "", "key_file": "", "keepalive_s": 30, "status_interval_s": 10},
    # §6.8: /healthz is unauthenticated, so it must not be reachable from the
    # network by default — an explicit host is required to expose it
    # (review item 20; the port number itself is unchanged).
    "health": {"host": "127.0.0.1", "port": 8099},
    "streams_file": "",
    "streams": [],
    "app": {"name": "echo", "options": {"publish_hz": 1.0, "frame_stride": 1}},
    "dev": {"raw_tensors": False, "max_fps": 1.0, "max_streams": 1},
}


class ConfigError(Exception):
    """Invalid configuration; message always contains the field path."""


@dataclass
class BaseConfig:
    schema: str
    device_id: str
    backend: dict
    runtime: dict
    native: dict
    state_dir: str
    tracker: dict
    analyzers: dict
    mqtt: dict
    health: dict
    streams_file: str
    streams: list
    app: dict
    dev: dict = field(default_factory=dict)
    source_path: str = ""


def _err(path: str, msg: str) -> None:
    raise ConfigError(f"{path}: {msg}")


def _merge(defaults: dict, override: dict) -> dict:
    out = dict(defaults)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _require(container: dict, key: str, path: str) -> None:
    if key not in container:
        _err(path, "required")


def _check_str(value, path: str, *, nonempty: bool = True) -> None:
    if not isinstance(value, str):
        _err(path, "must be a string")
    if nonempty and not value:
        _err(path, "must not be empty")


def _check_num(value, path: str, *, lo=None, hi=None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _err(path, "must be a number")
    if lo is not None and value < lo:
        _err(path, f"must be >= {lo}")
    if hi is not None and value > hi:
        _err(path, f"must be <= {hi}")


def _check_int(value, path: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _err(path, "must be an integer")
    if value < minimum:
        _err(path, f"must be >= {minimum}")


def _check_threshold(value, path: str) -> None:
    _check_num(value, path, lo=0.0, hi=1.0)


def _check_plugin_path(value, path: str) -> None:
    """§6.6 `analyzers.plugins[]`: absolute path to a plugin .so."""
    _check_str(value, path)
    if not os.path.isabs(value):
        _err(path, "must be an absolute path")
    if not value.endswith(".so"):
        _err(path, "must end with '.so'")


def merge_plugins(config_plugins, app_plugins, *,
                  path: str = "app.plugins") -> list[str]:
    """§6.5: the application's ``plugins()`` joins ``analyzers.plugins``.

    The merged list is what goes into every per-shard runtime config
    (``runtime_config``), so both sources are validated identically and
    duplicates collapse. Report item 15: the orchestration layer used to
    read ``cfg.analyzers.plugins`` only, so an application that loaded its
    own analyzer plugin silently ran without it.
    """
    out = list(config_plugins or [])
    for i, p in enumerate(app_plugins or []):
        _check_plugin_path(p, f"{path}[{i}]")
        if p not in out:
            out.append(p)
    return out


# §6.2 built-in analyzer names (§6.6 streams[].options.analyzers validation, M1.20).
# Must match create_analyzer() in core-cpp/vb/src/analyzers/plugin_loader.cpp:
# a name accepted here but unknown to vb-runtime passes load() and then fails
# every add, which the supervisor retries forever.
BUILTIN_ANALYZERS = frozenset({
    "line_cross", "zone", "dwell", "speed", "direction", "count_threshold",
    "pose_angle",
})
# Named in BASE-1 §6.2 but not implemented by this vb-runtime.
UNIMPLEMENTED_ANALYZERS = frozenset({"slot_coverage", "stage2", "text_vote"})


def _check_options(options: dict, path: str) -> None:
    """§6.6.1: max_fps / roi_crop are base-interpreted stream options."""
    if not isinstance(options, dict):
        _err(path, "must be an object")
    if "max_fps" in options:
        _check_num(options["max_fps"], f"{path}.max_fps", lo=0.0)
        if not math.isfinite(float(options["max_fps"])):
            _err(f"{path}.max_fps", "must be finite")
    if "roi_crop" in options:
        roi = options["roi_crop"]
        if roi is not None:
            if (not isinstance(roi, list) or len(roi) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           for v in roi)):
                _err(f"{path}.roi_crop", "must be null or [x0, y0, x1, y1] numbers")
            x0, y0, x1, y1 = roi
            if any(not math.isfinite(float(v)) for v in roi):
                _err(f"{path}.roi_crop", "coordinates must be finite")
            if not (0 <= x0 < x1 <= 1) or not (0 <= y0 < y1 <= 1):
                _err(f"{path}.roi_crop", "requires 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
            if x1 - x0 < 0.05 or y1 - y0 < 0.05:
                _err(f"{path}.roi_crop", "crop sides must be >= 0.05")
    if "analyzers" in options:   # §6.6 / M1.20: shape + name check only
        an = options["analyzers"]
        if not isinstance(an, list):
            _err(f"{path}.analyzers", "must be an array")
        for i, a in enumerate(an):
            p = f"{path}.analyzers[{i}]"
            if not isinstance(a, dict):
                _err(p, "must be an object")
            name = a.get("name")
            if not isinstance(name, str) or not name:
                _err(f"{p}.name", "required")
            if name in UNIMPLEMENTED_ANALYZERS:
                _err(f"{p}.name", f"analyzer {name!r} is not implemented by "
                     "this vb-runtime")
            if not (name in BUILTIN_ANALYZERS or name.startswith("plugin:")):
                _err(f"{p}.name", f"unknown analyzer {name!r}")
            if "config" not in a:
                _err(f"{p}.config", "required")
            if not isinstance(a["config"], dict):
                _err(f"{p}.config", "must be an object")


def _check_decoder(backend: dict, tracker: dict, data: dict) -> None:
    """§6.11: `backend.decoder` shape (interpreted by vb-runtime, not Python)."""
    dec = backend.get("decoder")
    if dec is None:
        return
    if not isinstance(dec, dict):
        _err("backend.decoder", "must be an object")
    for k in dec:
        if k not in DECODER_KEYS:
            _err(f"backend.decoder.{k}", "unknown decoder key")
    t = dec.get("type")
    if not isinstance(t, str) or not t:
        _err("backend.decoder.type", "required, must be a string")
    if t == "ctc":
        _err("backend.decoder", "ctc is only valid in stage2")
    if t not in DECODER_TYPES:
        _err("backend.decoder.type",
             f"must be one of {sorted(DECODER_TYPES)}, got {t!r}")
    if t == "raw" and not (data.get("dev") or {}).get("raw_tensors"):
        _err("backend.decoder", "raw decoder requires dev.raw_tensors=true")
    _check_int(dec.get("num_classes", 80), "backend.decoder.num_classes", minimum=1)
    _check_int(dec.get("reg_max", 16), "backend.decoder.reg_max", minimum=1)
    strides = dec.get("strides", [8, 16, 32])
    if (not isinstance(strides, list) or not strides
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in strides)):
        _err("backend.decoder.strides", "must be a non-empty array of positive integers")
    _check_int(dec.get("keypoints", 17), "backend.decoder.keypoints", minimum=0)
    _check_int(dec.get("top_k", 1), "backend.decoder.top_k", minimum=1)
    if not isinstance(dec.get("softmax", True), bool):
        _err("backend.decoder.softmax", "must be a boolean")
    if t == "classify" and tracker.get("enabled", True):
        _err("backend.decoder", "classify decoder requires tracker.enabled=false")


def _check_dev(data: dict, *, allow_dev: bool) -> None:
    """§6.12 top-level `dev` object (dev-mode raw tensor passthrough)."""
    dev = data.get("dev")
    if dev is None:
        return
    if not isinstance(dev, dict):
        _err("dev", "must be an object")
    for k in dev:
        if k not in ("raw_tensors", "max_fps", "max_streams"):
            _err(f"dev.{k}", "unknown dev key")
    merged = _merge(DEFAULTS["dev"], dev)
    if not isinstance(merged["raw_tensors"], bool):
        _err("dev.raw_tensors", "must be a boolean")
    if merged["raw_tensors"] and not allow_dev:
        raise ConfigError("dev.raw_tensors is not allowed in production configs")
    _check_num(merged["max_fps"], "dev.max_fps")
    if not 0 < merged["max_fps"] <= 2:
        _err("dev.max_fps", "must be in (0, 2]")
    _check_int(merged["max_streams"], "dev.max_streams", minimum=1)
    if merged["max_streams"] != 1:
        _err("dev.max_streams", "must be 1")


_INPUT_COLOR_ORDERS = ("bgr", "rgb")


def _check_input(backend: dict) -> None:
    """backend.input (§6.11): only what the model actually expects.

    Absent, each backend resolves it from the decoder family — yolox/raw take
    BGR 0-255, the Ultralytics families take RGB 0-1. Declaring it here
    overrides that; it is never inferred from the frame format.
    """
    if not isinstance(backend, dict):
        return
    inp = backend.get("input")
    if inp is None:
        return
    if not isinstance(inp, dict):
        _err("backend.input", "must be an object")
    unknown = set(inp) - {"color_order", "divide"}
    if unknown:
        _err("backend.input", f"unknown key(s): {sorted(unknown)}")
    if "color_order" in inp and inp["color_order"] not in _INPUT_COLOR_ORDERS:
        _err("backend.input.color_order", 'must be "bgr" or "rgb"')
    if "divide" in inp:
        d = inp["divide"]
        if not isinstance(d, (int, float)) or isinstance(d, bool) or d <= 0:
            _err("backend.input.divide", "must be a number > 0")


def _validate(data: dict, *, partial: bool = False, allow_dev: bool = False) -> None:
    if not isinstance(data, dict):
        _err("$", "top level must be an object")
    for key in data:
        if key not in TOP_LEVEL_KEYS:
            _err(key, "unknown top-level key")
    _check_dev(data, allow_dev=allow_dev)

    _require(data, "schema", "schema")
    if data.get("schema") != SCHEMA_ID:
        _err("schema", f"must be {SCHEMA_ID!r}")
    _require(data, "device_id", "device_id")
    _check_str(data.get("device_id"), "device_id")

    # backend
    _require(data, "backend", "backend")
    backend = data.get("backend")
    if not isinstance(backend, dict):
        _err("backend", "must be an object")
    _require(backend, "name", "backend.name")
    _check_str(backend.get("name"), "backend.name")
    _require(backend, "model_path", "backend.model_path")
    _check_str(backend.get("model_path"), "backend.model_path")
    size = backend.get("input_size", DEFAULTS["backend"]["input_size"])
    if (not isinstance(size, list) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)):
        _err("backend.input_size", "must be two positive integers [w, h]")
    if backend.get("align", "center") not in ("center", "top_left"):
        _err("backend.align", "must be 'center' or 'top_left'")
    # The keys below stay in the vb.config/1 surface, but this vb-runtime
    # implements only one value: every backend letterboxes with Align::Center
    # and the inference pool runs NMS at a fixed 0.45. Reject anything else
    # instead of silently running with a different setting.
    if backend.get("align", "center") != "center":
        _err("backend.align", f"{backend['align']!r} is not supported by this "
             "vb-runtime (only 'center')")
    _check_threshold(backend.get("score_threshold", 0.35), "backend.score_threshold")
    _check_threshold(backend.get("nms_threshold", 0.45), "backend.nms_threshold")
    if float(backend.get("nms_threshold", 0.45)) != 0.45:
        _err("backend.nms_threshold", f"{backend['nms_threshold']!r} is not "
             "supported by this vb-runtime (only 0.45)")
    if not isinstance(backend.get("options", {}), dict):
        _err("backend.options", "must be an object")

    # runtime
    runtime = _merge(DEFAULTS["runtime"], data.get("runtime", {}))
    workers = runtime["workers"]
    if workers != "auto":
        _check_int(workers, "runtime.workers", minimum=1)
    _check_int(runtime["max_streams_per_worker"], "runtime.max_streams_per_worker", minimum=0)
    _check_int(runtime["contexts_per_worker"], "runtime.contexts_per_worker", minimum=1)
    _check_int(runtime["publishers_per_process"], "runtime.publishers_per_process", minimum=1)
    _check_int(runtime["publish_queue"], "runtime.publish_queue", minimum=1)
    _check_int(runtime["max_streams"], "runtime.max_streams", minimum=1)
    _check_num(runtime["open_timeout_s"], "runtime.open_timeout_s", lo=0.0)
    _check_num(runtime["reconnect_delay_s"], "runtime.reconnect_delay_s", lo=0.0)
    _check_num(runtime["restart_backoff_s"], "runtime.restart_backoff_s", lo=0.0)

    # native
    native = _merge(DEFAULTS["native"], data.get("native", {}))
    _check_str(native["binary"], "native.binary")
    if not os.path.isabs(native["binary"]):
        _err("native.binary", "must be an absolute path")
    _check_num(native["hello_timeout_s"], "native.hello_timeout_s", lo=0.0)
    _check_int(native["snapshot_ring"], "native.snapshot_ring", minimum=1)
    q = native["jpeg_quality"]
    _check_int(q, "native.jpeg_quality", minimum=1)
    # vb-runtime encodes snapshots at a fixed quality of 85.
    if 1 <= q <= 100 and q != 85:
        _err("native.jpeg_quality", f"{q!r} is not supported by this "
             "vb-runtime (only 85)")
    if q > 100:
        _err("native.jpeg_quality", "must be <= 100")

    if "state_dir" in data:
        _check_str(data["state_dir"], "state_dir")

    # tracker
    tracker = _merge(DEFAULTS["tracker"], data.get("tracker", {}))
    if not isinstance(tracker["enabled"], bool):
        _err("tracker.enabled", "must be a boolean")
    _check_threshold(tracker["iou_threshold"], "tracker.iou_threshold")
    _check_threshold(tracker["dist_threshold"], "tracker.dist_threshold")
    _check_int(tracker["max_misses"], "tracker.max_misses", minimum=0)
    _check_num(tracker["max_lost_s"], "tracker.max_lost_s", lo=0.0)
    _check_int(tracker["min_hits"], "tracker.min_hits", minimum=1)
    if not isinstance(tracker["class_aware"], bool):
        _err("tracker.class_aware", "must be a boolean")
    if tracker["anchor"] not in ("center", "bottom_center"):
        _err("tracker.anchor", "must be 'center' or 'bottom_center'")
    _check_decoder(backend, tracker, data)

    # analyzers
    analyzers = _merge(DEFAULTS["analyzers"], data.get("analyzers", {}))
    plugins = analyzers["plugins"]
    if not isinstance(plugins, list):
        _err("analyzers.plugins", "must be an array")
    for i, p in enumerate(plugins):
        _check_plugin_path(p, f"analyzers.plugins[{i}]")

    # mqtt
    mqtt = _merge(DEFAULTS["mqtt"], data.get("mqtt", {}))
    if not partial:
        _require(data, "mqtt", "mqtt")
        _require(mqtt, "host", "mqtt.host")
        _check_str(mqtt["host"], "mqtt.host")
        _require(mqtt, "topic_root", "mqtt.topic_root")
        _check_str(mqtt["topic_root"], "mqtt.topic_root")
    _check_int(mqtt["port"], "mqtt.port", minimum=1)
    if mqtt["port"] > 65535:
        _err("mqtt.port", "must be <= 65535")
    _check_str(mqtt.get("client_id", ""), "mqtt.client_id", nonempty=False)
    _check_input(backend)

    _check_num(mqtt["keepalive_s"], "mqtt.keepalive_s", lo=1.0)
    _check_num(mqtt["status_interval_s"], "mqtt.status_interval_s", lo=0.1)
    if not partial:
        _require(data, "app", "app")
    app = _merge(DEFAULTS["app"], data.get("app", {}))
    if "module" in app or not partial:
        if "module" not in app:
            _err("app.module", "required")
        _check_str(app["module"], "app.module")
    if "options" in app:
        if not isinstance(app["options"], dict):
            _err("app.options", "must be an object")
        if "frame_every" in app["options"]:   # §6.6, M1.20
            _check_int(app["options"]["frame_every"], "app.options.frame_every",
                       minimum=0)

    # health
    health = _merge(DEFAULTS["health"], data.get("health", {}))
    _check_str(health["host"], "health.host")
    _check_int(health["port"], "health.port", minimum=1)
    if health["port"] > 65535:
        _err("health.port", "must be <= 65535")

    if "streams_file" in data:
        _check_str(data["streams_file"], "streams_file", nonempty=False)

    # streams
    streams = data.get("streams", [])
    if not isinstance(streams, list):
        _err("streams", "must be an array")
    seen = set()
    for i, s in enumerate(streams):
        if not isinstance(s, dict):
            _err(f"streams[{i}]", "must be an object")
        _require(s, "stream_id", f"streams[{i}].stream_id")
        _require(s, "url", f"streams[{i}].url")
        _check_str(s["stream_id"], f"streams[{i}].stream_id")
        if not STREAM_ID_RE.match(s["stream_id"]):
            _err(f"streams[{i}].stream_id", "must match ^[A-Za-z0-9_-]{1,64}$")
        if s["stream_id"] in seen:
            _err(f"streams[{i}].stream_id", f"duplicate stream_id {s['stream_id']!r}")
        seen.add(s["stream_id"])
        _check_str(s["url"], f"streams[{i}].url")
        if s.get("transport", "tcp") not in ("tcp", "udp"):
            _err(f"streams[{i}].transport", "must be 'tcp' or 'udp'")
        if "score_threshold" in s and s["score_threshold"] is not None:
            _check_threshold(s["score_threshold"], f"streams[{i}].score_threshold")
        _check_options(s.get("options", {}), f"streams[{i}].options")
    # Explicit workers cannot hold more than workers * max_streams_per_worker
    # streams; reject instead of silently overfilling shards (M1.13 review).
    per_worker = runtime["max_streams_per_worker"]
    if workers != "auto" and per_worker > 0 and len(streams) > workers * per_worker:
        _err("runtime.workers",
             f"{workers} workers x max_streams_per_worker {per_worker} "
             f"< {len(streams)} streams")


def load(path: str, *, partial: bool = False, allow_dev: bool = False) -> BaseConfig:
    """Load, merge defaults, validate and return a BaseConfig.

    ``partial=True`` (§6.6): do not require ``mqtt``/``app`` — for
    ``vision_base.embed`` (§6.13.1). ``allow_dev=True`` (§6.12): permit
    ``dev.raw_tensors=true`` (dev mode only, e.g. ``main --dev``).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _validate(data, partial=partial, allow_dev=allow_dev)
    merged = _merge(DEFAULTS, data)
    # streams_file (if the file exists) overrides `streams` (§6.6).
    if merged["streams_file"] and os.path.exists(merged["streams_file"]):
        with open(merged["streams_file"], "r", encoding="utf-8") as f:
            streams = json.load(f)
        if not isinstance(streams, list):
            _err("streams_file", "content must be an array of stream objects")
        merged["streams"] = streams
        _validate(merged, partial=partial, allow_dev=allow_dev)
    return BaseConfig(
        schema=merged["schema"], device_id=merged["device_id"],
        backend=merged["backend"], runtime=merged["runtime"],
        native=merged["native"], state_dir=merged["state_dir"],
        tracker=merged["tracker"], analyzers=merged["analyzers"],
        mqtt=merged["mqtt"], health=merged["health"],
        streams_file=merged["streams_file"], streams=merged["streams"],
        app=merged["app"], dev=merged["dev"], source_path=str(path),
    )


def runtime_config(cfg: BaseConfig, shard_index: int) -> dict:
    """Per-shard runtime config (§6.9 rule 4): no `streams` key.

    Streams are always delivered via `add` control lines, never via this file.
    """
    _check_int(shard_index, "shard_index", minimum=0)
    out = {
        "backend": copy.deepcopy(cfg.backend),
        "contexts_per_worker": cfg.runtime["contexts_per_worker"],
        "tracker": copy.deepcopy(cfg.tracker),
        "analyzers": {"plugins": list(cfg.analyzers.get("plugins", []))},
        "snapshot_ring": cfg.native.get("snapshot_ring", 2),
        "open_timeout_s": cfg.runtime["open_timeout_s"],
    }
    if cfg.dev.get("raw_tensors"):
        # §6.12: the native child reads dev limits (max_fps / max_streams)
        # from this file; include `dev` only in dev mode.
        out["dev"] = copy.deepcopy(cfg.dev)
    return out
