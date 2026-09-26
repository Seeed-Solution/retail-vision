"""Control decision table tests (M1.11).

Covers: no-ack cases (wrong schema/device, empty request_id), unknown
command, bad params, per-command decision rules via a fake ops object,
request-id dedup (redelivery returns stored ack, does not re-execute, LRU
256 bound), unexpected ops exception -> ok=false, and ack schema conformance
of every generated ack against contracts/vb-ack.schema.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import jsonschema

from vision_base.control import CommandError, ControlPlane

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / "contracts" / "fixtures" / "vb"
ACK_SCHEMA = json.loads(
    (FIXTURES.parent.parent / "vb-ack.schema.json").read_text())

SESSION = "1790000000000"
DEV = "rk3576-demo-01"


class Ops:
    """Fake supervisor-side operations recording every executed command."""

    def __init__(self):
        self.calls = []
        self.streams = {"cam-01"}
        self.max_streams = 2
        self.explode_on = None

    def _tick(self, name, params):
        self.calls.append((name, params))
        if self.explode_on == name:
            raise RuntimeError("boom")

    def add_stream(self, params):
        if params["stream_id"] in self.streams:
            raise CommandError(f"stream already exists: {params['stream_id']}")
        if len(self.streams) >= self.max_streams:
            raise CommandError(f"at most {self.max_streams} streams")
        self._tick("add_stream", params)
        self.streams.add(params["stream_id"])
        return {"stream_id": params["stream_id"], "url": params["url"],
                "shard": 0, "decode_path": "gpu",
                "score_threshold": params.get("score_threshold", 0.35),
                "persisted": True}

    def remove_stream(self, params):
        if params["stream_id"] not in self.streams:
            raise CommandError(f"no such stream: {params['stream_id']}")
        self._tick("remove_stream", params)
        self.streams.discard(params["stream_id"])
        return {"stream_id": params["stream_id"], "removed": True, "persisted": True}

    def set_threshold(self, params):
        if params["stream_id"] not in self.streams:
            raise CommandError(f"no such stream: {params['stream_id']}")
        self._tick("set_threshold", params)
        return {"stream_id": params["stream_id"],
                "score_threshold": params["score_threshold"]}

    def list_streams(self, params):
        self._tick("list_streams", params)
        return {"streams": sorted(self.streams)}


def cmd(request_id, command, params=None, **extra):
    req = {"schema": "vb.command/1", "timestamp": 1790000000000,
           "device_id": DEV, "request_id": request_id, "command": command,
           "params": params if params is not None else {}}
    req.update(extra)
    return req


def handle(ops, req, now_ms=1790000000000):
    return ControlPlane(DEV).handle(req, SESSION, now_ms, ops)


def test_no_ack_for_wrong_schema_or_device():
    ops = Ops()
    assert handle(ops, {"schema": "vb.command/2", "device_id": DEV, "request_id": "r",
                        "command": "list_streams", "params": {}}) is None
    assert handle(ops, cmd("r", "list_streams", device_id="other-dev")) is None
    assert handle(ops, "not-a-dict") is None
    assert ops.calls == []


def test_no_ack_for_empty_request_id(caplog):
    ops = Ops()
    assert handle(ops, cmd("", "list_streams")) is None
    assert ops.calls == []


def test_unknown_command():
    ops = Ops()
    ack = handle(ops, cmd("r1", "restart"))
    assert ack["ok"] is False and ack["request_id"] == "r1"
    assert "unknown command" in ack["error"]
    assert ack["command"] == "restart"
    jsonschema.validate(ack, ACK_SCHEMA)


def test_params_not_object():
    ops = Ops()
    ack = handle(ops, cmd("r2", "add_stream", "oops"))
    assert ack["ok"] is False
    assert ack["error"] == "params must be an object"


def test_add_stream_missing_fields():
    ops = Ops()
    ack = handle(ops, cmd("r3", "add_stream", {"stream_id": "cam-02"}))
    assert ack["ok"] is False and "params.url" in ack["error"]
    ack = handle(ops, cmd("r4", "add_stream", {"url": "rtsp://x"}))
    assert ack["ok"] is False and "stream_id" in ack["error"]


def test_add_stream_duplicate_and_capacity():
    ops = Ops()  # starts with cam-01, max 2
    ack = handle(ops, cmd("r5", "add_stream", {"stream_id": "cam-01", "url": "rtsp://x"}))
    assert ack["ok"] is False and "already exists" in ack["error"]
    ack = handle(ops, cmd("r6", "add_stream", {"stream_id": "cam-02", "url": "rtsp://y"}))
    assert ack["ok"] is True  # 2nd of at most 2
    ack = handle(ops, cmd("r6b", "add_stream", {"stream_id": "cam-03", "url": "rtsp://z"}))
    assert ack["ok"] is False and "at most 2 streams" in ack["error"]


def test_add_stream_ok_applied_shape():
    ops = Ops()
    ops.streams.clear()
    ack = handle(ops, cmd("r7", "add_stream",
                          {"stream_id": "cam-02", "url": "rtsp://y", "score_threshold": 0.5}))
    assert ack["ok"] is True
    assert ack["applied"]["stream_id"] == "cam-02"
    assert ack["applied"]["persisted"] is True
    jsonschema.validate(ack, ACK_SCHEMA)


def test_remove_stream_missing():
    ops = Ops()
    ack = handle(ops, cmd("r8", "remove_stream", {"stream_id": "nope"}))
    assert ack["ok"] is False and "no such stream" in ack["error"]


def test_remove_stream_ok():
    ops = Ops()
    ack = handle(ops, cmd("r9", "remove_stream", {"stream_id": "cam-01"}))
    assert ack["ok"] is True
    assert ack["applied"] == {"stream_id": "cam-01", "removed": True, "persisted": True}


def test_set_threshold_range_and_target():
    ops = Ops()
    ack = handle(ops, cmd("r10", "set_threshold",
                          {"stream_id": "cam-01", "score_threshold": 1.5}))
    assert ack["ok"] is False and "[0,1]" in ack["error"]
    ack = handle(ops, cmd("r11", "set_threshold",
                          {"stream_id": "ghost", "score_threshold": 0.5}))
    assert ack["ok"] is False
    ack = handle(ops, cmd("r12", "set_threshold",
                          {"stream_id": "cam-01", "score_threshold": 0.7}))
    assert ack["ok"] is True
    assert ack["applied"] == {"stream_id": "cam-01", "score_threshold": 0.7}


def test_list_streams():
    ops = Ops()
    ack = handle(ops, cmd("r13", "list_streams"))
    assert ack["ok"] is True and ack["applied"] == {"streams": ["cam-01"]}


def test_unexpected_exception_becomes_error_ack():
    ops = Ops()
    ops.explode_on = "list_streams"
    ack = handle(ops, cmd("r14", "list_streams"))
    assert ack["ok"] is False
    assert ack["error"] == "RuntimeError: boom"
    jsonschema.validate(ack, ACK_SCHEMA)


def test_request_id_dedup_no_reexecute():
    ops = Ops()
    plane = ControlPlane(DEV)
    req = cmd("dup-1", "remove_stream", {"stream_id": "cam-01"})
    ack1 = plane.handle(req, SESSION, 1000, ops)
    assert ack1["ok"] is True
    ack2 = plane.handle(req, SESSION, 9999, ops)  # redelivery
    assert ack2 == dict(ack1, timestamp=9999)
    assert len(ops.calls) == 1  # executed once
    # failing acks are deduped too
    req2 = cmd("dup-2", "remove_stream", {"stream_id": "cam-01"})  # already removed
    a = plane.handle(req2, SESSION, 1000, ops)
    assert a["ok"] is False
    b = plane.handle(req2, SESSION, 1001, ops)
    assert b["ok"] is False and b["timestamp"] == 1001


def test_seen_lru_bound_256():
    ops = Ops()
    plane = ControlPlane(DEV, seen_size=256)
    for i in range(257):
        plane.handle(cmd(f"r-{i}", "list_streams"), SESSION, 1000 + i, ops)
    # r-0 evicted -> re-executed (seen in calls), newer ids not re-executed
    n_calls = len(ops.calls)
    plane.handle(cmd("r-0", "list_streams"), SESSION, 9999, ops)
    assert len(ops.calls) == n_calls + 1
    # r-1 became the LRU when r-0 was re-inserted; r-2 is still deduped
    plane.handle(cmd("r-2", "list_streams"), SESSION, 9999, ops)
    assert len(ops.calls) == n_calls + 1


def test_all_generated_acks_pass_schema():
    ops = Ops()
    ops.streams.clear()
    plane = ControlPlane(DEV)
    reqs = [
        cmd("s-1", "add_stream", {"stream_id": "a", "url": "u"}),
        cmd("s-2", "add_stream", "bad"),
        cmd("s-3", "remove_stream", {"stream_id": "a"}),
        cmd("s-4", "set_threshold", {"stream_id": "a", "score_threshold": 0.9}),
        cmd("s-5", "list_streams"),
        cmd("s-6", "nope"),
    ]
    for req in reqs:
        ack = plane.handle(req, SESSION, 123, ops)
        assert ack is not None
        jsonschema.validate(ack, ACK_SCHEMA)
