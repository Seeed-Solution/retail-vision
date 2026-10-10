"""Fake native runtime process for tests (spec BASE-1 §8 M1.12).

Speaks the §6.3 protocol over an inherited socketpair fd: reads JSON control
lines, answers hello / reply / VBC1 stream_state / VBR1 frames / VBS1
snapshots. Behaviour knobs come from the environment:

- ``FAKE_EXIT_BEFORE_HELLO=1``: exit(1) immediately (pre-hello crash).
- ``FAKE_NO_HELLO=1``: never send hello, just idle (hello-timeout test).
- ``FAKE_CRASH_AFTER=N``: os._exit(1) once more than N control lines were read.
- ``FAKE_FPS``: frame emission rate per stream (default 20).
- ``FAKE_STATS=1``: emit one stats record per second.
- ``FAKE_VERSION``: override the hello ``runtime_version`` (default
  ``0.1.0-fake``); a major/minor mismatch with ``vision_base`` makes
  ``RuntimeClient.start`` raise ``RuntimeError_`` (§6.14, M1.26).
- ``FAKE_REJECT_STREAM=<id>``: answer ``ok:false`` to ``add`` for that
  stream_id (optionally only while ``FAKE_REJECT_FILE`` exists), used to
  exercise add rollback and replay isolation.
- ``FAKE_DEAF=1``: send hello, then stop reading the IPC socket entirely
  (the "child alive but not draining its socket" case).

Usage: ``python fake_runtime.py --ipc-fd N``
"""
from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time

FD = -1
for i, a in enumerate(sys.argv):
    if a == "--ipc-fd" and i + 1 < len(sys.argv):
        FD = int(sys.argv[i + 1])

HDR = struct.Struct("<4sI")
BODY = struct.Struct("<IQd4i3fBBHffBBH")
DET = struct.Struct("<5fiI")

_write_lock = threading.Lock()
_lines_read = 0
_stop = threading.Event()


def send_record(magic: bytes, body: bytes) -> None:
    with _write_lock:
        os.write(FD, HDR.pack(magic, len(body)) + body)


def send_control(obj: dict) -> None:
    send_record(b"VBC1", json.dumps(obj, separators=(",", ":")).encode())


def send_hello() -> None:
    send_control({"op": "hello",
                  "runtime_version": os.environ.get("FAKE_VERSION",
                                                    "0.1.0-fake"),
                  "abi": 1,
                  "backend": "fake", "caps": {"max_contexts": 2, "max_batch": 1,
                                              "keypoints": 0,
                                              "exclusive_device": False},
                  "model_hw": [320, 320], "model_sha256": "0" * 64,
                  "attr_names": [], "stage2_ready": False, "pid": os.getpid()})


def send_reply(req: str, ok: bool, applied=None, error: str = "") -> None:
    obj = {"op": "reply", "req": req, "ok": ok}
    if ok:
        obj["applied"] = applied or {}
    else:
        obj["error"] = error
    send_control(obj)


def send_state(idx: int, state: str, error: str = "") -> None:
    send_control({"op": "stream_state", "stream_index": idx, "state": state,
                  "error": error})


def encode_frame(idx: int, seq: int) -> bytes:
    wall_ms = time.time() * 1000.0
    n = 1
    head = BODY.pack(idx, seq, wall_ms, 1280, 720, 320, 320,
                     0.25, 0.0, 0.0, 0, 0, n, 1.0, 0.5, 0, 0, 0)
    dets = DET.pack(0.5, 0.5 + 0.01 * (seq % 10), 0.2, 0.3, 0.9, 0, seq + 1)
    return head + dets


def encode_vbt1(idx: int, seq: int) -> bytes:
    """One small VBT1 record (§6.12): a single f32 tensor [2, 2]."""
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    t = struct.pack("<BBBB4ifi", 0, 2, 0, 0, 2, 2, 0, 0, 1.0, 0)
    t += struct.pack("<H", 4) + b"out0" + struct.pack("<I", len(data)) + data
    head = struct.pack("<IQd4i3fB3xHH", idx, seq, time.time() * 1000.0,
                       320, 240, 416, 416, 1.3, 0.0, 52.0, 0, 1, 0)
    return head + t


_seq_lock = threading.Lock()
_seqs: dict[int, int] = {}


def emit_frames(idx: int, fps: float) -> None:
    def run():
        sent_event = False
        while not _stop.is_set():
            with _seq_lock:
                _seqs[idx] = _seqs.get(idx, 0) + 1
                seq = _seqs[idx]
            send_record(b"VBR1", encode_frame(idx, seq))
            if not sent_event:
                sent_event = True
                send_record(b"VBE1", json.dumps({
                    "stream_index": idx, "seq": seq,
                    "wall_ms": time.time() * 1000.0,
                    "analyzer": "tick", "type": "tick", "track_id": seq,
                    "fields": {"seq": seq}}).encode())
            if _stop.wait(1.0 / fps):
                return
    threading.Thread(target=run, daemon=True).start()


def handle_line(line: bytes) -> None:
    global _lines_read
    msg = json.loads(line.decode("utf-8"))
    _lines_read += 1
    crash_after = int(os.environ.get("FAKE_CRASH_AFTER", "0") or 0)
    if crash_after and _lines_read > crash_after:
        os._exit(1)
    op = msg.get("op")
    req = msg.get("req", "")
    if op == "add":
        idx = int(msg["stream"]["index"])
        reject = os.environ.get("FAKE_REJECT_STREAM", "")
        marker = os.environ.get("FAKE_REJECT_FILE", "")
        if reject and msg["stream"].get("id") == reject \
                and (not marker or os.path.exists(marker)):
            send_reply(req, False, error=f"synthetic rejection of {reject}")
            return
        send_reply(req, True, {"stream_index": idx})
        send_state(idx, "starting")
        if os.environ.get("FAKE_DEV_TENSORS"):
            send_record(b"VBT1", encode_vbt1(idx, 1))
        send_state(idx, "running")
        emit_frames(idx, float(os.environ.get("FAKE_FPS", "20") or 20))
    elif op == "remove":
        send_reply(req, True, {"stream_index": msg.get("stream_index")})
        send_state(int(msg.get("stream_index", 0)), "stopped")
    elif op == "set_threshold":
        send_reply(req, True, {"value": msg.get("value")})
    elif op == "configure_analyzer":
        send_reply(req, True, {"stream_index": msg.get("stream_index"),
                               "name": msg.get("name")})
    elif op == "infer_image":
        send_reply(req, True, {"text": "AB", "mean_conf": 0.9,
                               "min_char_conf": 0.8, "w": 64, "h": 32,
                               "infer_ms": 0.1})
    elif op == "snapshot":
        meta = {"req": req, "stream_index": msg.get("stream_index", 0),
                "seq": msg.get("seq", 0), "track_id": msg.get("track_id", 0),
                "w": 8, "h": 4, "mime": "image/jpeg"}
        body = struct.pack("<I", len(json.dumps(meta).encode())) + \
            json.dumps(meta).encode() + b"\xff\xd8fake-jpeg"
        send_record(b"VBS1", body)
    elif op == "stop":
        send_reply(req, True, {})
        _stop.set()
        os.close(FD)
        os._exit(0)
    else:
        send_reply(req, False, error=f"unknown op {op}")


def stats_loop() -> None:
    if not os.environ.get("FAKE_STATS"):
        return
    def run():
        while not _stop.wait(1.0):
            send_control({"op": "stats", "rss_kb": 4096, "cpu_s": 0.1,
                          "streams": []})
    threading.Thread(target=run, daemon=True).start()


def main() -> None:
    if FD < 0:
        sys.exit("usage: fake_runtime.py --ipc-fd N")
    if os.environ.get("FAKE_EXIT_BEFORE_HELLO"):
        os._exit(1)
    if os.environ.get("FAKE_NO_HELLO"):
        time.sleep(120)
        os._exit(0)
    send_hello()
    if os.environ.get("FAKE_DEAF"):
        while True:                # alive, but never drains the socket
            time.sleep(60)
    stats_loop()
    buf = b""
    while True:
        try:
            chunk = os.read(FD, 65536)
        except OSError:
            os._exit(0)
        if not chunk:
            os._exit(0)
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                handle_line(line)


if __name__ == "__main__":
    main()
