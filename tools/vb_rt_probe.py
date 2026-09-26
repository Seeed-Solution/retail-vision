#!/usr/bin/env python3
"""vb-runtime IPC probe (spec BASE-1 §8 M1.8 acceptance helper).

Usage: vb_rt_probe.py SOCK [--add URL] [--expect-frames N]

Connects to a vb-runtime --listen socket, waits for hello, sends add for a
synthetic stream, waits for the ok reply and >= N VBR1 frame records, then
sends stop and exits 0. Any failure exits non-zero.
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import time


def recv_exact(s: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("eof")
        buf += chunk
    return buf


def main() -> int:
    sock_path = sys.argv[1]
    add_url = "synthetic://"
    expect_frames = 30
    args = sys.argv[2:]
    for i in range(0, len(args) - 1, 2):
        if args[i] == "--add":
            add_url = args[i + 1]
        elif args[i] == "--expect-frames":
            expect_frames = int(args[i + 1])
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10.0)
    s.connect(sock_path)

    # hello
    magic, body_len = struct.unpack("<4sI", recv_exact(s, 8))
    body = recv_exact(s, body_len)
    hello = json.loads(body)
    assert magic == b"VBC1" and hello["op"] == "hello", hello
    print("hello:", json.dumps(hello)[:120])

    s.sendall(json.dumps({
        "op": "add", "req": "r-1",
        "stream": {"index": 0, "id": "probe", "url": add_url, "name": "",
                   "transport": "tcp", "score_threshold": 0.35, "options": {}},
        "analyzers": [],
    }).encode() + b"\n")

    frames = 0
    reply_ok = False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and (frames < expect_frames or not reply_ok):
        magic, body_len = struct.unpack("<4sI", recv_exact(s, 8))
        body = recv_exact(s, body_len)
        if magic == b"VBC1":
            rec = json.loads(body)
            if rec.get("op") == "reply" and rec.get("req") == "r-1":
                assert rec["ok"], rec
                reply_ok = True
        elif magic == b"VBR1":
            frames += 1
    assert reply_ok, "no ok reply for add"
    assert frames >= expect_frames, f"only {frames} frames"

    s.sendall(json.dumps({"op": "stop", "req": "r-9"}).encode() + b"\n")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        magic, body_len = struct.unpack("<4sI", recv_exact(s, 8))
        body = recv_exact(s, body_len)
        if magic == b"VBC1" and json.loads(body).get("req") == "r-9":
            print(f"probe ok: {frames} frames")
            return 0
    print("no stop reply", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
