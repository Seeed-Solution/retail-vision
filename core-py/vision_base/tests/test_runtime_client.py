"""Tests for vision_base.runtime_client (spec BASE-1 §8 M1.12) against the
Python fake native process (tests/fake_runtime.py)."""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from vision_base.runtime_client import RuntimeGone, RuntimeError_, RuntimeClient
from vision_base.types import StreamSpec

FAKE = os.path.join(os.path.dirname(__file__), "fake_runtime.py")


def make_argv() -> list[str]:
    return [sys.executable, FAKE, "--ipc-fd", "{fd}"]


def noop(*a):
    pass


def make_client(**cb) -> RuntimeClient:
    cbs = {"on_frame": noop, "on_event": noop, "on_stats": noop,
           "on_state": noop, "on_exit": noop}
    cbs.update(cb)
    return RuntimeClient(make_argv(), "unused.json", **cbs)


def add_stream(client: RuntimeClient, index: int = 0) -> dict:
    spec = StreamSpec(stream_id=f"cam-{index}", url="fake://")
    return client.request("add", 5.0, stream={
        "index": index, "id": spec.stream_id, "url": spec.url, "name": "",
        "transport": spec.transport, "score_threshold": 0.35,
        "options": spec.options}, analyzers=[])


def test_hello_and_pid(tmp_path, monkeypatch):
    client = make_client()
    hello = client.start(5.0)
    assert hello.runtime_version == "0.1.0-fake"
    assert hello.abi == 1
    assert hello.backend == "fake"
    assert hello.model_hw == (320, 320)
    assert hello.pid == client.pid
    assert client.pid == client.proc.pid
    client.stop()
    assert client.proc.wait(timeout=5) == 0


def test_hello_timeout_raises(monkeypatch):
    monkeypatch.setenv("FAKE_NO_HELLO", "1")
    client = make_client()
    with pytest.raises(TimeoutError):
        client.start(1.0)
    assert client.proc.poll() is not None  # child killed


def test_exit_before_hello_raises_runtime_gone(monkeypatch):
    monkeypatch.setenv("FAKE_EXIT_BEFORE_HELLO", "1")
    exits: list[int] = []
    client = make_client(on_exit=exits.append)
    with pytest.raises(RuntimeGone):
        client.start(5.0)
    assert exits == []  # pre-hello exits surface only via start() raising (review 2026-09-26)


def test_request_reply_pairing_by_req():
    client = make_client()
    client.start(5.0)
    results: dict[int, dict] = {}

    def do(i: int) -> None:
        results[i] = client.request("set_threshold", 5.0, stream_index=0,
                                    value=0.1 * (i + 1))

    threads = [threading.Thread(target=do, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(results) == 5
    for i, reply in results.items():
        assert reply["ok"] is True
        assert reply["applied"]["value"] == pytest.approx(0.1 * (i + 1))
    client.stop()


def test_ok_false_raises_runtime_error():
    client = make_client()
    client.start(5.0)
    with pytest.raises(RuntimeError_):
        client.request("bogus_op", 5.0)
    client.stop()


def test_crash_pending_request_gone_and_on_exit(monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_AFTER", "1")
    exits: list[int] = []
    client = make_client(on_exit=exits.append)
    client.start(5.0)
    assert add_stream(client)["ok"] is True      # line 1 -> ok
    with pytest.raises(RuntimeGone):             # line 2 -> crash, no reply
        client.request("set_threshold", 5.0, stream_index=0, value=0.5)
    deadline = time.time() + 5
    while not exits and time.time() < deadline:
        time.sleep(0.05)
    assert exits and exits[0] != 0


def test_snapshot_returns_meta_and_jpeg():
    client = make_client()
    client.start(5.0)
    add_stream(client)
    meta, jpeg = client.snapshot(0, max_side=192, timeout_s=3.0)
    assert meta["stream_index"] == 0
    assert meta["mime"] == "image/jpeg"
    assert jpeg.startswith(b"\xff\xd8")
    client.stop()


def test_infer_image():
    client = make_client()
    client.start(5.0)
    applied = client.infer_image(b"\xff\xd8xx", timeout_s=3.0)
    assert applied["text"] == "AB"
    client.stop()


def test_request_on_reader_thread_raises():
    errors: list[BaseException] = []

    def on_stats(stats: dict) -> None:
        try:
            client.request("set_threshold", 1.0, stream_index=0, value=0.5)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    client = make_client(on_stats=on_stats)
    import os
    os.environ["FAKE_STATS"] = "1"
    try:
        client.start(5.0)
        deadline = time.time() + 5
        while not errors and time.time() < deadline:
            time.sleep(0.05)
    finally:
        os.environ.pop("FAKE_STATS", None)
        client.stop()
    assert errors and isinstance(errors[0], RuntimeError)
    assert "deadlock" in str(errors[0])


def test_on_frame_and_on_event_delivery():
    frames: list[tuple[int, object]] = []
    events: list[tuple[int, object]] = []
    client = make_client(on_frame=lambda i, r: frames.append((i, r)),
                         on_event=lambda i, e: events.append((i, e)))
    client.start(5.0)
    add_stream(client, index=3)
    deadline = time.time() + 5
    while (not frames or not events) and time.time() < deadline:
        time.sleep(0.05)
    assert frames
    idx, res = frames[0]
    assert idx == 3
    assert res.seq >= 1
    assert res.geom.src_w == 1280 and res.geom.model_w == 320
    assert len(res.detections) == 1
    d = res.detections[0]
    assert d.track_id >= 1 and 0.0 <= d.score <= 1.0
    idx, ev = events[0]
    assert idx == 3 and ev.type == "tick" and ev.analyzer == "tick"
    client.stop()


def test_stop_reclaims_child_that_stopped_reading(monkeypatch):
    """Review item 19: a child that is alive but no longer drains its socket
    must not be able to keep ``stop()`` from reclaiming it.

    Without a send deadline the first ``sendall`` after the socket buffer
    fills blocks forever while holding the write lock, and ``stop()`` — which
    itself sends a control line — never returns.
    """
    monkeypatch.setenv("FAKE_DEAF", "1")
    client = make_client()
    try:
        client.start(5.0)
        # fill the socket buffer: 4 MiB cannot fit, and the child reads none
        # of it, so the write deadline is what ends the call.
        big = "x" * (4 * 1024 * 1024)
        with pytest.raises(TimeoutError):
            client.request("add", 1.0, stream={"index": 0, "id": "cam-0",
                                               "url": big})
        assert client.proc.poll() is None        # the child is still alive

        done = []
        t = threading.Thread(target=lambda: (client.stop(timeout_s=1.0),
                                             done.append(True)))
        t.daemon = True
        t.start()
        t.join(timeout=20.0)
        assert done, "stop() did not return: a blocked send still owns the lock"
        assert client.proc.poll() is not None    # child reclaimed
    finally:
        client.kill()
