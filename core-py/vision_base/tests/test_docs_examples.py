"""Docs/examples consistency tests (spec BASE-1 §8 M1.25).

- quickstart.md code blocks are byte-identical to ``examples/quickstart/``
- example files are within the size limits
- ``python -m vision_base.main --config examples/quickstart/config.json
  --validate`` passes
- quickstart.md stays free of internals (shard / context pool / VBR1 /
  supervisor / IPC)
- extending.md has the five layer headings and the a–d subsection headings
- the quickstart config with a synthetic backend, run against
  ``fake_runtime``, drives ``DoorCounter`` to produce both of its outputs
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from vision_base.hooks import StreamContext  # noqa: E402
from vision_base.runtime_client import RuntimeClient  # noqa: E402
from vision_base.types import Event, StreamSpec  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
QUICKSTART_MD = ROOT / "docs" / "quickstart.md"
EXTENDING_MD = ROOT / "docs" / "extending.md"
EXAMPLES = ROOT / "examples" / "quickstart"
FAKE = pathlib.Path(__file__).resolve().parent / "fake_runtime.py"
FAKE_ARGV = [sys.executable, str(FAKE), "--ipc-fd", "{fd}"]

INTERNAL_RE = re.compile(r"shard|context pool|VBR1|supervisor|IPC", re.IGNORECASE)


def _fenced(md: str, lang: str) -> list[str]:
    return re.findall(rf"^```{lang}\n(.*?)^```$", md, re.M | re.S)


# ------------------------------------------------------- quickstart.md ↔ files

def test_quickstart_config_block_matches_file():
    blocks = [b for b in _fenced(QUICKSTART_MD.read_text(), "json")
              if json.loads(b).get("schema") == "vb.config/1"]
    assert len(blocks) == 1
    assert blocks[0].rstrip("\n") == (EXAMPLES / "config.json").read_text().rstrip("\n")


def test_quickstart_hooks_block_matches_file():
    blocks = _fenced(QUICKSTART_MD.read_text(), "python")
    assert len(blocks) == 1
    assert blocks[0].rstrip("\n") == (EXAMPLES / "hooks.py").read_text().rstrip("\n")


def test_example_sizes():
    cfg_lines = (EXAMPLES / "config.json").read_text().splitlines()
    hook_lines = (EXAMPLES / "hooks.py").read_text().splitlines()
    assert len(cfg_lines) <= 20
    assert len(hook_lines) <= 20


def test_quickstart_mentions_both_examples():
    md = QUICKSTART_MD.read_text()
    assert "examples/quickstart/config.json" in md
    assert "examples/quickstart/hooks.py" in md


# ------------------------------------------------------------- --validate run

def test_quickstart_config_validates():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "core-py")
    r = subprocess.run(
        [sys.executable, "-m", "vision_base.main",
         "--config", str(EXAMPLES / "config.json"), "--validate"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------- no internals on the
# ------------------------------------------------------- quickstart surface

def test_quickstart_free_of_internals():
    assert not INTERNAL_RE.findall(QUICKSTART_MD.read_text())


# ---------------------------------------------------------- extending.md shape

def test_extending_structure():
    md = EXTENDING_MD.read_text()
    layers = re.findall(r"^## Layer ([1-5]) — ", md, re.M)
    assert layers == ["1", "2", "3", "4", "5"]
    subs = re.findall(r"^### ([a-d])\) ", md, re.M)
    assert subs == ["a", "b", "c", "d"]
    assert "stable surface" in md.lower()


# ------------------------------------------------------ DoorCounter via fake
# ------------------------------------------------------ runtime + synthetic
# ------------------------------------------------------ backend

def _import_doorcounter():
    import importlib
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module("hooks").DoorCounter
    finally:
        sys.path.remove(str(EXAMPLES))


def test_doorcounter_produces_both_topics_over_fake_runtime():
    # quickstart config with the backend swapped for synthetic and the app
    # module pointed at the quickstart hooks.py
    cfg = json.loads((EXAMPLES / "config.json").read_text())
    cfg["backend"] = {"name": "synthetic", "model_path": "synthetic://"}
    cfg["app"] = {"module": "hooks:DoorCounter"}
    # sanity: the mutated config must still be a valid vb.config/1
    from vision_base.config import load
    tmp = ROOT / "build" / "test-docs-quickstart.json"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(cfg))
    try:
        load(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    DoorCounter = _import_doorcounter()
    app = DoorCounter()
    app.configure(dict(cfg["app"].get("options", {})), cfg["device_id"])

    got_event = threading.Event()
    received: list[Event] = []

    def on_event(idx, ev):
        received.append(ev)
        got_event.set()

    client = RuntimeClient(FAKE_ARGV, "/dev/null",
                           on_frame=lambda i, r: None, on_event=on_event,
                           on_stats=lambda s: None, on_state=lambda i, s, e: None,
                           on_exit=lambda code: got_event.set())
    client.start(hello_timeout_s=10.0)
    try:
        reply = client.request("add", 5.0, stream={
            "index": 0, "id": "door", "url": cfg["streams"][0]["url"],
            "transport": "tcp", "options": {}}, analyzers=[])
        assert reply["ok"]
        ctx = StreamContext(0, StreamSpec("door", cfg["streams"][0]["url"],
                                          options=cfg["streams"][0].get("options", {})),
                            client, ())
        # a real event from the fake runtime proves the pipeline is live
        assert got_event.wait(10.0), "no event from fake_runtime"

        # inject a line_cross event through the fake-runtime-backed context
        ev = Event(stream_id="door", seq=received[-1].seq + 1,
                   wall_ms=time.time() * 1000.0, analyzer="line_cross",
                   type="line_cross", track_id=7,
                   fields={"line_id": "door", "direction": "forward",
                           "anchor": [0.52, 0.61], "class_id": 0,
                           "score": 0.81})
        out = app.on_event(ctx, ev)
        assert [o.topic_suffix for o in out] == ["events/door", "count/door"]
        assert out[0].payload["schema"] == "vb.event/1"
        assert out[1].payload == {"inside": 1}
        assert ctx.state["inside"] == 1

        # a second forward crossing increments the running count
        import dataclasses
        ev2 = dataclasses.replace(ev, seq=ev.seq + 1)
        out2 = app.on_event(ctx, ev2)
        assert [o.topic_suffix for o in out2] == ["events/door", "count/door"]
        assert out2[1].payload == {"inside": 2}
    finally:
        try:
            client.stop()
        except Exception:
            client.kill()
