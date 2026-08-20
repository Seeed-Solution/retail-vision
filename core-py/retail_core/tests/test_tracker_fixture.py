"""Cross-implementation behaviour fixture.

The expectations below were read off the reCamera C++ authority
(solutions/retail-vision/main/person_tracker.cpp and zone_metrics.cpp), not
recorded from this Python port, so the fixture is usable to check the C++ and
Python implementations against each other rather than only pinning the port.

`fixtures/tracker_sequence.json` holds the detection-box sequence and the
expected per-frame track ids and dwell states. Feeding the same sequence to a
C++ harness must produce the same two columns.

Run: python3 -m pytest core-py/retail_core/tests -q
  or python3 core-py/retail_core/tests/test_tracker_fixture.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from retail_core import (ASSISTANCE, ENGAGED, STATE_NAMES, TRANSIENT,  # noqa: E402
                         DetectionBox, PersonTracker, TrackerConfig, ZoneMetrics)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "tracker_sequence.json"


def run_sequence(frames, fps=15.0):
    metrics = ZoneMetrics(60.0)
    tracker = PersonTracker(TrackerConfig(frame_width=640, frame_height=640),
                            on_track_removed=metrics.on_track_removed)
    observed = []
    now = 0.0
    for frame in frames:
        now += 1.0 / fps
        detections = [DetectionBox(*box) for box in frame]
        persons = tracker.update(detections, now)
        metrics.update(tracker.state_counts(), tracker.entry_count,
                       tracker.exit_count, now)
        observed.append({
            "track_ids": sorted(p.track_id for p in persons),
            "states": {p.track_id: STATE_NAMES[p.dwell_state] for p in persons},
        })
    return tracker, metrics, observed


def test_fixture():
    data = json.loads(FIXTURE.read_text())
    tracker, metrics, observed = run_sequence(data["frames"], data["fps"])

    for check in data["expect_frames"]:
        i = check["frame"]
        got = observed[i]
        assert got["track_ids"] == check["track_ids"], (
            f"frame {i}: track ids {got['track_ids']} != {check['track_ids']} "
            f"({check['why']})")
        for tid, state in check.get("states", {}).items():
            assert got["states"][int(tid)] == state, (
                f"frame {i}: track {tid} state {got['states'][int(tid)]} != {state} "
                f"({check['why']})")

    expected = data["expect_final"]
    assert tracker.entry_count == expected["entry_count"], (
        f"entry_count {tracker.entry_count} != {expected['entry_count']}")
    snap = metrics.snapshot()
    for key, value in expected["zone"].items():
        assert snap[key] == value, f"zone.{key} {snap[key]} != {value}"
    print(f"fixture OK: {len(data['frames'])} frames, "
          f"{len(data['expect_frames'])} frame assertions, "
          f"entry={tracker.entry_count} exit={tracker.exit_count}, zone={snap}")


def test_slots_are_batch_indices():
    """Two people in one frame must produce slots 0 and 1, not track ids."""
    from retail_core import build_vision_payload
    frames = [[[0.30, 0.55, 0.16, 0.66, 0.85], [0.62, 0.50, 0.14, 0.60, 0.72]]] * 40
    tracker, metrics, _ = run_sequence(frames)
    persons = [t for t in tracker.tracks.values() if t.lost_frames == 0]
    payload = build_vision_payload(1709500000000, 40, 15.0, 20.0,
                                   metrics.snapshot(), persons, 1280, 720, 640, 640)
    assert [p["slot"] for p in payload["persons"]] == [0, 1]
    assert len({p["track_id"] for p in payload["persons"]}) == 2
    print("slot fixture OK:", [(p["slot"], p["track_id"]) for p in payload["persons"]])


if __name__ == "__main__":
    test_fixture()
    test_slots_are_batch_indices()
    print("all fixture checks passed")
