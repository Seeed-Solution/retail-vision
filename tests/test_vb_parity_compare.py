import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import vb_parity_compare as parity


def det(**kw):
    out = {"cx": .5, "cy": .5, "w": .2, "h": .2, "class_id": 0}
    out.update(kw)
    return out


def frame(dets):
    return {"seq": 1, "detections": dets}


def test_single_sided_empty_is_a_failure():
    ok, lines = parity.compare([frame([])], [frame([det()])], .9, 1, None)
    assert not ok
    assert any("single-sided empty" in line for line in lines)


def test_mixed_empty_frames_allowed_but_all_empty_fails():
    ok, _ = parity.compare([frame([]), frame([det()])],
                           [frame([]), frame([det()])], .9, 1, None)
    assert ok
    ok, lines = parity.compare([frame([])], [frame([])], .9, 1, None)
    assert not ok
    assert any("no matched detection pairs" in line for line in lines)


@pytest.mark.parametrize("ref,got", [
    ([frame([det(w=0)])], [frame([det()])]),
    ([frame([det(cx=float("nan"))])], [frame([det()])]),
])
def test_invalid_bbox_rejected(ref, got):
    with pytest.raises(ValueError, match="bbox"):
        parity.compare(ref, got, .9, 1, None)


def test_keypoint_gate_missing_or_mismatched_does_not_pass():
    ok, lines = parity.compare([frame([det(keypoints=[[.1, .1, .9]])])],
                               [frame([det()])], .9, 1, .01)
    assert not ok
    assert any("keypoint gate" in line for line in lines)


def test_keypoint_gate_requires_a_pairable_non_empty_metric():
    ok, lines = parity.compare([frame([det(keypoints=[])])],
                               [frame([det(keypoints=[])])], .9, 1, .01)
    assert not ok
    assert any("non-empty keypoint arrays" in line for line in lines)


def test_keypoint_gate_accepts_non_empty_matching_points():
    points = [[.1, .1, .9], [.2, .2, .9]]
    ok, _ = parity.compare([frame([det(keypoints=points)])],
                           [frame([det(keypoints=points)])], .9, 1, .01)
    assert ok


def test_bbox_only_compare_does_not_require_keypoints():
    ok, _ = parity.compare([frame([det()])], [frame([det()])], .9, 1, None)
    assert ok


def test_direct_compare_rejects_bad_threshold_and_frame_count():
    with pytest.raises(ValueError, match="threshold"):
        parity.compare([frame([det()])], [frame([det()])], float("nan"), 1, None)
    with pytest.raises(ValueError, match="non-empty"):
        parity.compare([frame([det()])], [], .9, 1, None)


def test_cli_false_pass_counterexample_exits_one(tmp_path):
    ref = tmp_path / "ref"
    got = tmp_path / "got"
    ref.mkdir(); got.mkdir()
    (ref / "parity.jsonl").write_text(json.dumps(frame([])) + "\n")
    (got / "parity.jsonl").write_text(json.dumps(frame([det()])) + "\n")
    proc = subprocess.run([
        sys.executable, str(ROOT / "tools/vb_parity_compare.py"),
        "--ref", str(ref), "--got", str(got), "--iou", ".9",
        "--count-diff", "1",
    ], text=True, capture_output=True)
    assert proc.returncode == 1
    assert "single-sided empty" in proc.stdout
