#!/usr/bin/env python3
"""Compare two `vb-runtime --parity <dir>` runs (spec BASE-1 §7).

The reference run is the CPU ONNX backend; the other run is the platform
backend. Both write one JSON line per frame to `<dir>/parity.jsonl`, in capture
order, so frame *i* of one file is paired with frame *i* of the other.

Per frame:
  * the detection counts may differ by at most --count-diff, and
  * every reference detection must be matched (one-to-one, greedy by highest
    IoU) by a detection whose IoU is at least --iou.

Exit code 0 when every paired frame passes, 1 otherwise, 2 on a usage/IO error.

  uv run python tools/vb_parity_compare.py --ref /tmp/par-cpu --got /tmp/par-rk \
      --iou 0.9 --count-diff 1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# A quantised platform backend and its float reference should agree on which
# class each box is; comparing across classes would report a passing IoU for a
# box that is simply wrong.
CLASS_AWARE = True


def load(path: pathlib.Path) -> list[dict]:
    frames = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not JSON: {exc}") from exc
    return frames


def box(det: dict) -> tuple[float, float, float, float]:
    cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = box(a)
    bx0, by0, bx1, by1 = box(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(ref: list[dict], got: list[dict]) -> list[tuple[float, dict, dict]]:
    """Greedy best-IoU one-to-one matching, reference detections in order."""
    candidates = []
    for i, r in enumerate(ref):
        for j, g in enumerate(got):
            if CLASS_AWARE and r.get("class_id") != g.get("class_id"):
                continue
            candidates.append((iou(r, g), i, j))
    candidates.sort(key=lambda c: -c[0])
    used_ref, used_got, pairs = set(), set(), []
    for value, i, j in candidates:
        if i in used_ref or j in used_got:
            continue
        used_ref.add(i)
        used_got.add(j)
        pairs.append((value, ref[i], got[j]))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference parity dir (CPU ONNX)")
    ap.add_argument("--got", required=True, help="platform parity dir")
    ap.add_argument("--iou", type=float, default=0.9,
                    help="minimum per-detection IoU (default 0.9, the quantised line)")
    ap.add_argument("--count-diff", type=int, default=1,
                    help="allowed per-frame detection count difference (default 1)")
    ap.add_argument("--max-failures", type=int, default=10,
                    help="stop listing per-frame failures after this many")
    args = ap.parse_args()

    ref_path = pathlib.Path(args.ref) / "parity.jsonl"
    got_path = pathlib.Path(args.got) / "parity.jsonl"
    for p in (ref_path, got_path):
        if not p.is_file():
            print(f"parity_compare: missing {p}", file=sys.stderr)
            return 2

    ref_frames = load(ref_path)
    got_frames = load(got_path)
    if not ref_frames or not got_frames:
        print(f"parity_compare: empty parity file "
              f"(ref {len(ref_frames)} frames, got {len(got_frames)} frames)",
              file=sys.stderr)
        return 2
    if len(ref_frames) != len(got_frames):
        print(f"parity_compare: frame count differs: ref {len(ref_frames)}, "
              f"got {len(got_frames)}", file=sys.stderr)
        return 2

    failures = []
    ious: list[float] = []
    worst = (1.0, -1)
    count_diffs = 0
    for index, (rf, gf) in enumerate(zip(ref_frames, got_frames)):
        rd = rf.get("detections", [])
        gd = gf.get("detections", [])
        diff = abs(len(rd) - len(gd))
        count_diffs = max(count_diffs, diff)
        pairs = match(rd, gd)
        unmatched = len(rd) - len(pairs)
        frame_ious = [p[0] for p in pairs]
        ious.extend(frame_ious)
        if frame_ious and min(frame_ious) < worst[0]:
            worst = (min(frame_ious), index)
        problems = []
        if diff > args.count_diff:
            problems.append(f"count ref={len(rd)} got={len(gd)}")
        if unmatched > 0:
            problems.append(f"{unmatched} reference detection(s) unmatched")
        below = [v for v in frame_ious if v < args.iou]
        if below:
            problems.append(f"{len(below)} IoU below {args.iou} "
                            f"(min {min(below):.4f})")
        if problems:
            failures.append((index, rf.get("seq"), problems))

    frames = len(ref_frames)
    mean_iou = sum(ious) / len(ious) if ious else float("nan")
    print(f"parity_compare: frames={frames} matched_pairs={len(ious)} "
          f"mean_iou={mean_iou:.4f} worst_iou={worst[0]:.4f}@frame{worst[1]} "
          f"max_count_diff={count_diffs} "
          f"iou_threshold={args.iou} count_diff_limit={args.count_diff}")
    for index, seq, problems in failures[: args.max_failures]:
        print(f"  FAIL frame {index} (seq {seq}): " + "; ".join(problems))
    if len(failures) > args.max_failures:
        print(f"  ... {len(failures) - args.max_failures} more failing frames")
    if failures:
        print(f"parity_compare: FAIL ({len(failures)}/{frames} frames)")
        return 1
    print(f"parity_compare: PASS ({frames}/{frames} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
