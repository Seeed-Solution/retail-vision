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

Quantised-model acceptance is the three-gate ruling of 2026-09-28 (spec §7
「量化 parity 门槛口径」):
  1. count difference <= 1 per frame (against the float ONNX reference);
  2. faithfulness cross-check: the float reference, run on the device's own
     canvas, must reproduce the device output with paired IoU >= 0.95 - this
     isolates conversion fidelity from decode/preprocess stack differences;
  3. box-level agreement (mean IoU, IoU>=0.9 share) is recorded as a
     diagnostic, not a gate.

`--faithfulness` runs gate 2: same comparison, threshold defaulting to 0.95,
reporting the paired-IoU distribution the ruling asks for:

  uv run python tools/vb_parity_compare.py --ref /tmp/par-cpu-on-device \
      --got /tmp/par-rk --faithfulness

Pose models (M2.3): when detections carry per-detection `keypoints`
([[x, y, conf], ...]), matched pairs are compared point-wise and the mean /
max keypoint distance (box diagonal units, i.e. normalised) is reported. A
half-cell parameterisation error shifts keypoints by stride/2 pixels, which
box IoU does not expose; pass --kpt-max to make it a gate.

`--selftest` builds synthetic record pairs in memory and checks the tool's
own verdicts (pass, faithfulness failure, count failure, keypoint shift).
"""

from __future__ import annotations

import argparse
import json
import math
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


def _finite_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _validate_detection(det: object, frame_index: int, side: str) -> None:
    if not isinstance(det, dict):
        raise ValueError(f"frame {frame_index} {side}: detection must be an object")
    for key in ("cx", "cy", "w", "h"):
        if key not in det or not _finite_number(det[key]):
            raise ValueError(f"frame {frame_index} {side}: bbox {key} must be finite")
    if float(det["w"]) <= 0 or float(det["h"]) <= 0:
        raise ValueError(f"frame {frame_index} {side}: bbox width/height must be > 0")
    if "keypoints" in det and det["keypoints"] is not None:
        points = det["keypoints"]
        if not isinstance(points, list):
            raise ValueError(f"frame {frame_index} {side}: keypoints must be an array")
        for point in points:
            if (not isinstance(point, list) or len(point) != 3 or
                    any(not _finite_number(v) for v in point)):
                raise ValueError(f"frame {frame_index} {side}: invalid keypoint")


def _validate_inputs(ref_frames: list[dict], got_frames: list[dict], iou_th: float,
                     count_limit: int, kpt_max: float | None) -> None:
    if not _finite_number(iou_th) or not 0 <= float(iou_th) <= 1:
        raise ValueError("iou threshold must be finite and in [0, 1]")
    if not isinstance(count_limit, int) or isinstance(count_limit, bool) or count_limit < 0:
        raise ValueError("count difference limit must be a nonnegative integer")
    if kpt_max is not None and (not _finite_number(kpt_max) or float(kpt_max) < 0):
        raise ValueError("keypoint threshold must be finite and >= 0")
    if not ref_frames or not got_frames:
        raise ValueError(f"frame lists must be non-empty (ref {len(ref_frames)}, got {len(got_frames)})")
    if len(ref_frames) != len(got_frames):
        raise ValueError(f"frame count differs: ref {len(ref_frames)}, got {len(got_frames)}")
    for index, (rf, gf) in enumerate(zip(ref_frames, got_frames)):
        for frame, side in ((rf, "ref"), (gf, "got")):
            detections = frame.get("detections", []) if isinstance(frame, dict) else None
            if not isinstance(detections, list):
                raise ValueError(f"frame {index} {side}: detections must be an array")
            for det in detections:
                _validate_detection(det, index, side)


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


def kpt_distances(ref: list[dict], got: list[dict]) -> list[float]:
    """Per-matched-pair mean keypoint L2 distance (normalised units)."""
    rk = ref.get("keypoints")
    gk = got.get("keypoints")
    if not rk or not gk or len(rk) != len(gk):
        return []
    dists = []
    for (rx, ry, _), (gx, gy, _) in zip(rk, gk):
        dists.append(((rx - gx) ** 2 + (ry - gy) ** 2) ** 0.5)
    return [sum(dists) / len(dists)] if dists else []


def compare(ref_frames: list[dict], got_frames: list[dict], iou_th: float,
            count_limit: int, kpt_max: float | None) -> tuple[bool, list[str]]:
    """Returns (pass, report lines). Shared by the file path and --selftest."""
    _validate_inputs(ref_frames, got_frames, iou_th, count_limit, kpt_max)
    lines: list[str] = []
    failures: list[str] = []
    global_failures: list[str] = []
    ious: list[float] = []
    kpts: list[float] = []
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
        frame_kpts: list[float] = []
        for _, r, g in pairs:
            if kpt_max is not None and (
                    not isinstance(r.get("keypoints"), list) or
                    not isinstance(g.get("keypoints"), list) or
                    len(r["keypoints"]) != len(g["keypoints"]) or
                    not r["keypoints"]):
                problems.append("keypoint gate requires matching non-empty keypoint arrays")
            frame_kpts.extend(kpt_distances(r, g))
        kpts.extend(frame_kpts)
        if bool(rd) != bool(gd):
            problems.append("single-sided empty detection frame")
        if diff > count_limit:
            problems.append(f"count ref={len(rd)} got={len(gd)}")
        if unmatched > 0:
            problems.append(f"{unmatched} reference detection(s) unmatched")
        below = [v for v in frame_ious if v < iou_th]
        if below:
            problems.append(f"{len(below)} paired IoU below {iou_th} "
                            f"(min {min(below):.4f})")
        if kpt_max is not None:
            bad_kpts = [v for v in frame_kpts if v > kpt_max]
            if bad_kpts:
                problems.append(f"{len(bad_kpts)} keypoint distances above "
                                f"{kpt_max} (max {max(bad_kpts):.4f})")
        if problems:
            failures.append(f"frame {index} (seq {rf.get('seq')}): " +
                            "; ".join(problems))
    frames = len(ref_frames)
    if not ious:
        global_failures.append("no matched detection pairs in the complete input")
    mean_iou = sum(ious) / len(ious) if ious else float("nan")
    lines.append(
        f"parity_compare: frames={frames} matched_pairs={len(ious)} "
        f"mean_iou={mean_iou:.4f} worst_iou={worst[0]:.4f}@frame{worst[1]} "
        f"max_count_diff={count_diffs} "
        f"iou_threshold={iou_th} count_diff_limit={count_limit}")
    if kpts:
        mean_kpt = sum(kpts) / len(kpts)
        over_95 = sum(1 for v in kpts if v >= 0.95)
        lines.append(
            f"parity_compare: keypoints pairs_with_kpts={len(kpts)} "
            f"mean_dist={mean_kpt:.5f} max_dist={max(kpts):.5f}"
            + (f" kpt_gate={kpt_max}" if kpt_max is not None else " (report only)"))
    lines.extend(f"  FAIL {f}" for f in failures[:10])
    lines.extend(f"  FAIL {f}" for f in global_failures)
    if len(failures) > 10:
        lines.append(f"  ... {len(failures) - 10} more failing frames")
    if failures or global_failures:
        lines.append(f"parity_compare: FAIL ({len(failures)}/{frames} frames"
                     + (f"; {len(global_failures)} global check(s)" if global_failures else "")
                     + ")")
    else:
        lines.append(f"parity_compare: PASS ({frames}/{frames} frames)")
    return not failures and not global_failures, lines


def selftest() -> int:
    """Build synthetic record pairs and check the tool's own verdicts."""
    def det(cx, cy, w, h, score=0.9, cls=0, kpts=None):
        d = {"cx": cx, "cy": cy, "w": w, "h": h, "score": score, "class_id": cls}
        if kpts is not None:
            d["keypoints"] = kpts
        return d

    def frame(dets, seq=0):
        return {"seq": seq, "detections": dets}

    check = 0

    # 1. Identical records pass at both thresholds.
    a = [frame([det(0.5, 0.5, 0.2, 0.3)], 1)]
    ok, lines = compare(a, [frame([det(0.5, 0.5, 0.2, 0.3)], 1)], 0.9, 1, None)
    print("\n".join(lines))
    if not ok:
        print("selftest: identical records must pass")
        check = 1
    ok, _ = compare(a, [frame([det(0.5, 0.5, 0.2, 0.3)], 1)], 0.95, 1, None)
    if not ok:
        print("selftest: identical records must pass faithfulness")
        check = 1

    # 2. A box with paired IoU 0.905 (0.02 shift on a 0.4x0.5 box) passes the
    #    quantised line but fails faithfulness (the stack-vs-conversion split
    #    the ruling separates).
    a = [frame([det(0.5, 0.5, 0.4, 0.5)], 1)]
    shifted = [frame([det(0.52, 0.5, 0.4, 0.5)], 1)]
    ok, _ = compare(a, shifted, 0.9, 1, None)
    if not ok:
        print("selftest: IoU-0.905 pair must pass the 0.9 line")
        check = 1
    ok, lines = compare(a, shifted, 0.95, 1, None)
    print("\n".join(lines))
    if ok:
        print("selftest: IoU-0.905 pair must fail faithfulness")
        check = 1

    # 3. Count difference beyond the limit fails both modes.
    three = [frame([det(0.5, 0.5, 0.4, 0.5), det(0.2, 0.2, 0.1, 0.1),
                    det(0.8, 0.8, 0.1, 0.1)], 1)]
    ok, _ = compare(a, three, 0.9, 1, None)
    if ok:
        print("selftest: count diff 1 vs 3 must fail")
        check = 1

    # 4. A half-cell keypoint shift (0.00625 normalised = 4 px at 640) shows in
    #    the report and trips the gate when asked to.
    kref = [[0.5, 0.4, 0.9]] * 17
    kgot = [[0.50625, 0.40625, 0.9]] * 17
    ok, lines = compare([frame([det(0.5, 0.5, 0.2, 0.3, kpts=kref)], 1)],
                        [frame([det(0.5, 0.5, 0.2, 0.3, kpts=kgot)], 1)],
                        0.9, 1, None)
    print("\n".join(lines))
    if not any("keypoints" in ln for ln in lines):
        print("selftest: keypoint report missing")
        check = 1
    ok, lines = compare([frame([det(0.5, 0.5, 0.2, 0.3, kpts=kref)], 1)],
                        [frame([det(0.5, 0.5, 0.2, 0.3, kpts=kgot)], 1)],
                        0.9, 1, 0.005)
    print("\n".join(lines))
    if ok:
        print("selftest: 0.00625 keypoint shift must trip a 0.005 gate")
        check = 1

    if check == 0:
        print("parity_compare selftest: all verdicts as expected")
    return check


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", help="reference parity dir (CPU ONNX)")
    ap.add_argument("--got", help="platform parity dir")
    ap.add_argument("--iou", type=float, default=None,
                    help="minimum paired IoU (default 0.9, or 0.95 with "
                         "--faithfulness)")
    ap.add_argument("--count-diff", type=int, default=1,
                    help="allowed per-frame detection count difference (default 1)")
    ap.add_argument("--faithfulness", action="store_true",
                    help="gate-2 mode of the 2026-09-28 ruling: the reference "
                         "is the float model on the device's own canvas; "
                         "default --iou becomes 0.95")
    ap.add_argument("--kpt-max", type=float, default=None,
                    help="fail when a matched pair's mean keypoint distance "
                         "exceeds this (normalised; report-only when absent)")
    ap.add_argument("--max-failures", type=int, default=10,
                    help="stop listing per-frame failures after this many")
    ap.add_argument("--selftest", action="store_true",
                    help="run the in-memory verdict checks and exit")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    iou_th = args.iou
    if iou_th is None:
        iou_th = 0.95 if args.faithfulness else 0.9

    for name in ("--ref", "--got"):
        if not getattr(args, name[2:]):
            ap.error(f"{name} is required (or use --selftest)")
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

    try:
        ok, lines = compare(ref_frames, got_frames, iou_th, args.count_diff,
                            args.kpt_max)
    except ValueError as exc:
        print(f"parity_compare: invalid input: {exc}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
