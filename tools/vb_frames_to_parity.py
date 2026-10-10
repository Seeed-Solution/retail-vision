#!/usr/bin/env python3
"""Turn a `vb-runtime --standalone --output jsonl --frame-every 1` capture into
the `<dir>/parity.jsonl` shape that tools/vb_parity_compare.py reads.

Why this exists (BASE-1 M2.1): the standalone path (§6.10) reads `streams` from
the config file, so it is the supported way to produce per-frame detections
from a real source; this script re-expresses its `vb.frame/1` records as
parity records. It is also the only parity path that carries per-detection
keypoints (M2.3): `vb-runtime --parity` writes boxes only, while vb.frame/1
records carry them, so pose-model comparisons run through here.

  uv run python tools/vb_frames_to_parity.py --frames <capture.jsonl> \
      --out <parity-dir> [--stream parity-0] [--max-frames 50]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def box_to_cxcywh(box: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = box
    return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "w": abs(x2 - x1), "h": abs(y2 - y1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="standalone jsonl capture")
    ap.add_argument("--out", required=True, help="directory to write parity.jsonl into")
    ap.add_argument("--stream", default=None,
                    help="stream_id to keep (default: every stream, in order)")
    ap.add_argument("--stream-index", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=50)
    args = ap.parse_args()

    src = pathlib.Path(args.frames)
    if not src.is_file():
        print(f"vb_frames_to_parity: missing {src}", file=sys.stderr)
        return 2

    records = []
    with src.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"{src}:{lineno}: not JSON: {exc}", file=sys.stderr)
                return 2
            if obj.get("schema") != "vb.frame/1":
                continue
            if args.stream is not None and obj.get("stream_id") != args.stream:
                continue
            records.append(obj)
            if len(records) >= args.max_frames:
                break

    if not records:
        print(f"vb_frames_to_parity: no vb.frame/1 records in {src}", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parity.jsonl"
    with out_path.open("w") as fh:
        for obj in records:
            dets = []
            for det in obj.get("detections", []):
                d = box_to_cxcywh(det["box"])
                d["score"] = det.get("score", 0.0)
                d["class_id"] = det.get("class_id", 0)
                d["track_id"] = det.get("track_id", 0)
                # Pose models: per-detection keypoints as [[x, y, conf], ...],
                # already normalised in the vb.frame/1 record. Both runs of a
                # comparison share the source geometry, so the coordinates
                # compare directly (spec BASE-1 §M2.3: keypoint coordinates
                # must be compared separately - the half-cell decoder issue).
                if "keypoints" in det:
                    d["keypoints"] = det["keypoints"]
                dets.append(d)
            rec = {
                "stream_index": args.stream_index,
                "seq": obj.get("seq", 0),
                "wall_ms": obj.get("ts_ms", 0),
                "inference_ms": obj.get("inference_ms", 0.0),
                "detections": dets,
            }
            fh.write(json.dumps(rec) + "\n")

    print(f"vb_frames_to_parity: {len(records)} frames -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
