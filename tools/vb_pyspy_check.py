#!/usr/bin/env python3
"""vb_pyspy_check — §10.5 ③ py-spy sample judgement.

Consumes ``py-spy record --format raw`` output (one sample per line, stack
frames joined by ``;``, each frame ``func (file:line)`` — recorded *without*
--idle, so every line is an active sample). Checks:

  (a) no sample's stack contains a frame from a --forbid module  -> failure
  (b) active_samples / (duration × rate) ≤ --max-active (0.25)   -> failure
  (c) prints the top-10 modules by frame count among active samples.

Exit 1 on failure. Serves as a pure function over the text file so it can be
validated without a live device.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

DEFAULT_FORBID = ("numpy,cv2,PIL,onnxruntime,tensorrt,hailo_platform,rknnlite,torch")


def _module_of_frame(frame: str) -> str:
    """Best-effort module label: text between '(' and the last '/' of the path,
    else the function prefix before ' ('."""
    frame = frame.strip()
    if "(" in frame and frame.endswith(")"):
        inner = frame[frame.rindex("(") + 1:-1]
        if "/" in inner:
            return inner.rsplit("/", 1)[0]
        if "." in inner:
            return inner.rsplit(".", 1)[0]
        return inner
    return frame.split(" (")[0].split(".")[0]


def check(text: str, duration: float, rate: int, max_active: float,
          forbid: list[str]) -> dict:
    samples = [line for line in text.splitlines() if line.strip()]
    active = len(samples)
    total_slots = duration * rate

    forbidden_hits: Counter = Counter()
    forbidden_samples = 0
    module_frames: Counter = Counter()
    for line in samples:
        frames = [f for f in line.split(";") if f.strip()]
        hit = False
        for frame in frames:
            module = _module_of_frame(frame)
            module_frames[module] += 1
            for bad in forbid:
                # 'numpy' matches 'site-packages/numpy/...', 'numpy.linalg...',
                # '... (numpy/core/_x.py:1)'; word boundary avoids 'numpyx'.
                if re.search(rf"(^|[^A-Za-z0-9_]){re.escape(bad)}($|[^A-Za-z0-9_])",
                             module):
                    forbidden_hits[bad] += 1
                    hit = True
        if hit:
            forbidden_samples += 1

    share = active / total_slots if total_slots > 0 else 1.0
    failures = []
    if forbidden_samples:
        failures.append(f"forbidden module frames in {forbidden_samples} sample(s): "
                        + ", ".join(f"{k}={v}" for k, v in sorted(forbidden_hits.items())))
    if share > max_active:
        failures.append(f"active share {share:.4f} > {max_active}")
    return {
        "duration_s": duration,
        "rate": rate,
        "active_samples": active,
        "active_share": round(share, 4),
        "max_active": max_active,
        "forbidden_modules": forbid,
        "forbidden_frame_hits": dict(sorted(forbidden_hits.items())),
        "top_modules": [{"module": m, "frames": n}
                        for m, n in module_frames.most_common(10)],
        "pass": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vb_pyspy_check", description="BASE-1 §10.5 ③ py-spy check")
    parser.add_argument("record", help="py-spy raw output file")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--max-active", type=float, default=0.25)
    parser.add_argument("--forbid", default=DEFAULT_FORBID,
                        help="comma-separated module names")
    parser.add_argument("--json-out", help="optional JSON report path")
    args = parser.parse_args(argv)

    with open(args.record, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    report = check(text, args.duration, args.rate, args.max_active,
                   [m.strip() for m in args.forbid.split(",") if m.strip()])
    out = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
    print(out)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
