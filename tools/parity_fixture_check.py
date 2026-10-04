#!/usr/bin/env python3
"""parity_fixture_check.py — verify the parity frame-alignment fixture.

Both sides of a `vb-runtime --parity` run write one JSON line per captured
frame to <dir>/parity.jsonl. Line pairing in vb_parity_compare.py is only
meaningful if each side captured the SAME frame sequence. The fixture
(tools/parity_feed.sh) publishes the clip once at a rate slow enough that no
consumer drops frames, so each side must have logged exactly N lines (N is
printed by parity_feed.sh as PARITY_FRAMES=<N>).

Exit codes:
  0  fixture aligned: both files have exactly --frames lines
  3  fixture failure: line counts differ from N (not a model disagreement —
     re-run the experiment; do NOT loosen compare thresholds)
  2  usage / IO error
"""

import argparse
import json
import sys


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--ref", required=True, help="reference side parity.jsonl")
    ap.add_argument("--got", required=True, help="device side parity.jsonl")
    ap.add_argument("--frames", type=int, required=True,
                    help="expected payload frame count (PARITY_FRAMES from parity_feed.sh)")
    args = ap.parse_args()

    if args.frames <= 0:
        print(f"--frames must be > 0, got {args.frames}", file=sys.stderr)
        return 2

    try:
        ref_n = count_lines(args.ref)
    except OSError as e:
        print(f"cannot read --ref {args.ref}: {e}", file=sys.stderr)
        return 2
    try:
        got_n = count_lines(args.got)
    except OSError as e:
        print(f"cannot read --got {args.got}: {e}", file=sys.stderr)
        return 2

    problems = []
    if ref_n != args.frames:
        delta = ref_n - args.frames
        kind = "extra" if delta > 0 else "missing"
        problems.append(f"ref  ({args.ref}): {ref_n} lines, expected {args.frames} ({kind}: {abs(delta)})")
    if got_n != args.frames:
        delta = got_n - args.frames
        kind = "extra" if delta > 0 else "missing"
        problems.append(f"got  ({args.got}): {got_n} lines, expected {args.frames} ({kind}: {abs(delta)})")

    if problems:
        print("PARITY FIXTURE NOT ALIGNED (this is a fixture/dropped-frame failure, "
              "not a model disagreement):")
        for p in problems:
            print(f"  - {p}")
        print("Re-run the experiment (parity_feed.sh + both consumers); do not adjust thresholds.")
        return 3

    print(f"PARITY FIXTURE ALIGNED: ref={ref_n} lines, got={got_n} lines, expected {args.frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
