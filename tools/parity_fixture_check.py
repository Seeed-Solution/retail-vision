#!/usr/bin/env python3
"""parity_fixture_check.py — verify the parity frame-alignment fixture.

Both sides of a `vb-runtime --parity` run write one JSON line per captured
frame to <dir>/parity.jsonl. Line pairing in vb_parity_compare.py is only
meaningful if each side captured the SAME frame sequence. The fixture
(tools/parity_feed.sh) publishes the clip once at a rate slow enough that no
consumer drops frames, so each side must have logged exactly N lines (N is
printed by parity_feed.sh as PARITY_FRAMES=<N>).

Line counts alone CANNOT detect a *content phase* shift: a consumer that
silently drops the first payload frame (measured on radxa gst_source float
consumers, 2026-10) still logs N lines, but its line i is the reference line
i+1's frame — every downstream line-paired comparison is poisoned by +1.

When parity_feed.sh runs with `--pattern even-empty` it interleaves blank
frames into the payload, so the per-line detection-count sequence is a known
pattern: 0,>=1,0,>=1,... (even lines empty). This tool then verifies that
pattern per side (`--expect-pattern`), which catches any ODD total shift
(in particular the drop-first-frame offset +1) independently on each side,
without requiring the two sides' models to agree on exact counts.

Exit codes:
  0  fixture aligned: both files have exactly --frames lines and (if given)
     both match --expect-pattern
  3  fixture failure: line counts differ from N, or content phase mismatch
     (not a model disagreement — re-run the experiment; do NOT loosen
     compare thresholds)
  2  usage / IO error

vb_parity_compare.py responsibility boundary: THIS tool's exit 3 means the
fixture was not aligned (dropped frames / late attach / phase shift) — the
parity.jsonl line pairing is meaningless and compare results must be
discarded. Only after this tool exits 0 does an `vb_parity_compare.py`
exit 1 mean a real model disagreement.
"""

import argparse
import json
import sys


def read_counts(path: str):
    """Return (n_lines, [detection count per line])."""
    counts = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts.append(None)
                continue
            dets = rec.get("detections")
            counts.append(len(dets) if isinstance(dets, list) else None)
    return len(counts), counts


def pattern_spec(spec: str, frames: int):
    """Parse --expect-pattern into [expected count or -1 per line].

    -1 = "any non-zero count". Returns None on a bad spec.
    """
    if spec == "even-empty":
        return [-1 if i % 2 else 0 for i in range(frames)]
    if spec == "odd-empty":
        return [0 if i % 2 else -1 for i in range(frames)]
    parts = spec.split(",")
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        return None
    if any(v < 0 for v in vals):
        return None
    return vals


def pattern_errors(counts, expected):
    """Return list of (index, expected, actual) pattern mismatches."""
    errs = []
    for i, want in enumerate(expected):
        if i >= len(counts):
            break
        got = counts[i]
        if got is None:
            errs.append((i, want, "unparseable"))
        elif want == -1:
            if got == 0:
                errs.append((i, ">=1", got))
        elif got != want:
            errs.append((i, want, got))
    return errs


def phase_shift_conclusion(counts, expected):
    """Best-effort offset diagnosis: does counts match expected shifted by k?

    Comparison happens in phase space (empty vs non-empty) so the "any
    non-zero" sentinel (-1) in expected participates correctly. This catches
    odd shifts only when the pattern is a pure alternation.
    """
    def norm(seq):
        return [0 if (x == 0 or x == "unparseable") else 1
                for x in seq[: len(expected)]]
    n = min(len(counts), len(expected))
    c, e = norm(counts), [0 if x == 0 else 1 for x in expected]
    # Best-effort: only small shifts are diagnosed (large truncations are
    # already reported as line-count problems and make the search ambiguous).
    for k in range(1, min(6, max(2, n // 3))):
        if c[: n - k] == e[k:]:
            return k
        if c[k:] == e[: n - k]:
            return -k
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--ref", required=True, help="reference side parity.jsonl")
    ap.add_argument("--got", required=True, help="device side parity.jsonl")
    ap.add_argument("--frames", type=int, required=True,
                    help="expected payload frame count (PARITY_FRAMES from parity_feed.sh)")
    ap.add_argument("--expect-pattern", default=None,
                    help="content phase pattern (PARITY_PATTERN from parity_feed.sh): "
                         "'even-empty' | 'odd-empty' | comma list of exact counts")
    ap.add_argument("--frames-only", action="store_true",
                    help="legacy mode: check line counts only, skip the phase check")
    args = ap.parse_args()

    if args.frames <= 0:
        print(f"--frames must be > 0, got {args.frames}", file=sys.stderr)
        return 2

    expected = None
    if args.expect_pattern and not args.frames_only:
        expected = pattern_spec(args.expect_pattern, args.frames)
        if expected is None:
            print(f"bad --expect-pattern {args.expect_pattern!r} "
                  "(use 'even-empty', 'odd-empty' or a comma list)", file=sys.stderr)
            return 2

    try:
        ref_n, ref_counts = read_counts(args.ref)
    except OSError as e:
        print(f"cannot read --ref {args.ref}: {e}", file=sys.stderr)
        return 2
    try:
        got_n, got_counts = read_counts(args.got)
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

    if expected is not None:
        for label, counts, path in (("ref", ref_counts, args.ref),
                                    ("got", got_counts, args.got)):
            errs = pattern_errors(counts, expected)
            if errs:
                shown = ", ".join(f"line {i}: want {w} got {g}" for i, w, g in errs[:5])
                more = f" (+{len(errs) - 5} more)" if len(errs) > 5 else ""
                problems.append(
                    f"{label} ({path}): content phase mismatch vs --expect-pattern "
                    f"{args.expect_pattern} [{shown}{more}]")
                shift = phase_shift_conclusion(counts, expected)
                if shift is not None:
                    if shift > 0:
                        problems.append(
                            f"{label}: sequence matches the expected pattern shifted by "
                            f"+{shift} (first {shift} payload frame(s) dropped or "
                            f"sequence left-shifted)")
                    else:
                        problems.append(
                            f"{label}: sequence matches the expected pattern shifted by "
                            f"{shift} (extra frame(s) at the head / right-shifted)")

    if problems:
        print("PARITY FIXTURE NOT ALIGNED (this is a fixture/dropped-frame failure, "
              "not a model disagreement):")
        for p in problems:
            print(f"  - {p}")
        print("Re-run the experiment (parity_feed.sh + both consumers); do not adjust thresholds.")
        return 3

    if expected is not None:
        print(f"PARITY FIXTURE ALIGNED: ref={ref_n} lines, got={got_n} lines, expected {args.frames}; "
              f"content phase OK (pattern {args.expect_pattern}, "
              f"ref counts={ref_counts[:12]}{'...' if ref_n > 12 else ''}, "
              f"got counts={got_counts[:12]}{'...' if got_n > 12 else ''})")
    else:
        print(f"PARITY FIXTURE ALIGNED: ref={ref_n} lines, got={got_n} lines, expected {args.frames}"
              + (" (frames-only: no content phase check)" if args.frames_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
