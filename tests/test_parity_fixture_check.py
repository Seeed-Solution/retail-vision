"""Unit tests for tools/parity_fixture_check.py (fixture alignment checker)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "parity_fixture_check.py"

spec = importlib.util.spec_from_file_location("parity_fixture_check", TOOL)
pfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfc)


def write_jsonl(path: Path, n: int) -> Path:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"detections": [{"box": [i, 0, 1, 1], "score": 0.9}]}))
            f.write("\n")
    return path


def run_tool(ref: Path, got: Path, frames: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--ref", str(ref), "--got", str(got), "--frames", str(frames)],
        capture_output=True, text=True,
    )


def test_equal_counts_pass(tmp_path):
    ref = write_jsonl(tmp_path / "a.jsonl", 5)
    got = write_jsonl(tmp_path / "b.jsonl", 5)
    proc = run_tool(ref, got, 5)
    assert proc.returncode == 0, proc.stderr
    assert "ALIGNED" in proc.stdout


def test_ref_missing_lines_fails_3(tmp_path):
    ref = write_jsonl(tmp_path / "a.jsonl", 4)
    got = write_jsonl(tmp_path / "b.jsonl", 5)
    proc = run_tool(ref, got, 5)
    assert proc.returncode == 3
    assert "missing: 1" in proc.stdout
    assert "a.jsonl" in proc.stdout


def test_got_missing_lines_fails_3(tmp_path):
    ref = write_jsonl(tmp_path / "a.jsonl", 5)
    got = write_jsonl(tmp_path / "b.jsonl", 2)
    proc = run_tool(ref, got, 5)
    assert proc.returncode == 3
    assert "missing: 3" in proc.stdout
    assert "b.jsonl" in proc.stdout


def test_extra_lines_fails_3(tmp_path):
    ref = write_jsonl(tmp_path / "a.jsonl", 5)
    got = write_jsonl(tmp_path / "b.jsonl", 7)
    proc = run_tool(ref, got, 5)
    assert proc.returncode == 3
    assert "extra: 2" in proc.stdout


def test_both_sides_reported(tmp_path):
    ref = write_jsonl(tmp_path / "a.jsonl", 3)
    got = write_jsonl(tmp_path / "b.jsonl", 8)
    proc = run_tool(ref, got, 5)
    assert proc.returncode == 3
    assert "missing: 2" in proc.stdout
    assert "extra: 3" in proc.stdout


def test_empty_files_zero_lines(tmp_path):
    ref = tmp_path / "a.jsonl"
    ref.write_text("")
    got = tmp_path / "b.jsonl"
    got.write_text("")
    proc = run_tool(ref, got, 0 if False else 4)
    assert proc.returncode == 3
    assert "missing: 4" in proc.stdout


def test_zero_frames_expectation_is_usage_error(tmp_path):
    ref = write_jsonl(tmp_path / "a.jsonl", 0)
    got = write_jsonl(tmp_path / "b.jsonl", 0)
    proc = run_tool(ref, got, 0)
    assert proc.returncode == 2


def test_negative_frames_expectation_is_usage_error(tmp_path):
    ref = write_jsonl(tmp_path / "a.jsonl", 1)
    got = write_jsonl(tmp_path / "b.jsonl", 1)
    proc = run_tool(ref, got, -1)
    assert proc.returncode == 2


def test_missing_ref_file_is_usage_error(tmp_path):
    got = write_jsonl(tmp_path / "b.jsonl", 1)
    proc = run_tool(tmp_path / "nope.jsonl", got, 1)
    assert proc.returncode == 2


def test_blank_lines_do_not_count(tmp_path):
    ref = tmp_path / "a.jsonl"
    ref.write_text('{"detections":[]}\n\n{"detections":[]}\n')
    got = write_jsonl(tmp_path / "b.jsonl", 2)
    proc = run_tool(ref, got, 2)
    assert proc.returncode == 0, proc.stdout


def test_equal_lines_but_different_content_still_passes(tmp_path):
    # Content fingerprints are NOT this tool's job; only alignment (line counts).
    ref = write_jsonl(tmp_path / "a.jsonl", 3)
    got = tmp_path / "b.jsonl"
    with got.open("w") as f:
        for i in range(3):
            f.write(json.dumps({"detections": [{"box": [i * 7, 1, 2, 2], "score": 0.42}]}))
            f.write("\n")
    proc = run_tool(ref, got, 3)
    assert proc.returncode == 0, proc.stdout


def test_usage_error_when_args_missing():
    proc = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True)
    assert proc.returncode == 2


def test_read_counts_module_helper(tmp_path):
    p = write_jsonl(tmp_path / "x.jsonl", 9)
    n, counts = pfc.read_counts(str(p))
    assert n == 9
    assert counts == [1] * 9


# ---- content phase checks (--expect-pattern) ----

def write_counts(path: Path, counts) -> Path:
    with path.open("w") as f:
        for c in counts:
            f.write(json.dumps({"detections": [{"box": [0, 0, 1, 1], "score": 0.9}] * c}))
            f.write("\n")
    return path


def run_tool_pat(ref: Path, got: Path, frames: int, pattern: str,
                 extra=()) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--ref", str(ref), "--got", str(got),
         "--frames", str(frames), "--expect-pattern", pattern, *extra],
        capture_output=True, text=True,
    )


ODD_EMPTY = [3, 0, 2, 0, 5, 0, 1, 0]  # payload line 0 is content, odd lines gray


def test_phase_matching_pattern_passes(tmp_path):
    ref = write_counts(tmp_path / "a.jsonl", ODD_EMPTY)
    got = write_counts(tmp_path / "b.jsonl", ODD_EMPTY)
    proc = run_tool_pat(ref, got, 8, "odd-empty")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALIGNED" in proc.stdout
    assert "content phase OK" in proc.stdout


def test_phase_pattern_allows_cross_model_count_divergence(tmp_path):
    # ref/got are different backends: exact counts may differ, the PHASE must not.
    ref = write_counts(tmp_path / "a.jsonl", ODD_EMPTY)
    got = write_counts(tmp_path / "b.jsonl", [6, 0, 9, 0, 2, 0, 4, 0])
    proc = run_tool_pat(ref, got, 8, "odd-empty")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_phase_shift_by_one_fails_with_offset(tmp_path):
    # The measured <rk3588-board> bug: float consumer drops payload frame 0, its
    # sequence is the correct one left-shifted by 1; line counts still match.
    ref = write_counts(tmp_path / "a.jsonl", ODD_EMPTY)
    got = write_counts(tmp_path / "b.jsonl", ODD_EMPTY[1:] + [3])
    proc = run_tool_pat(ref, got, 8, "odd-empty")
    assert proc.returncode == 3
    assert "content phase mismatch" in proc.stdout
    assert "shifted by +1" in proc.stdout


def test_phase_missing_frame_fails(tmp_path):
    ref = write_counts(tmp_path / "a.jsonl", ODD_EMPTY)
    got = write_counts(tmp_path / "b.jsonl", ODD_EMPTY[:-1])  # 7 lines
    proc = run_tool_pat(ref, got, 8, "odd-empty")
    assert proc.returncode == 3
    assert "missing: 1" in proc.stdout


def test_phase_frames_only_ignores_pattern(tmp_path):
    ref = write_counts(tmp_path / "a.jsonl", ODD_EMPTY)
    got = write_counts(tmp_path / "b.jsonl", ODD_EMPTY[1:] + [3])
    proc = run_tool_pat(ref, got, 8, "odd-empty", extra=("--frames-only",))
    assert proc.returncode == 0, proc.stdout
    assert "frames-only" in proc.stdout


def test_phase_even_empty_pattern(tmp_path):
    counts = [0, 4, 0, 2, 0, 5]
    ref = write_counts(tmp_path / "a.jsonl", counts)
    got = write_counts(tmp_path / "b.jsonl", counts)
    proc = run_tool_pat(ref, got, 6, "even-empty")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # inverted phase fails
    got2 = write_counts(tmp_path / "c.jsonl", [4, 0, 2, 0, 5, 0])
    proc2 = run_tool_pat(ref, got2, 6, "even-empty")
    assert proc2.returncode == 3
    assert "shifted by +1" in proc2.stdout


def test_phase_explicit_count_list(tmp_path):
    ref = write_counts(tmp_path / "a.jsonl", [1, 0, 2])
    got = write_counts(tmp_path / "b.jsonl", [1, 0, 2])
    proc = run_tool_pat(ref, got, 3, "1,0,2")
    assert proc.returncode == 0, proc.stdout
    got2 = write_counts(tmp_path / "c.jsonl", [1, 1, 2])
    proc2 = run_tool_pat(ref, got2, 3, "1,0,2")
    assert proc2.returncode == 3
    assert "line 1: want 0 got 1" in proc2.stdout


def test_bad_pattern_spec_is_usage_error(tmp_path):
    ref = write_counts(tmp_path / "a.jsonl", [1, 0])
    got = write_counts(tmp_path / "b.jsonl", [1, 0])
    proc = run_tool_pat(ref, got, 2, "nonsense")
    assert proc.returncode == 2


def test_pattern_length_shorter_than_frames_only_checks_published_prefix(tmp_path):
    # explicit list shorter than frames: extra lines are not pattern-checked
    ref = write_counts(tmp_path / "a.jsonl", [1, 0, 9])
    got = write_counts(tmp_path / "b.jsonl", [1, 0, 9])
    proc = run_tool_pat(ref, got, 3, "1,0")
    assert proc.returncode == 0, proc.stdout
