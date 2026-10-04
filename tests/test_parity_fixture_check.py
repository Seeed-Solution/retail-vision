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


def test_count_lines_module_helper(tmp_path):
    p = write_jsonl(tmp_path / "x.jsonl", 9)
    assert pfc.count_lines(str(p)) == 9
