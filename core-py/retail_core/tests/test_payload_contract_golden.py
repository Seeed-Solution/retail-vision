"""Golden fixtures for contracts/validate_payload.py.

``contracts/fixtures/retail/valid/*.json`` must produce no errors;
``contracts/fixtures/retail/invalid/*.json`` hold ``{"message": ..., "errors": [...]}``
and must produce exactly that error list. The valid set includes payloads
captured from the wire golden, so the validator and the formatter are pinned
against each other.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "contracts" / "fixtures" / "retail"
sys.path.insert(0, str(ROOT / "contracts"))

from validate_payload import check  # noqa: E402

VALID = sorted((FIXTURES / "valid").glob("*.json"))
INVALID = sorted((FIXTURES / "invalid").glob("*.json"))


def test_fixture_sets_present():
    assert len(VALID) >= 3 and len(INVALID) >= 6


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_valid(path):
    assert check(json.loads(path.read_text())) == []


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.stem)
def test_invalid(path):
    case = json.loads(path.read_text())
    assert check(case["message"]) == case["errors"]


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text("".join(p.read_text().replace("\n", "") + "\n" for p in VALID))
    bad = tmp_path / "bad.jsonl"
    bad.write_text("".join(json.dumps(json.loads(p.read_text())["message"]) + "\n"
                           for p in INVALID) + "not json\n")
    script = ROOT / "contracts" / "validate_payload.py"
    ok = subprocess.run([sys.executable, str(script), str(good)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout
    assert f"checked={len(VALID)} invalid=0" in ok.stdout
    ko = subprocess.run([sys.executable, str(script), str(bad)], capture_output=True, text=True)
    assert ko.returncode == 1
    assert f"checked={len(INVALID) + 1} invalid={len(INVALID) + 1}" in ko.stdout
