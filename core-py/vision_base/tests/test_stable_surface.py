"""Stable-surface snapshot tests (spec BASE-1 §6.14.2, M1.26).

Covers every acceptance bullet of the §8 M1.26 row:
- temp contracts copy with a required field removed from
  vb-event.schema.json -> check fails;
- an optional property added -> passes;
- temp module with a renamed AppHooks.on_event parameter -> fails;
- fake runtime hello with mismatched runtime_version major.minor ->
  RuntimeError_.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

from vision_base import __version__
from vision_base.runtime_client import (RuntimeClient, RuntimeError_,
                                         check_runtime_version)

_TOOLS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools")
_spec = importlib.util.spec_from_file_location(
    "vb_stable_surface", os.path.join(_TOOLS, "vb_stable_surface.py"))
vss = importlib.util.module_from_spec(_spec)
sys.modules["vb_stable_surface"] = vss
_spec.loader.exec_module(vss)

REPO = vss.REPO
FAKE = os.path.join(os.path.dirname(__file__), "fake_runtime.py")


def noop(*a):
    pass


def make_client() -> RuntimeClient:
    return RuntimeClient([sys.executable, FAKE, "--ipc-fd", "{fd}"],
                         "unused.json", on_frame=noop, on_event=noop,
                         on_stats=noop, on_state=noop, on_exit=noop)


# ------------------------------------------------------- snapshot vs. file

def test_check_against_committed_snapshot_passes():
    recorded = json.loads((REPO / "contracts" / "stable-surface.json")
                          .read_text())
    errors, additions = vss.compare(recorded, vss.build_snapshot(REPO))
    assert errors == []
    assert additions == []


def test_cli_check_passes():
    r = subprocess.run([sys.executable, str(REPO / "tools" /
                        "vb_stable_surface.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "stable surface check passed" in r.stdout


def temp_contracts(tmp_path):
    dst = tmp_path / "contracts"
    dst.mkdir()
    for name in vss.STABLE_SCHEMAS:
        shutil.copy(REPO / "contracts" / f"{name}.schema.json", dst)
    fixtures = tmp_path / "analyzers"
    shutil.copytree(REPO / vss.ANALYZER_FIXTURES, fixtures)
    return dst, fixtures


def check_with(contracts_dir, fixtures_dir):
    recorded = json.loads((REPO / "contracts" / "stable-surface.json")
                          .read_text())
    current = dict(schemas=vss.collect_schemas(contracts_dir),
                   analyzers=vss.collect_analyzers(fixtures_dir),
                   python=recorded["python"], abi=recorded["abi"])
    return vss.compare(recorded, current)


def test_removed_required_field_fails(tmp_path):
    contracts, fixtures = temp_contracts(tmp_path)
    path = contracts / "vb-event.schema.json"
    doc = json.loads(path.read_text())
    doc["required"].remove("track_id")
    path.write_text(json.dumps(doc))
    errors, _ = check_with(contracts, fixtures)
    assert any("schemas.vb-event.required" in e and "track_id" in e
               for e in errors)


def test_removed_property_fails(tmp_path):
    contracts, fixtures = temp_contracts(tmp_path)
    path = contracts / "vb-event.schema.json"
    doc = json.loads(path.read_text())
    del doc["properties"]["fields"]
    path.write_text(json.dumps(doc))
    errors, _ = check_with(contracts, fixtures)
    assert any("schemas.vb-event.properties" in e and "fields" in e
               for e in errors)


def test_added_optional_property_passes(tmp_path):
    contracts, fixtures = temp_contracts(tmp_path)
    path = contracts / "vb-event.schema.json"
    doc = json.loads(path.read_text())
    doc["properties"]["confidence"] = {"type": "number"}   # optional: not
    path.write_text(json.dumps(doc))                       # in required
    errors, additions = check_with(contracts, fixtures)
    assert errors == []
    assert any("confidence" in a for a in additions)


def test_removed_event_field_fails(tmp_path):
    contracts, fixtures = temp_contracts(tmp_path)
    # The recorded field set is a union over all fixtures of an analyzer,
    # so the field must be removed from every line_cross fixture.
    for path in fixtures.glob("line_cross_*.json"):
        doc = json.loads(path.read_text())
        for ev in doc.get("expect_events", []):
            ev.get("fields", {}).pop("direction", None)
        path.write_text(json.dumps(doc))
    errors, _ = check_with(contracts, fixtures)
    assert any("analyzers.line_cross" in e and "direction" in e
               for e in errors)


# ------------------------------------------------------- Python signatures

def load_patched_hooks(tmp_path, old: str, new: str):
    """Load vision_base/hooks.py with one textual patch as a submodule."""
    text = (REPO / "core-py" / "vision_base" / "hooks.py").read_text()
    assert old in text
    path = tmp_path / "_hooks_tmp.py"
    path.write_text(text.replace(old, new))
    spec = importlib.util.spec_from_file_location("vision_base._hooks_tmp",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vision_base._hooks_tmp"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_renamed_on_event_parameter_fails(tmp_path):
    mod = load_patched_hooks(
        tmp_path, "def on_event(self, ctx: StreamContext, ev: Event)",
        "def on_event(self, ctx: StreamContext, event: Event)")
    recorded = json.loads((REPO / "contracts" / "stable-surface.json")
                          .read_text())
    current = dict(vss.build_snapshot(REPO))
    current["python"]["hooks.AppHooks"] = vss._class_surface(mod.AppHooks)
    errors, _ = vss.compare(recorded, current)
    assert any("python.hooks.AppHooks.on_event" in e for e in errors)


def test_added_hook_method_is_only_an_addition(tmp_path):
    mod = load_patched_hooks(
        tmp_path,
        "def on_event(self, ctx: StreamContext, ev: Event) -> list[Outgoing]: ...",
        "def on_event(self, ctx: StreamContext, ev: Event) -> list[Outgoing]: ...\n"
        "    def on_optional_new(self, ctx) -> list[Outgoing]: ...")
    recorded = json.loads((REPO / "contracts" / "stable-surface.json")
                          .read_text())
    current = dict(vss.build_snapshot(REPO))
    current["python"]["hooks.AppHooks"] = vss._class_surface(mod.AppHooks)
    errors, additions = vss.compare(recorded, current)
    assert errors == []
    assert any("on_optional_new" in a for a in additions)


def test_abi_change_fails(tmp_path):
    header = tmp_path / "vb_analyzer_abi.h"
    text = (REPO / vss.ABI_HEADER).read_text().replace(
        "#define VB_ANALYZER_ABI 1", "#define VB_ANALYZER_ABI 2")
    header.write_text(text)
    recorded = json.loads((REPO / "contracts" / "stable-surface.json")
                          .read_text())
    errors, _ = vss.compare(recorded, dict(vss.build_snapshot(REPO),
                                           abi=vss.collect_abi(header)))
    assert any("abi.VB_ANALYZER_ABI" in e for e in errors)


# ------------------------------------------------------- runtime versioning

def test_version_check_allows_patch_differences():
    check_runtime_version("0.1.0")          # exact
    check_runtime_version(__version__)      # self
    check_runtime_version("0.1.99-tls")     # same major.minor, patch


@pytest.mark.parametrize("bad", [f"0.2.0", f"1.1.0", f"99.9.0"])
def test_version_check_rejects_major_minor_mismatch(bad):
    maj, minor, _ = (int(x) for x in __version__.split(".")[:3])
    bad = {"0.2.0": f"{maj}.{minor + 1}.0",
           "1.1.0": f"{maj + 1}.{minor}.0",
           "99.9.0": f"{maj + 9}.{minor + 9}.0"}[bad]
    with pytest.raises(RuntimeError_):
        check_runtime_version(bad)


def test_mismatched_fake_hello_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("FAKE_VERSION", "99.0.0-fake")
    client = make_client()
    with pytest.raises(RuntimeError_, match="major.minor mismatch"):
        client.start(5.0)
    client.kill()
    assert client.proc.poll() is not None


def test_matching_fake_hello_still_works():
    client = make_client()
    hello = client.start(5.0)
    assert hello.runtime_version == "0.1.0-fake"   # same major.minor
    client.stop()


def test_invalid_version_string_raises():
    with pytest.raises(RuntimeError_):
        check_runtime_version("not-a-version")
