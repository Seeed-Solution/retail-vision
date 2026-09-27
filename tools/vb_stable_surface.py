#!/usr/bin/env python3
"""Stable-surface snapshot tool (spec BASE-1 §6.14.2, M1.26).

Collects everything the base declares a *stable* external surface
(§6.14.1) into ``contracts/stable-surface.json``:

- the six stable schemas' ``required`` lists and ``properties`` key sets
  (``vb.config/1``, ``vb.event/1``, ``vb.frame/1``, ``vb.status/1``,
  ``vb.command/1``, ``vb.ack/1``);
- each analyzer's event ``type`` strings and the union of event field
  names (from the §6.2.4 fixtures' ``expect_events``);
- ``inspect.signature`` strings of the stable Python surface
  (``AppHooks``, ``StreamContext``, ``Outgoing``, ``ConfigApp``,
  ``vision_base.geom``, ``vision_base.embed.Runtime``,
  ``vision_base.mqtt`` public API);
- the C ABI header ``vb_analyzer_abi.h`` parsed textually: the
  ``VB_ANALYZER_ABI`` value, struct declarations and free function
  declarations (no compiler involved).

``--write`` regenerates the snapshot (review the diff in the PR and add a
line to CHANGELOG.md). ``--check`` compares the current surface against
the committed snapshot: **removing, renaming or changing** any recorded
item fails (exit 1); **additions** are reported but pass — within the
same schema id / base major only additions are allowed (§6.14.1).

Standard library only.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SNAPSHOT = REPO / "contracts" / "stable-surface.json"

STABLE_SCHEMAS = ["vb-config", "vb-event", "vb-frame", "vb-status",
                  "vb-command", "vb-ack"]
ABI_HEADER = "core-cpp/vb/include/vb/vb_analyzer_abi.h"
ANALYZER_FIXTURES = "contracts/fixtures/vb/analyzers"


# --------------------------------------------------------------- collection

def collect_schemas(contracts_dir: Path = REPO / "contracts") -> dict:
    out: dict[str, Any] = {}
    for name in STABLE_SCHEMAS:
        doc = json.loads((contracts_dir / f"{name}.schema.json").read_text())
        out[name] = {
            "required": sorted(doc.get("required", [])),
            "properties": sorted(doc.get("properties", {}).keys()),
        }
    return out


def collect_analyzers(fixtures_dir: Path = REPO / ANALYZER_FIXTURES) -> dict:
    """analyzer name -> {event type -> sorted union of field names}."""
    out: dict[str, dict[str, list[str]]] = {}
    for path in sorted(fixtures_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        analyzer = doc.get("analyzer")
        if not analyzer:
            continue
        per_type = out.setdefault(analyzer, {})
        for ev in doc.get("expect_events", []):
            fields = per_type.setdefault(ev["type"], set())
            fields.update(ev.get("fields", {}).keys())
    return {a: {t: sorted(fs) for t, fs in sorted(per.items())}
            for a, per in sorted(out.items())}


def _class_surface(cls: type) -> dict:
    """Public method signatures (+ dataclass fields via __init__)."""
    out: dict[str, str] = {}
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member):
            out[name] = str(inspect.signature(member))
    return out


def _module_surface(module: Any) -> dict:
    """Public functions + dataclasses of a module."""
    out: dict[str, Any] = {}
    for name, member in sorted(vars(module).items()):
        if name.startswith("_") or getattr(member, "__module__",
                                           None) != module.__name__:
            continue
        if inspect.isfunction(member):
            out[name] = str(inspect.signature(member))
        elif inspect.isclass(member):
            try:
                sig = str(inspect.signature(member))
            except (TypeError, ValueError):
                continue
            out[name] = sig
    return out


def collect_python(vision_base: Any | None = None) -> dict:
    if vision_base is None:
        import vision_base  # noqa: PLC0415
    from vision_base import apps, embed, geom, hooks, mqtt  # noqa: PLC0415
    return {
        "hooks.AppHooks": _class_surface(hooks.AppHooks),
        "hooks.StreamContext": _class_surface(hooks.StreamContext),
        "hooks.Outgoing": str(inspect.signature(hooks.Outgoing)),
        "apps.ConfigApp": _class_surface(apps.ConfigApp),
        "geom": _module_surface(geom),
        "embed.Runtime": _class_surface(embed.Runtime),
        "mqtt": _module_surface(mqtt),
        "mqtt.MqttClient": _class_surface(mqtt.MqttClient),
    }


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def collect_abi(header: Path = REPO / ABI_HEADER) -> dict:
    """Textual parse of vb_analyzer_abi.h — no compiler needed."""
    text = _strip_comments(header.read_text())
    abi = re.search(r"#define\s+VB_ANALYZER_ABI\s+(\d+)", text)
    if abi is None:
        raise SystemExit(f"VB_ANALYZER_ABI not found in {header}")
    structs: dict[str, list[str]] = {}
    body_wo_structs = text
    for m in re.finditer(r"typedef\s+struct\s+(\w+)\s*\{(.*?)\}\s*\w+;",
                         text, flags=re.S):
        name, body = m.group(1), m.group(2)
        members = [" ".join(chunk.split())
                   for chunk in body.replace("\n", " ").split(";")
                   if chunk.strip()]
        structs[name] = members
        body_wo_structs = body_wo_structs.replace(m.group(0), "")
    functions = []
    for line in body_wo_structs.splitlines():
        line = " ".join(line.split())
        if re.fullmatch(r"(?:const\s+)?[A-Za-z_][\w\s\*]*\**\w+\s*\([^()]*\)\s*;",
                        line):
            functions.append(line)
    return {"VB_ANALYZER_ABI": int(abi.group(1)),
            "structs": structs,
            "functions": sorted(functions)}


def build_snapshot(repo_root: Path = REPO) -> dict:
    sys.path.insert(0, str(repo_root / "core-py"))
    import vision_base  # noqa: F401,PLC0415
    return {
        "schemas": collect_schemas(repo_root / "contracts"),
        "analyzers": collect_analyzers(repo_root / ANALYZER_FIXTURES),
        "python": collect_python(vision_base),
        "abi": collect_abi(repo_root / ABI_HEADER),
    }


# ---------------------------------------------------------------- comparing

def _compare(path: str, recorded: Any, current: Any,
             errors: list[str], additions: list[str]) -> None:
    if isinstance(recorded, dict) and isinstance(current, dict):
        for key in sorted(recorded):
            if key not in current:
                errors.append(f"removed {path}.{key}")
            else:
                _compare(f"{path}.{key}", recorded[key], current[key],
                         errors, additions)
        for key in sorted(set(current) - set(recorded)):
            additions.append(f"added {path}.{key}")
    elif isinstance(recorded, list) and isinstance(current, list):
        rs, cs = set(recorded), set(current)
        for item in sorted(rs - cs):
            errors.append(f"removed {path} item {item!r}")
        for item in sorted(cs - rs):
            additions.append(f"added {path} item {item!r}")
    elif recorded != current:
        errors.append(f"changed {path}: {recorded!r} -> {current!r}")


def compare(recorded: dict, current: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    additions: list[str] = []
    _compare("", recorded, current, errors, additions)
    return errors, additions


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true",
                       help="regenerate contracts/stable-surface.json")
    group.add_argument("--check", action="store_true",
                       help="compare current surface against the snapshot")
    args = ap.parse_args()
    current = build_snapshot(REPO)
    if args.write:
        SNAPSHOT.write_text(json.dumps(current, indent=1,
                                       sort_keys=True) + "\n")
        print(f"wrote {SNAPSHOT.relative_to(REPO)}")
        return 0
    if not SNAPSHOT.exists():
        print(f"error: {SNAPSHOT} not found; run --write first",
              file=sys.stderr)
        return 1
    recorded = json.loads(SNAPSHOT.read_text())
    errors, additions = compare(recorded, current)
    for a in additions:
        print(f"NOTE: stable surface {a} (addition allowed within "
              "same schema id / base major; review in PR + CHANGELOG)")
    if errors:
        for e in errors:
            print(f"FAIL: stable surface {e}", file=sys.stderr)
        print("stable surface check failed: removals/changes are "
              "breaking; bump the schema id or the base major (§6.14.1)",
              file=sys.stderr)
        return 1
    print("stable surface check passed "
          f"({len(additions)} addition(s), 0 removals/changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
