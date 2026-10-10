#!/usr/bin/env python3
"""Stable-surface snapshot tool (spec BASE-1 §6.14.2, M1.26).

Collects everything the base declares a *stable* external surface
(§6.14.1) into ``contracts/stable-surface.json``:

- the six stable schemas (``vb.config/1``, ``vb.event/1``, ``vb.frame/1``,
  ``vb.status/1``, ``vb.command/1``, ``vb.ack/1``) **recursively**: at every
  level the ``required`` list, the ``properties`` key set, each property's
  declared ``type``/``default``/bounds/``pattern``/``enum``, and the same
  again for ``items`` (array elements) and ``oneOf`` branches;
- each analyzer's event ``type`` strings and the union of event field
  names (from the §6.2.4 fixtures' ``expect_events``);
- ``inspect.signature`` strings of the stable Python surface
  (``hooks.AppHooks``, ``hooks.StreamContext``, ``hooks.Outgoing``,
  ``apps.ConfigApp``, ``vision_base.geom``, ``vision_base.embed.Runtime``,
  ``vision_base.mqtt`` public API) — methods, ``classmethod``/
  ``staticmethod``, ``property`` and public state fields alike;
- every user-visible ``vision_base.types`` dataclass with its field names,
  annotations and defaults;
- the C ABI header ``vb_analyzer_abi.h`` parsed textually: the
  ``VB_ANALYZER_ABI`` value, struct declarations and free function
  declarations (no compiler involved).

``--write`` regenerates the snapshot (review the diff in the PR and add a
line to CHANGELOG.md). ``--check`` compares the current surface against
the committed snapshot: **removing, renaming or changing** any recorded
item fails (exit 1); **additions** are reported but pass — within the same
schema id / base major only additions are allowed (§6.14.1).

Two asymmetries matter and are deliberate:

- A **new ``required`` field** (schema root or any nested object) is not an
  addition: it breaks every existing config/payload, so it fails. Only a new
  *optional* property is a legal addition.
- C ABI **member lists are order-sensitive**: reordering struct members
  changes the binary offsets without changing the name set, so they are
  compared positionally rather than as sets.

Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SNAPSHOT = REPO / "contracts" / "stable-surface.json"

STABLE_SCHEMAS = ["vb-config", "vb-event", "vb-frame", "vb-status",
                  "vb-command", "vb-ack"]
ABI_HEADER = "core-cpp/vb/include/vb/vb_analyzer_abi.h"
ANALYZER_FIXTURES = "contracts/fixtures/vb/analyzers"

# Schema keywords recorded verbatim for one node. Structure keywords
# (required/properties/items/oneOf) are handled separately below.
_SCHEMA_SCALAR_KEYS = ("$id", "type", "default", "minimum", "maximum",
                       "exclusiveMinimum", "exclusiveMaximum", "minItems",
                       "maxItems", "minLength", "maxLength", "pattern",
                       "enum", "const", "additionalProperties")

# List-valued keywords whose *order* carries no meaning; compared as sets.
# ``required`` is additionally strict: an added entry is a breaking change.
_UNORDERED_LIST_KEYS = frozenset({"required", "enum", "type"})


# --------------------------------------------------------------- collection

def _schema_node(node: Any) -> Any:
    """Recursive stable descriptor of one JSON-schema node."""
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key in _SCHEMA_SCALAR_KEYS:
        if key in node:
            out[key] = _schema_node(node[key])
    if "required" in node:
        out["required"] = sorted(node["required"])
    if "properties" in node:
        out["properties"] = {k: _schema_node(v)
                             for k, v in sorted(node["properties"].items())}
    if "items" in node:
        out["items"] = _schema_node(node["items"])
    if "oneOf" in node:
        out["oneOf"] = [_schema_node(branch) for branch in node["oneOf"]]
    return out


def collect_schemas(contracts_dir: Path = REPO / "contracts") -> dict:
    out: dict[str, Any] = {}
    for name in STABLE_SCHEMAS:
        doc = json.loads((contracts_dir / f"{name}.schema.json").read_text())
        out[name] = _schema_node(doc)
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


def _init_state_fields(cls: type) -> list[str]:
    """Public instance attributes assigned in ``__init__``.

    ``ctx.state``, ``rt.frames_dropped`` and friends are set on the
    instance, so they appear neither in the class annotations nor in
    ``vars(cls)``; the only static source is the constructor body.
    """
    init = cls.__dict__.get("__init__")
    if not inspect.isfunction(init):
        return []
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
    except (OSError, TypeError, SyntaxError, IndentationError, ValueError):
        return []
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                names.add(target.attr)
    return sorted(n for n in names if not n.startswith("_"))


def _attribute_value(value: Any) -> str:
    """Repr for immutable class attributes; a marker for anything else."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    return "attribute"


def _class_surface(cls: type) -> dict:
    """Public members: methods, classmethods, properties, state fields."""
    out: dict[str, str] = {}
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(member, classmethod):
            out[name] = f"classmethod{inspect.signature(getattr(cls, name))}"
        elif isinstance(member, staticmethod):
            out[name] = f"staticmethod{inspect.signature(getattr(cls, name))}"
        elif isinstance(member, property):
            out[name] = "property"
        elif inspect.isfunction(member):
            out[name] = str(inspect.signature(member))
        elif not inspect.isroutine(member) and not inspect.isdatadescriptor(member):
            # Plain public class attribute (e.g. ConfigApp.name).
            out[name] = f"attribute = {_attribute_value(member)}"
    for name, annotation in sorted(getattr(cls, "__annotations__", {}).items()):
        if not name.startswith("_"):
            out.setdefault(name, f"attribute: {annotation}")
    for name in _init_state_fields(cls):
        out.setdefault(name, "attribute")
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


def _field_default(field: dataclasses.Field) -> str:
    if field.default is not dataclasses.MISSING:
        return repr(field.default)
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        factory = getattr(field.default_factory, "__name__", None) \
            or repr(field.default_factory)
        return f"factory:{factory}"
    return "required"


def _dataclass_surface(cls: type) -> dict:
    """Field name -> {type, default} for a user-visible dataclass."""
    out: dict[str, dict[str, str]] = {}
    for field in dataclasses.fields(cls):
        if field.name.startswith("_"):
            continue
        out[field.name] = {"type": str(field.type),
                           "default": _field_default(field)}
    return out


def _types_surface(types_module: Any) -> dict:
    out: dict[str, dict] = {}
    for name, member in sorted(vars(types_module).items()):
        if name.startswith("_") or not inspect.isclass(member):
            continue
        if member.__module__ != types_module.__name__:
            continue
        if dataclasses.is_dataclass(member):
            out[name] = _dataclass_surface(member)
    return out


def collect_python(vision_base: Any | None = None) -> dict:
    if vision_base is None:
        import vision_base  # noqa: PLC0415
    from vision_base import apps, embed, geom, hooks, mqtt, types  # noqa: PLC0415
    return {
        "hooks.AppHooks": _class_surface(hooks.AppHooks),
        "hooks.StreamContext": _class_surface(hooks.StreamContext),
        "hooks.Outgoing": str(inspect.signature(hooks.Outgoing)),
        "apps.ConfigApp": _class_surface(apps.ConfigApp),
        "geom": _module_surface(geom),
        "embed.Runtime": _class_surface(embed.Runtime),
        "mqtt": _module_surface(mqtt),
        "mqtt.MqttClient": _class_surface(mqtt.MqttClient),
        "types": _types_surface(types),
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

def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _compare_lists(path: str, recorded: list, current: list,
                   errors: list[str], additions: list[str]) -> None:
    key = path.rsplit(".", 1)[-1]
    if key in _UNORDERED_LIST_KEYS:
        try:
            rs, cs = set(recorded), set(current)
        except TypeError:  # unhashable (dict) items: fall through to ordered
            rs = cs = None
        if rs is not None:
            for item in sorted(rs - cs, key=repr):
                errors.append(f"removed {path} item {item!r}")
            for item in sorted(cs - rs, key=repr):
                if key == "required":
                    # A new required field breaks every existing config /
                    # payload, so it is not a legal addition (§6.14.1).
                    errors.append(f"added required field {path} item "
                                  f"{item!r} (must be optional)")
                else:
                    additions.append(f"added {path} item {item!r}")
            return
    # Order-sensitive (C ABI member lists: order fixes the binary offsets).
    for i, item in enumerate(recorded):
        if i >= len(current):
            errors.append(f"removed {path}[{i}] item {item!r}")
        else:
            _compare(f"{path}[{i}]", item, current[i], errors, additions)
    for i in range(len(recorded), len(current)):
        additions.append(f"added {path}[{i}] item {current[i]!r}")


def _compare(path: str, recorded: Any, current: Any,
             errors: list[str], additions: list[str]) -> None:
    if isinstance(recorded, dict) and isinstance(current, dict):
        for key in sorted(recorded):
            if key not in current:
                errors.append(f"removed {_join(path, key)}")
            else:
                _compare(_join(path, key), recorded[key], current[key],
                         errors, additions)
        for key in sorted(set(current) - set(recorded)):
            additions.append(f"added {_join(path, key)}")
    elif isinstance(recorded, list) and isinstance(current, list):
        _compare_lists(path, recorded, current, errors, additions)
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
