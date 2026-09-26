#!/usr/bin/env python3
"""Validate JSONL fixtures/records against the vb.* JSON Schemas.

Usage:
    python contracts/validate_vb.py <kind> <file.jsonl|->
where <kind> is one of: command, ack, event, frame.

Each non-empty line must be one JSON object passing
``contracts/vb-<kind>.schema.json`` (jsonschema, dev dependency group).
Exit code 0 iff at least one line and all lines valid.
"""
import json
import pathlib
import sys

import jsonschema

SCHEMA_DIR = pathlib.Path(__file__).resolve().parent
KINDS = ("command", "ack", "event", "frame")


def load_schema(kind: str) -> dict:
    path = SCHEMA_DIR / f"vb-{kind}.schema.json"
    return json.loads(path.read_text())


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in KINDS:
        print(__doc__)
        return 2
    kind, src = sys.argv[1], sys.argv[2]
    schema = load_schema(kind)
    validator = jsonschema.Draft202012Validator(schema)
    stream = sys.stdin if src == "-" else open(src)
    total = bad = 0
    for line in stream:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            bad += 1
            print(f"[{total}] not JSON: {exc}")
            continue
        errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
        if errors:
            bad += 1
            for e in errors:
                loc = "/".join(str(p) for p in e.path) or "<root>"
                print(f"[{total}] {loc}: {e.message}")
    print(f"kind={kind} checked={total} invalid={bad}")
    return 1 if bad or total == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
