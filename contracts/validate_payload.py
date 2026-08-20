#!/usr/bin/env python3
"""Validate captured VisionPayload messages against the contract.

Dependency-free: checks required keys, types, enums and the batch invariants
that a plain JSON Schema pass would not catch (slot == batch index, one shared
timestamp, zone counts consistent with the persons array).

Usage: mosquitto_sub ... | python3 contracts/validate_payload.py
       python3 contracts/validate_payload.py captured.jsonl
"""
import json
import sys

STATES = {"transient", "dwelling", "engaged", "assistance"}


def check(msg):
    errors = []
    for key in ("timestamp", "frame_width", "frame_height", "zone", "persons"):
        if key not in msg:
            errors.append(f"missing {key}")
    if errors:
        return errors
    if not isinstance(msg["timestamp"], int):
        errors.append("timestamp must be integer epoch ms")
    zone = msg["zone"]
    for key in ("occupancy_count", "browsing_count", "engaged_count", "assist_count",
                "peak_customer", "avg_dwell_time", "avg_engagement_time",
                "avg_velocity", "entry_count", "exit_count"):
        if key not in zone:
            errors.append(f"zone.{key} missing")
    persons = msg["persons"]
    for i, p in enumerate(persons):
        if p.get("slot") != i:
            errors.append(f"persons[{i}].slot={p.get('slot')} must equal the batch index")
        if p.get("state") not in STATES:
            errors.append(f"persons[{i}].state={p.get('state')!r} not in {sorted(STATES)}")
        for key in ("track_id", "cx_pct", "cy_pct", "dwell_duration", "confidence"):
            if key not in p:
                errors.append(f"persons[{i}].{key} missing")
        for group, keys in (("bbox", "xywh"), ("velocity", ("vx", "vy", "speed_m_s"))):
            if group not in p:
                errors.append(f"persons[{i}].{group} missing")
                continue
            for key in keys:
                if key not in p[group]:
                    errors.append(f"persons[{i}].{group}.{key} missing")
    ids = [p.get("track_id") for p in persons]
    if len(ids) != len(set(ids)):
        errors.append("duplicate track_id inside one batch")
    counted = zone.get("browsing_count", 0) + zone.get("engaged_count", 0) + zone.get("assist_count", 0)
    if persons and counted > len(persons):
        errors.append(f"zone state counts ({counted}) exceed persons in batch ({len(persons)})")
    return errors


def main():
    stream = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    total = bad = 0
    max_persons = 0
    for line in stream:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            bad += 1
            print(f"[{total}] not JSON: {exc}")
            continue
        max_persons = max(max_persons, len(msg.get("persons", [])))
        errors = check(msg)
        if errors:
            bad += 1
            for e in errors:
                print(f"[{total}] {e}")
    print(f"checked={total} invalid={bad} max_persons_in_one_message={max_persons}")
    return 1 if bad or total == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
