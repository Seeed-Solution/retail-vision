#!/usr/bin/env python3
"""vb_cpu_split — §10.5 ② Python / native CPU share check.

Polls ``/healthz`` (BASE-1 §6.8) twice -- once at start, once after
``--duration`` seconds (intermediate samples are taken but only the first and
last are used for the delta, matching the spec's "取首尾差值"). Per-PID
cumulative ``cpu_s`` comes from the healthz body, which the supervisor builds
from ``procstat.py`` (``/proc/<pid>/stat`` utime+stime ÷ CLK_TCK).

Python processes = supervisor + ``shards[].pid``; native processes =
``shards[].runtime.pid``. Judgement:

  py_share = Σ Python ΔCPU / Σ all ΔCPU          ≤ --max-py-share (0.15)
  py_core  = Σ Python ΔCPU / duration             ≤ --max-py-core  (0.35)

Also reports the sum of ``shards[].hook_budget.core`` (present from M1.22 on)
so an over-budget run can be attributed. Writes the JSON report to --out and
exits 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def fetch_healthz(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pids_cpu(h: dict) -> tuple[dict[int, float | None], dict[int, float | None]]:
    """Return ({python_pid: cpu_s}, {native_pid: cpu_s}) from a healthz body."""
    py: dict[int, float | None] = {}
    native: dict[int, float | None] = {}
    sup = h.get("supervisor") or {}
    if isinstance(sup.get("pid"), int):
        py[sup["pid"]] = sup.get("cpu_s")
    for sh in h.get("shards") or []:
        if isinstance(sh.get("pid"), int):
            py[sh["pid"]] = sh.get("cpu_s")
        rt = sh.get("runtime") or {}
        if isinstance(rt.get("pid"), int):
            native[rt["pid"]] = rt.get("cpu_s")
    return py, native


def _delta(a: dict[int, float | None], b: dict[int, float | None]) -> tuple[dict[int, float], list[int]]:
    """ΔCPU per PID present in both samples; second return = PIDs that vanished."""
    out: dict[int, float] = {}
    vanished: list[int] = []
    for pid, cpu0 in a.items():
        cpu1 = b.get(pid)
        if cpu1 is None or cpu0 is None:
            vanished.append(pid)
        else:
            out[pid] = cpu1 - cpu0
    for pid in b:
        if pid not in a:
            vanished.append(pid)
    return out, vanished


def evaluate(first: dict, last: dict, duration: float,
             max_py_share: float, max_py_core: float) -> dict:
    py0, nat0 = _pids_cpu(first)
    py1, nat1 = _pids_cpu(last)
    py_d, py_gone = _delta(py0, py1)
    nat_d, nat_gone = _delta(nat0, nat1)
    py_total = sum(py_d.values())
    nat_total = sum(nat_d.values())
    all_total = py_total + nat_total
    py_share = (py_total / all_total) if all_total > 0 else 0.0
    py_core = py_total / duration if duration > 0 else 0.0
    hook_budget_core = 0.0
    for sh in last.get("shards") or []:
        hb = sh.get("hook_budget") or {}
        if isinstance(hb.get("core"), (int, float)):
            hook_budget_core += float(hb["core"])
    failures = []
    if py_gone or nat_gone:
        failures.append(f"process set changed during run "
                        f"(vanished pids: {sorted(set(py_gone + nat_gone))})")
    if py_share > max_py_share:
        failures.append(f"py_share {py_share:.4f} > {max_py_share}")
    if py_core > max_py_core:
        failures.append(f"py_core {py_core:.4f} > {max_py_core}")
    return {
        "duration_s": duration,
        "python": {"pids": {str(k): round(v, 4) for k, v in sorted(py_d.items())},
                   "cpu_s_total": round(py_total, 4)},
        "native": {"pids": {str(k): round(v, 4) for k, v in sorted(nat_d.items())},
                   "cpu_s_total": round(nat_total, 4)},
        "py_share": round(py_share, 4),
        "py_core": round(py_core, 4),
        "hook_budget_core_sum": round(hook_budget_core, 4),
        "max_py_share": max_py_share,
        "max_py_core": max_py_core,
        "pass": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vb_cpu_split", description="BASE-1 §10.5 ② CPU split check")
    parser.add_argument("--healthz", required=True, help="http://<dev>:8099/healthz")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=5.0,
                        help="sampling interval; only first/last are used for deltas")
    parser.add_argument("--max-py-share", type=float, default=0.15)
    parser.add_argument("--max-py-core", type=float, default=0.35)
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args(argv)

    samples = [fetch_healthz(args.healthz)]
    t0 = time.monotonic()
    while True:
        time.sleep(min(args.interval, max(0.0, args.duration - (time.monotonic() - t0))))
        if time.monotonic() - t0 >= args.duration - 1e-6:
            break
        samples.append(fetch_healthz(args.healthz))
    samples.append(fetch_healthz(args.healthz))
    elapsed = time.monotonic() - t0

    report = evaluate(samples[0], samples[-1], elapsed,
                      args.max_py_share, args.max_py_core)
    report["healthz"] = args.healthz
    report["samples"] = len(samples)
    report["measured_duration_s"] = round(elapsed, 2)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
