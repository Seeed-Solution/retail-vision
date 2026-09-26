"""Process statistics from /proc (spec BASE-1 §5.3: procstat.py).

Reads ``/proc/<pid>/stat`` (CPU seconds) and ``/proc/<pid>/status`` (VmRSS).
On platforms without /proc (macOS) returns None per field.
"""
from __future__ import annotations

import os

__all__ = ["sample"]

_CLK_TCK: int | None = None


def _clk_tck() -> int:
    global _CLK_TCK
    if _CLK_TCK is None:
        try:
            _CLK_TCK = os.sysconf("SC_CLK_TCK")
        except (ValueError, OSError, AttributeError):
            _CLK_TCK = 100
    return _CLK_TCK


def sample(pid: int) -> dict:
    """Return {"cpu_s": float | None, "rss_kb": int | None} for *pid*.

    Fields are None where /proc is unavailable (macOS) or the process is gone.
    """
    out: dict = {"cpu_s": None, "rss_kb": None}
    if not isinstance(pid, int) or pid <= 0 or not os.path.isdir("/proc"):
        return out
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
        # fields after ")<state>" are 1-based from field 3; utime is field 14,
        # stime field 15 -> after the state token they are indices 11, 12.
        rparen = data.rfind(b")")
        parts = data[rparen + 2:].split()
        utime, stime = int(parts[11]), int(parts[12])
        out["cpu_s"] = (utime + stime) / _clk_tck()
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(f"/proc/{pid}/status", "r", encoding="ascii", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_kb"] = int(line.split()[1])
                    break
    except (OSError, ValueError, IndexError):
        pass
    return out
