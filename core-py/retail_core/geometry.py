"""Polygon and directed-segment helpers.

Pure analytics logic: no inference SDK, no board library, no I/O.
Behaviour authority is the reCamera C++ implementation at
solutions/retail-vision/main/person_tracker.cpp

``point_in_polygon`` is vision_base's ray-casting test (same edge rule and
arithmetic; pinned by tests/fixtures/golden/math.json). The directed segment
crossing used for entry/exit counting is retail policy and stays here.
"""
from __future__ import annotations

from vision_base.geom import point_in_polygon

__all__ = ["point_in_polygon", "segment_crossing"]


def _cross(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def segment_crossing(ax, ay, bx, by, p0x, p0y, p1x, p1y):
    """Directed crossing of p0->p1 over segment a->b. +1 left->right, -1 right->left."""
    d0 = _cross(ax, ay, bx, by, p0x, p0y)
    d1 = _cross(ax, ay, bx, by, p1x, p1y)
    if (d0 > 0) == (d1 > 0) or d0 == 0 or d1 == 0:
        return 0
    e0 = _cross(p0x, p0y, p1x, p1y, ax, ay)
    e1 = _cross(p0x, p0y, p1x, p1y, bx, by)
    if (e0 > 0) == (e1 > 0):
        return 0
    return -1 if d0 > 0 else 1
