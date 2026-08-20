"""Polygon and directed-segment helpers.

Pure analytics logic: no inference SDK, no board library, no I/O.
Behaviour authority is the reCamera C++ implementation at
solutions/retail-vision/main/person_tracker.cpp
"""
from __future__ import annotations


def point_in_polygon(px, py, polygon):
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi:
                inside = not inside
        j = i
    return inside


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


