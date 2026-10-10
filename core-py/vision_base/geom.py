"""Layer-3 hook geometry helpers (spec BASE-1 §6.5.5, M1.22).

Pure standard library; small-list structured results only (no pixels,
no tensors). Part of the stable surface (§6.14).
"""
from __future__ import annotations

import math

from .letterbox import box_to_source_norm, to_source_norm
from .types import Detection, FrameResult

__all__ = ["source_boxes", "source_keypoints", "point_in_polygon",
           "joint_angle_deg"]


def source_boxes(res: FrameResult):
    """[(track_id, class_id, score, (x0, y0, x1, y1))] in source-normalized
    coordinates, via letterbox.box_to_source_norm."""
    out = []
    for d in res.detections:
        cx, cy, w, h = box_to_source_norm(res.geom, d.cx, d.cy, d.w, d.h)
        out.append((d.track_id, d.class_id, d.score,
                    (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)))
    return out


def source_keypoints(res: FrameResult, det: Detection):
    """Source-normalized (x, y, conf) for one detection's keypoints."""
    out = []
    for i in range(0, len(det.keypoints) - 2, 3):
        x, y, conf = det.keypoints[i:i + 3]
        sx, sy = to_source_norm(res.geom, x, y)
        out.append((sx, sy, conf))
    return out


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Ray casting; `polygon` is a list of [x, y] source-normalized points."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y) and \
                x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def joint_angle_deg(a, b, c, src_w: int, src_h: int) -> float:
    """∠ABC in degrees [0, 180], computed in source pixels (same formula as
    the pose_angle analyzer, §6.2.5): a/b/c are source-normalized (x, y)."""
    ax, ay = a[0] * src_w, a[1] * src_h
    bx, by = b[0] * src_w, b[1] * src_h
    cx, cy = c[0] * src_w, c[1] * src_h
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        raise ValueError("degenerate joint points (coincident)")
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))
