"""Scalar letterbox coordinate mapping (spec BASE-1 §6.4 "letterbox.py").

Pure scalar math only (``math``), mirrors the C++ ``src/letterbox.cpp``
formulas. Coordinates are normalized: model-canvas normalized in,
source-frame normalized out.
"""
from __future__ import annotations

from .types import LetterboxGeom

_ALIGNS = ("center", "top_left")


def _check_size(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def fit(src_w: int, src_h: int, model_w: int, model_h: int,
        align: str = "center") -> LetterboxGeom:
    """Fit (src_w, src_h) into the (model_w, model_h) canvas."""
    _check_size("src_w", src_w)
    _check_size("src_h", src_h)
    _check_size("model_w", model_w)
    _check_size("model_h", model_h)
    if align not in _ALIGNS:
        raise ValueError(f"align must be one of {_ALIGNS}, got {align!r}")
    scale = min(model_w / src_w, model_h / src_h)
    sw = round(src_w * scale)
    sh = round(src_h * scale)
    if align == "center":
        pad_x = (model_w - sw) // 2
        pad_y = (model_h - sh) // 2
    else:  # top_left
        pad_x = 0
        pad_y = 0
    return LetterboxGeom(src_w=src_w, src_h=src_h, model_w=model_w, model_h=model_h,
                         scale=scale, pad_x=float(pad_x), pad_y=float(pad_y),
                         align=align)


def to_source_norm(geom: LetterboxGeom, mx: float, my: float) -> tuple[float, float]:
    """Map a model-canvas normalized point to source-frame normalized."""
    px = mx * geom.model_w
    py = my * geom.model_h
    return ((px - geom.pad_x) / geom.scale) / geom.src_w, \
           ((py - geom.pad_y) / geom.scale) / geom.src_h


def to_model_norm(geom: LetterboxGeom, sx: float, sy: float) -> tuple[float, float]:
    """Forward transform: source-frame normalized -> model-canvas normalized."""
    px = (sx * geom.src_w) * geom.scale + geom.pad_x
    py = (sy * geom.src_h) * geom.scale + geom.pad_y
    return px / geom.model_w, py / geom.model_h


def box_to_source_norm(geom: LetterboxGeom, cx: float, cy: float,
                       w: float, h: float) -> tuple[float, float, float, float]:
    """Map a model-canvas normalized (center, size) box to source-frame normalized."""
    scx, scy = to_source_norm(geom, cx, cy)
    sw = w * geom.model_w / (geom.scale * geom.src_w)
    sh = h * geom.model_h / (geom.scale * geom.src_h)
    return scx, scy, sw, sh
