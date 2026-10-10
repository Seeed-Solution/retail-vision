"""Exact-value goldens for the retail geometry helpers.

Recorded on the pre-vision_base implementation. Floats are stored with
``repr`` round-tripping (``json`` does this), so equality below is exact,
not approximate: a replacement implementation must reproduce every bit.

Regenerate (only when a numeric change is intended):
    RETAIL_GOLDEN_REGEN=1 uv run pytest core-py/retail_core/tests/test_math_golden.py
"""
from __future__ import annotations

import json
import os
import pathlib
import random

from retail_core.geometry import point_in_polygon, segment_crossing
from retail_core.payload import letterbox_correction
from retail_core.video import aspect_fit_geometry

GOLDEN = pathlib.Path(__file__).resolve().parent / "fixtures" / "golden" / "math.json"
REGEN = os.environ.get("RETAIL_GOLDEN_REGEN") == "1"

# Frame sizes a camera/config produces and model inputs the backends use.
FRAMES = [(640, 480), (704, 576), (800, 600), (1280, 720), (1280, 960),
          (1920, 1080), (2560, 1440), (2592, 1944), (2688, 1520), (3840, 2160),
          (1080, 1920), (720, 1280), (640, 640), (1000, 999), (999, 1000),
          (1366, 768), (1, 1), (3, 7), (4096, 2160)]
MODELS = [(320, 320), (416, 416), (480, 480), (512, 512), (640, 640),
          (960, 960), (1280, 1280), (640, 384), (384, 640)]
SQUARE_SIZES = [320, 416, 480, 512, 640, 960, 1280]

POLYGONS = [
    [(0.2, 0.2), (0.8, 0.2), (0.8, 0.9), (0.2, 0.9)],
    [(0.1, 0.1), (0.9, 0.15), (0.5, 0.95)],
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.6, 0.4), (0.0, 1.0)],   # concave
    [(0.3, 0.3), (0.7, 0.3), (0.7, 0.3), (0.3, 0.7)],               # repeated vertex
    [(0.5, 0.5), (0.6, 0.6)],                                       # degenerate
    [],
]


def compute():
    out = {"letterbox_correction": [], "aspect_fit_geometry": [],
           "point_in_polygon": [], "segment_crossing": []}
    for fw, fh in FRAMES:
        for mw, mh in MODELS:
            out["letterbox_correction"].append(
                [fw, fh, mw, mh, list(letterbox_correction(fw, fh, mw, mh))])
        for size in SQUARE_SIZES:
            out["aspect_fit_geometry"].append(
                [fw, fh, size, list(aspect_fit_geometry(fw, fh, size))])
    grid = [i / 20.0 for i in range(-1, 22)]
    for k, poly in enumerate(POLYGONS):
        bits = "".join("1" if point_in_polygon(x, y, poly) else "0"
                       for x in grid for y in grid)
        out["point_in_polygon"].append([k, bits])
    rng = random.Random(20261010)
    for _ in range(400):
        pts = [round(rng.uniform(-0.2, 1.2), 3) for _ in range(8)]
        out["segment_crossing"].append([pts, segment_crossing(*pts)])
    # exact-hit cases (endpoint on the line) are handled specially
    out["segment_crossing"].append([[0, 0, 1, 0, 0.5, -1, 0.5, 0], segment_crossing(0, 0, 1, 0, 0.5, -1, 0.5, 0)])
    out["segment_crossing"].append([[0, 0, 1, 0, 0.5, -1, 0.5, 1], segment_crossing(0, 0, 1, 0, 0.5, -1, 0.5, 1)])
    out["segment_crossing"].append([[0, 0, 1, 0, 0.5, 1, 0.5, -1], segment_crossing(0, 0, 1, 0, 0.5, 1, 0.5, -1)])
    return out


def test_math_golden():
    observed = json.loads(json.dumps(compute()))
    if REGEN:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(observed, sort_keys=True) + "\n")
    golden = json.loads(GOLDEN.read_text())
    for key in golden:
        assert observed[key] == golden[key], key


def test_aspect_fit_rejects_non_positive():
    import pytest
    for args in ((0, 10, 640), (10, 0, 640), (10, 10, 0), (-1, 10, 640)):
        with pytest.raises(ValueError):
            aspect_fit_geometry(*args)


def test_vision_base_letterbox_cross_check():
    """Why the two letterbox helpers are not replaced by vision_base.letterbox.

    aspect_fit_geometry agrees with letterbox.fit on scale and padding for
    every grid input, but fit() does not return the scaled size the video
    sources need, so a wrapper would recompute round(src * scale) anyway.
    letterbox_correction uses the unrounded fit while vision_base rounds the
    resized image to whole pixels, so mapped coordinates differ for
    non-integral scales; the strict wire contract keeps the retail formula.
    """
    from vision_base.letterbox import fit, to_source_norm

    for fw, fh, size, (sw, sh, px, py) in compute()["aspect_fit_geometry"]:
        geom = fit(fw, fh, size, size)
        assert (geom.pad_x, geom.pad_y) == (px, py)
        assert (round(fw * geom.scale), round(fh * geom.scale)) == (sw, sh)

    differs = 0
    for fw, fh, mw, mh, (sx, sy, ox, oy) in compute()["letterbox_correction"]:
        geom = fit(fw, fh, mw, mh)
        vx, vy = to_source_norm(geom, 0.25, 0.75)
        differs += (vx, vy) != ((0.25 - ox) / sx, (0.75 - oy) / sy)
    assert differs > 0
