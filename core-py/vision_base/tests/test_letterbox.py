"""Letterbox tests against the hand-calculated cross-language fixture."""
from __future__ import annotations

import json
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from vision_base.letterbox import box_to_source_norm, fit, to_model_norm, to_source_norm
from vision_base.types import Detection, LetterboxGeom

FIXTURE = pathlib.Path(__file__).resolve().parents[3] / "contracts" / "fixtures" / "vb" / "letterbox_cases.json"

CASES = json.loads(FIXTURE.read_text())["cases"]


def test_fixture_expectations():
    for case in CASES:
        src_w, src_h = case["src"]
        model_w, model_h = case["model"]
        geom = fit(src_w, src_h, model_w, model_h, case["align"])
        for pt in case["points"]:
            sx, sy = to_source_norm(geom, *pt["model"])
            assert abs(sx - pt["expect"][0]) <= 1e-6, (case, pt, sx)
            assert abs(sy - pt["expect"][1]) <= 1e-6, (case, pt, sy)


def test_roundtrip_100_random_points():
    rng = random.Random(0)
    for case in CASES:
        src_w, src_h = case["src"]
        model_w, model_h = case["model"]
        geom = fit(src_w, src_h, model_w, model_h, case["align"])
        for _ in range(100):
            mx = rng.random()
            my = rng.random()
            sx, sy = to_source_norm(geom, mx, my)
            bx, by = to_model_norm(geom, sx, sy)
            assert abs(bx - mx) <= 1e-9
            assert abs(by - my) <= 1e-9


def test_center_matches_retail_letterbox_correction():
    from retail_core.payload import letterbox_correction

    for case in CASES:
        if case["align"] != "center":
            continue
        src_w, src_h = case["src"]
        model_w, model_h = case["model"]
        scale = min(model_w / src_w, model_h / src_h)
        if (src_w * scale) % 1 or (src_h * scale) % 1:
            # retail's formula uses the unrounded letterbox; the base rounds the
            # resized image to whole pixels, so the two only agree exactly when
            # the scaled size is already integral (rounding-boundary cases differ
            # by < 1 model pixel by design).
            continue
        rsx, rsy, rox, roy = letterbox_correction(src_w, src_h, model_w, model_h)
        geom = fit(src_w, src_h, model_w, model_h, "center")
        rng = random.Random(1)
        for _ in range(10):
            mx, my = rng.random(), rng.random()
            sx, sy = to_source_norm(geom, mx, my)
            ref_x = (mx - rox) / rsx
            ref_y = (my - roy) / rsy
            assert abs(sx - ref_x) <= 1e-6, (case, mx, sx, ref_x)
            assert abs(sy - ref_y) <= 1e-6, (case, my, sy, ref_y)


def test_box_to_source_norm():
    geom = fit(1280, 720, 640, 640, "center")
    cx, cy, w, h = box_to_source_norm(geom, 0.5, 0.5, 0.2, 0.3)
    # center maps like the point transform; w scales by model_w/(scale*src_w) = 1
    sx, sy = to_source_norm(geom, 0.5, 0.5)
    assert (cx, cy) == (sx, sy)
    assert abs(w - 0.2) <= 1e-9
    assert abs(h - 0.3 * 640 / (0.5 * 720)) <= 1e-9  # 0.3 * 640/360 = 0.5333...


def test_invalid_inputs():
    import pytest

    for bad in [(0, 720), (1280, 0), (-1, 720), (1280, -5)]:
        with pytest.raises(ValueError):
            fit(bad[0], bad[1], 640, 640)
    for bad in [(640, 0), (0, 640)]:
        with pytest.raises(ValueError):
            fit(1280, 720, bad[0], bad[1])
    with pytest.raises(ValueError):
        fit(1280, 720, 640, 640, "bottom_center")


def test_types_smoke():
    import dataclasses

    det = Detection(cx=0.5, cy=0.5, w=0.1, h=0.2, score=0.9, class_id=0, track_id=1)
    assert det.keypoints == ()
    assert det.attrs == ()
    geom = fit(1280, 720, 640, 640)
    assert isinstance(geom, LetterboxGeom)
    with pytest.raises(dataclasses.FrozenInstanceError):
        geom.scale = 2.0
