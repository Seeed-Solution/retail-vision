#!/usr/bin/env python3
"""Generate a constant-output ONNX graph for CPU-backend tests (BASE-1 §7/§8 M1.9).

The graph has a single float input `images` [1, 3, model_h, model_w] and a
single output `output` [1, n_anchors, 5 + n_cls] whose values come from
contracts/fixtures/vb/yolox_out_case1.bin, so running the CPU backend on any
frame must reproduce the fixture detections exactly (up to float rounding).

Usage: gen_const_onnx.py OUT.onnx [--fixture PATH_PREFIX]
(uses onnx from the dev dependency group; not imported by vision_base code)
"""
from __future__ import annotations

import argparse
import pathlib
import struct

import onnx
from onnx import TensorProto, helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument(
        "--fixture",
        default=str(
            pathlib.Path(__file__).resolve().parents[3]
            / "contracts/fixtures/vb/yolox_out_case1"
        ),
        help="fixture path prefix (json+bin)",
    )
    args = ap.parse_args()

    import json

    fx = json.loads(pathlib.Path(args.fixture + ".json").read_text())
    raw = pathlib.Path(args.fixture + ".bin").read_bytes()
    n = fx["n_anchors"] * (fx["n_cls"] + 5)
    vals = struct.unpack(f"<{n}f", raw[: 4 * n])
    out_shape = [1, fx["n_anchors"], fx["n_cls"] + 5]
    out_tensor = helper.make_tensor(
        "const_out", TensorProto.FLOAT, out_shape, list(vals)
    )

    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=out_tensor)],
        "const-output",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, fx["model_h"], fx["model_w"]]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, out_shape
            )
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
        producer_name="vision_base/gen_const_onnx",
    )
    model.ir_version = 8  # readable by ORT 1.20
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"wrote {args.out}: output shape {out_shape}")


if __name__ == "__main__":
    main()
