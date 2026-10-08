"""Generate identity ONNX models whose output exposes CpuBackend input pixels."""
from pathlib import Path
import sys

import onnx
from onnx import TensorProto, helper


def main(out: Path, size: int) -> None:
    value = helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, 3, size, size])
    result = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, size, size])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["in"], ["out"])],
        f"cpu_preprocess_identity_{size}", [value], [result],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: gen_cpu_preprocess_model.py OUT.onnx SIZE")
    main(Path(sys.argv[1]), int(sys.argv[2]))
