"""Small CPU preprocessor oracles; all model outputs expose input pixels."""
from pathlib import Path
import sys
import onnx
from onnx import helper, TensorProto
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for width in (2, 4):
    shape = helper.make_tensor('shape', TensorProto.INT64, [2], [3, width * 2])
    graph = helper.make_graph([helper.make_node('Constant', [], ['shape'], value=shape),
                              helper.make_node('Reshape', ['in', 'shape'], ['out'])], 'pixels',
                              [helper.make_tensor_value_info('in', TensorProto.FLOAT, [1, 3, 2, width])],
                              [helper.make_tensor_value_info('out', TensorProto.FLOAT, [3, width * 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)]); model.ir_version = 8
    onnx.save(model, out / f'pixels{width}.onnx')
for name, dtype, shape in [('badbatch', TensorProto.FLOAT, [2, 3, 4]), ('badtype', TensorProto.INT64, [3, 4])]:
    tensor = helper.make_tensor('constant', dtype, shape, [0] * (24 if name == 'badbatch' else 12))
    graph = helper.make_graph([helper.make_node('Constant', [], ['out'], value=tensor)], name,
                              [helper.make_tensor_value_info('in', TensorProto.FLOAT, [1, 3, 2, 2])],
                              [helper.make_tensor_value_info('out', dtype, shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)]); model.ir_version = 8
    onnx.save(model, out / f'{name}.onnx')
print('generated stage2 pixel/invalid models')
