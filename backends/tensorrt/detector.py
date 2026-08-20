"""TensorRT person detection backend for NVIDIA Jetson.

Same contract as backends/rknn: `detect(frame) -> ([DetectionBox], ms)`.
Everything above that call -- tracking, dwell state, zone metrics, payload,
publishing -- is retail_core and is byte-identical to what the RK boards run.

The engine is an ultralytics YOLO11n `.engine` built on the board it runs on
(a TensorRT plan is tied to the GPU arch and the TensorRT build that produced
it, so it is never baked into an image). Building the plan and running it are
separate jobs, though: the plan is produced once by the ultralytics image and
from then on executed by `tensorrt` directly -- the Python binding the host
already mounts into the container. That drops torch, ultralytics, onnx, scipy
and polars (~1.15 GB) out of the runtime image.

What ultralytics used to do on either side of the plan is done here instead:

* preprocess -- RGB HWC uint8 -> NCHW float32 /255. The video sources hand out
  RGB and the exported plan expects RGB, so nothing is channel-swapped. (The
  previous implementation flipped to BGR only because ultralytics flips an
  ndarray source back to RGB internally; going straight to TensorRT that flip
  would run the model on swapped pixels.)
* postprocess -- the export is a raw `[1, 84, 8400]` head: rows 0..3 are box
  cx,cy,w,h in input pixels, rows 4..83 are per-class scores already through
  sigmoid. Only row 4 (COCO class 0, person) is read, then threshold,
  xywh -> xyxy and NMS.
* unwrap the file -- an ultralytics `.engine` is not a bare TensorRT plan. It
  is a 4-byte little-endian length, that many bytes of JSON metadata (imgsz,
  class names, build settings) and only then the plan. Handing the whole file
  to `deserializeCudaEngine` fails the plan magic-tag assertion.
"""
from __future__ import annotations

import json
import time

import numpy as np

from retail_core.tracker import DetectionBox

from . import cuda_rt

PERSON_CLASS = 0


def read_plan(path):
    """Split an ultralytics `.engine` into (serialized plan, metadata dict).

    A bare TensorRT plan (`trtexec` output, say) has no header and is returned
    unchanged with empty metadata.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) > 5 and raw[4:5] == b"{":
        length = int.from_bytes(raw[:4], "little")
        if 0 < length <= len(raw) - 4:
            try:
                return raw[4 + length:], json.loads(raw[4:4 + length].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
    return raw, {}


def _nms(boxes, scores, threshold):
    """Greedy IoU NMS over xyxy boxes, highest score first."""
    order = scores.argsort()[::-1]
    keep = []
    areas = (np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
             * np.maximum(0.0, boxes[:, 3] - boxes[:, 1]))
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= threshold]
    return keep


def decode_person(output, confidence=0.35, nms_threshold=0.45, input_size=640):
    """`[1, 84, N]` (or `[1, N, 84]`) YOLO head -> person boxes in model pixels.

    Returns [{'box': [x1, y1, x2, y2], 'score': float}].
    """
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"unexpected detection head shape: {np.shape(output)}")
    # Ultralytics exports channels-first (84 x 8400). Accept the transposed
    # layout too rather than assume, since it is one comparison to be certain.
    if array.shape[0] > array.shape[1]:
        array = array.T
    channels = array.shape[0]
    if channels < 5:
        raise ValueError(f"detection head has too few channels: {channels}")
    scores = array[4 + PERSON_CLASS]
    selected = np.flatnonzero(scores >= confidence)
    if not selected.size:
        return []
    cx, cy, w, h = array[0:4, selected]
    half_w, half_h = w * 0.5, h * 0.5
    boxes = np.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), axis=1)
    boxes = boxes.clip(0.0, float(input_size))
    scores = scores[selected]
    return [{"box": boxes[i].astype(float).tolist(), "score": float(scores[i])}
            for i in _nms(boxes, scores, nms_threshold)]


class TensorRTPersonDetector:
    """Backend adapter. `detect()` is the entire contract with the core."""

    def __init__(self, model: str, input_size: int = 640):
        import tensorrt as trt

        self.trt = trt
        self.input_size = int(input_size)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        plan, self.metadata = read_plan(model)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {model}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self.stream = cuda_rt.stream_create()
        self.device_buffers = {}
        self.host_outputs = {}
        self.input_name = None
        self.output_names = []
        self._bind()

    # -- setup ------------------------------------------------------------
    def _bind(self):
        trt = self.trt
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = list(self.engine.get_tensor_shape(name))
            if is_input:
                if self.input_name is not None:
                    raise RuntimeError("engine has more than one input tensor")
                self.input_name = name
                # A dynamic batch (-1) is pinned to 1: one letterboxed frame
                # per detect() call is the whole contract with the core.
                shape = [1 if dim < 0 else int(dim) for dim in shape]
                self.context.set_input_shape(name, tuple(shape))
                self.input_shape = tuple(shape)
                self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            else:
                self.output_names.append(name)
        if self.input_name is None:
            raise RuntimeError("engine exposes no input tensor")

        # Host staging is page-locked. A pageable source or destination makes
        # the driver route the copy through its own staging buffer, so the
        # "async" transfer cannot overlap with compute and the stream gains
        # nothing. Buffers are fixed size — the video source letterboxes to the
        # engine's input square before handing a frame over — so there is no
        # resize path to grow them on.
        self.host_pointers = []
        self.host_input, pointer = cuda_rt.pinned_array(self.input_shape,
                                                        self.input_dtype)
        self.host_pointers.append(pointer)
        self.device_buffers[self.input_name] = cuda_rt.malloc(self.host_input.nbytes)
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            host, pointer = cuda_rt.pinned_array(shape, dtype)
            self.host_pointers.append(pointer)
            self.host_outputs[name] = host
            self.device_buffers[name] = cuda_rt.malloc(host.nbytes)
        for name, pointer in self.device_buffers.items():
            self.context.set_tensor_address(name, int(pointer.value))

    def describe(self):
        """Binding report, printed once at startup so a layout change is visible."""
        parts = [f"input {self.input_name}{self.input_shape}"
                 f":{np.dtype(self.input_dtype).name}"]
        parts += [f"output {name}{self.host_outputs[name].shape}"
                  f":{self.host_outputs[name].dtype.name}"
                  for name in self.output_names]
        for key in ("imgsz", "task", "version", "batch"):
            if key in self.metadata:
                parts.append(f"{key}={self.metadata[key]}")
        return "; ".join(parts)

    # -- inference --------------------------------------------------------
    def _preprocess(self, rgb: np.ndarray):
        # HWC uint8 RGB -> NCHW float32 in [0, 1], written into the reusable
        # host staging buffer so steady state allocates nothing.
        chw = rgb.transpose(2, 0, 1)
        np.multiply(chw, np.float32(1.0 / 255.0), out=self.host_input[0],
                    dtype=np.float32, casting="unsafe")
        return self.host_input

    def infer(self, rgb: np.ndarray):
        host_input = self._preprocess(rgb)
        start = time.perf_counter()
        cuda_rt.memcpy_host_to_device(self.device_buffers[self.input_name],
                                      host_input, self.stream)
        if not self.context.execute_async_v3(int(self.stream.value)):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        for name in self.output_names:
            cuda_rt.memcpy_device_to_host(self.host_outputs[name],
                                          self.device_buffers[name], self.stream)
        cuda_rt.stream_synchronize(self.stream)
        elapsed = (time.perf_counter() - start) * 1000.0
        return [self.host_outputs[name] for name in self.output_names], elapsed

    def detect(self, frame, score_threshold=0.35, nms_threshold=0.45):
        """frame: HWC uint8 RGB, already letterboxed to the model input size."""
        size = frame.shape[0]
        outputs, inference_ms = self.infer(frame)
        boxes = decode_person(outputs[0], score_threshold, nms_threshold, size)
        detections = [
            DetectionBox((b["box"][0] + b["box"][2]) / 2 / size,
                         (b["box"][1] + b["box"][3]) / 2 / size,
                         (b["box"][2] - b["box"][0]) / size,
                         (b["box"][3] - b["box"][1]) / size,
                         b["score"])
            for b in boxes
        ]
        return detections, inference_ms

    def close(self):
        for pointer in self.device_buffers.values():
            cuda_rt.free(pointer)
        self.device_buffers = {}
        # The numpy views point at CUDA-owned memory; drop them before the
        # pointers are released so nothing can read freed pages.
        self.host_input = None
        self.host_outputs = {}
        for pointer in getattr(self, "host_pointers", []):
            cuda_rt.host_free(pointer)
        self.host_pointers = []
        if getattr(self, "stream", None) is not None:
            cuda_rt.stream_destroy(self.stream)
            self.stream = None
        self.context = None
        self.engine = None
        self.runtime = None
