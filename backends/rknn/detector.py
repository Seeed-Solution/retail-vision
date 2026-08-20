"""RKNN Lite person detection backend.

The only layer that may not be shared: tensor layout, pre/post-processing and
the ABI-locked host library are all runtime-specific. It exposes exactly one
thing to the core: `detect(frame) -> [DetectionBox]`.

The model is the YOLO11n "raw head" export shared with the fall-detection
solution: a single-class (person) DFL head emitting, per stride, a 1-channel
class map, a 64-channel box distribution and a 51-channel keypoint map. Only
the class and box tensors are decoded here; the keypoint tensors are ignored,
which makes this a person detector at strictly lower cost than pose decode.
"""
from __future__ import annotations

import time

import numpy as np

from retail_core.tracker import DetectionBox

_PROJ = np.arange(16, dtype=np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _nchw(output):
    a = np.asarray(output)
    if a.ndim == 3:
        a = a[None]
    channels = (1, 51, 64)
    if a.ndim == 4 and a.shape[1] not in channels and a.shape[-1] in channels:
        a = a.transpose(0, 3, 1, 2)
    return a


def _nms(boxes, scores, threshold):
    order = scores.argsort()[::-1]
    keep = []
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
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
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= threshold]
    return keep


def decode_person(outputs, confidence=0.35, nms_threshold=0.45, input_size=640):
    """Return [{'box': [x1,y1,x2,y2] in model px, 'score': float}] for class person."""
    outs = [_nchw(x) for x in outputs]
    cls_maps, box_maps = [], []
    for x in outs:
        if x.ndim != 4:
            continue
        if x.shape[1] == 1:
            cls_maps.append(x)
        elif x.shape[1] == 64:
            box_maps.append(x)
    cls_maps.sort(key=lambda x: -x.shape[-1])
    box_maps.sort(key=lambda x: -x.shape[-1])
    boxes, scores_all = [], []
    for bd, cl in zip(box_maps, cls_maps):
        _, _, h, w = bd.shape
        stride = input_size // h
        n = h * w
        score = cl.reshape(-1, n).max(0).astype(np.float32)
        if score.min(initial=0) < 0 or score.max(initial=0) > 1:
            score = _sigmoid(score)
        selected = np.flatnonzero(score >= confidence)
        if not selected.size:
            continue
        gx = (selected % w).astype(np.float32)
        gy = (selected // w).astype(np.float32)
        dist = (_softmax(bd.reshape(4, 16, n)[:, :, selected].astype(np.float32))
                * _PROJ[None, :, None]).sum(1)
        boxes.append(np.stack(((gx + .5 - dist[0]) * stride, (gy + .5 - dist[1]) * stride,
                               (gx + .5 + dist[2]) * stride, (gy + .5 + dist[3]) * stride), 1))
        scores_all.append(score[selected])
    if not boxes:
        return []
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores_all)
    return [{"box": boxes[i].clip(0, input_size).astype(float).tolist(),
             "score": float(scores[i])}
            for i in _nms(boxes, scores, nms_threshold)]


class RKNNPersonDetector:
    """Backend adapter. `detect()` is the entire contract with the core."""

    def __init__(self, model: str, core_mask=None):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model)
        if ret:
            raise RuntimeError(f"load_rknn({model})={ret}")
        ret = self.rknn.init_runtime(**({"core_mask": core_mask} if core_mask is not None else {}))
        if ret:
            raise RuntimeError(f"init_runtime({model})={ret}")

    def infer(self, rgb: np.ndarray):
        x = np.ascontiguousarray(rgb[None], dtype=np.uint8)
        start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[x])
        elapsed = (time.perf_counter() - start) * 1000.0
        if outputs is None:
            raise RuntimeError("RKNN inference returned None")
        return outputs, elapsed

    def close(self):
        self.rknn.release()

    def detect(self, frame, score_threshold=0.35, nms_threshold=0.45):
        """frame: HWC uint8 RGB, already letterboxed to the model input size."""
        size = frame.shape[0]
        outputs, inference_ms = self.infer(frame)
        boxes = decode_person(outputs, score_threshold, nms_threshold, size)
        detections = [
            DetectionBox((b["box"][0] + b["box"][2]) / 2 / size,
                         (b["box"][1] + b["box"][3]) / 2 / size,
                         (b["box"][2] - b["box"][0]) / size,
                         (b["box"][3] - b["box"][1]) / size,
                         b["score"])
            for b in boxes
        ]
        return detections, inference_ms
