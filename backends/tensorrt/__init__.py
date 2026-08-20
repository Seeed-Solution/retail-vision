"""TensorRT backend: YOLO11 person detector on NVIDIA Jetson.

No video source is registered here. The runtime image carries an OpenCV built
without GStreamer, so this board uses the shared `cv2_ffmpeg` source from
retail_core.video rather than a board-specific one.
"""
from .detector import PERSON_CLASS, TensorRTPersonDetector, decode_person  # noqa: F401


def create_detector(config, stream):
    detector = TensorRTPersonDetector(config["model_path"],
                                      int(config.get("input_size", 640)))
    # Printed once per stream: a plan rebuilt with a different export would
    # change these shapes, and the decode assumes a [1, 84, N] head.
    print(f"[{stream['id']}] tensorrt bindings: {detector.describe()}", flush=True)
    return detector
