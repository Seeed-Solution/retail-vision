"""RKNN Lite backend: person detector + Rockchip MPP video source."""
from .detector import RKNNPersonDetector, decode_person  # noqa: F401
from . import video_mpp  # noqa: F401  (registers gstreamer_mpp)


def create_detector(config, stream):
    return RKNNPersonDetector(config["model_path"], stream.get("core_mask"))
