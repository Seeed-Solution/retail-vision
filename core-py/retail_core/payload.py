"""VisionPayload construction.

Pure formatting logic shared by every backend; see contracts/MQTT.md.
Behaviour authority is solutions/retail-vision/main/mqtt_payload.cpp.
"""
from __future__ import annotations

from .dwell import STATE_NAMES


def letterbox_correction(frame_width, frame_height, model_width, model_height):
    """Scale/offset that maps model-normalized coords back to display-normalized."""
    display_aspect = frame_width / frame_height
    model_aspect = model_width / model_height
    scale_x = scale_y = 1.0
    offset_x = offset_y = 0.0
    if display_aspect > model_aspect:
        scale_y = model_aspect / display_aspect
        offset_y = (1.0 - scale_y) / 2.0
    elif display_aspect < model_aspect:
        scale_x = display_aspect / model_aspect
        offset_x = (1.0 - scale_x) / 2.0
    return scale_x, scale_y, offset_x, offset_y


def build_vision_payload(timestamp_ms, frame_id, fps, inference_time_ms, zone,
                         persons, frame_width, frame_height,
                         model_width=640, model_height=640):
    """One batched VisionPayload per publish cycle (see contracts/MQTT.md)."""
    sx, sy, ox, oy = letterbox_correction(frame_width, frame_height,
                                          model_width, model_height)
    out = []
    for slot, p in enumerate(persons):
        real_cx = (p.detection.x - ox) / sx
        real_cy = (p.detection.y - oy) / sy
        real_w = p.detection.w / sx
        real_h = p.detection.h / sy
        out.append({
            # Index within this batch. Every person in one message shares a
            # timestamp, so a time-series consumer needs a distinguishing key;
            # the index keeps that key bounded by people-per-frame instead of
            # growing for the life of the deployment.
            "slot": slot,
            "track_id": p.track_id,
            "state": STATE_NAMES[p.dwell_state],
            "cx_pct": round(real_cx * 100.0, 1),
            "cy_pct": round(real_cy * 100.0, 1),
            "dwell_duration": round(p.dwell_duration_sec, 1),
            "confidence": round(p.detection.score, 2),
            "bbox": {"x": round(real_cx - real_w / 2, 4),
                     "y": round(real_cy - real_h / 2, 4),
                     "w": round(real_w, 4),
                     "h": round(real_h, 4)},
            "velocity": {"vx": round(p.velocity_x / frame_width, 2),
                         "vy": round(p.velocity_y / frame_height, 2),
                         "speed_m_s": round(p.speed_m_s, 2)},
        })
    return {
        "timestamp": int(timestamp_ms),
        "frame_id": int(frame_id),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "fps": round(fps, 1),
        "inference_time_ms": round(inference_time_ms, 1),
        "zone": zone,
        "persons": out,
    }
