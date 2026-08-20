"""Shared retail people-flow analytics core.

Nothing in this package imports an inference SDK or a board library. A backend
supplies `detect(frame) -> [DetectionBox]` and the rest -- tracking, dwell
state machine, rolling zone metrics, payload construction and batched MQTT
publishing -- is identical on RKNN, TensorRT and any future runtime.
"""
from .dwell import (ASSISTANCE, DWELLING, ENGAGED, STATE_NAMES, TRANSIENT,
                    DwellThresholds)
from .payload import build_vision_payload, letterbox_correction
from .publisher import MqttPublisher, PublishCycle, results_topic, status_topic
from .tracker import (DetectionBox, PersonTracker, StateCount, TrackedPerson,
                      TrackerConfig, TrackRecord)
from .zone_metrics import ZoneMetrics

__all__ = ["ASSISTANCE", "DWELLING", "ENGAGED", "TRANSIENT", "STATE_NAMES",
           "DwellThresholds", "DetectionBox", "PersonTracker", "StateCount",
           "TrackedPerson", "TrackerConfig", "TrackRecord", "ZoneMetrics",
           "build_vision_payload", "letterbox_correction", "MqttPublisher",
           "PublishCycle", "results_topic", "status_topic"]
