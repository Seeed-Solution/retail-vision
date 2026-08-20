"""Dwell state vocabulary and thresholds.

Pure analytics logic: no inference SDK, no board library, no I/O.
Behaviour authority is the reCamera C++ implementation at
solutions/retail-vision/main/person_tracker.cpp
"""
from __future__ import annotations


TRANSIENT, DWELLING, ENGAGED, ASSISTANCE = 0, 1, 2, 3

STATE_NAMES = {TRANSIENT: "transient", DWELLING: "dwelling",
               ENGAGED: "engaged", ASSISTANCE: "assistance"}


class DwellThresholds:
    """Defaults match the C++ TrackerConfig and the MQTT contract."""

    def __init__(self, dwell_speed_threshold=10.0, dwell_min_duration=1.5,
                 dwell_assistance_threshold=20.0, dwell_min_frames=5,
                 stationary_stable_threshold=30, stationary_decay_slow=2,
                 stationary_decay_fast=5):
        self.dwell_speed_threshold = float(dwell_speed_threshold)
        self.dwell_min_duration = float(dwell_min_duration)
        self.dwell_assistance_threshold = float(dwell_assistance_threshold)
        self.dwell_min_frames = int(dwell_min_frames)
        self.stationary_stable_threshold = int(stationary_stable_threshold)
        self.stationary_decay_slow = int(stationary_decay_slow)
        self.stationary_decay_fast = int(stationary_decay_fast)


def update_stationary_frames(track, is_stationary, thresholds):
    """Decay rather than hard-reset, so a brief gesture does not drop the dwell."""
    if is_stationary:
        track.stationary_frames += 1
    elif track.stationary_frames > thresholds.stationary_stable_threshold:
        track.stationary_frames = max(0, track.stationary_frames - thresholds.stationary_decay_slow)
    else:
        track.stationary_frames = max(0, track.stationary_frames - thresholds.stationary_decay_fast)


def update_dwell_state(track, now, thresholds):
    is_stationary = track.speed_px_s < thresholds.dwell_speed_threshold
    update_stationary_frames(track, is_stationary, thresholds)
    if not (is_stationary and track.stationary_frames >= thresholds.dwell_min_frames):
        if track.stationary_frames == 0:
            track.dwell_state = TRANSIENT
            track.dwell_start_time = 0.0
            track.dwell_duration_sec = 0.0
        # stationary_frames between 1 and min_frames keeps the current state:
        # that is the decay grace period, not a state change.
        return
    if track.dwell_start_time <= 0.0:
        track.dwell_start_time = now
    track.dwell_duration_sec = now - track.dwell_start_time
    previous = track.dwell_state
    if track.dwell_duration_sec >= thresholds.dwell_assistance_threshold:
        track.dwell_state = ASSISTANCE
    elif track.dwell_duration_sec >= thresholds.dwell_min_duration:
        track.dwell_state = ENGAGED
    else:
        track.dwell_state = DWELLING
    if previous < ENGAGED <= track.dwell_state:
        track.engagement_start_time = now
