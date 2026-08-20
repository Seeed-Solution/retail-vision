"""Rolling-window zone metrics.

Pure analytics logic. Behaviour authority is the reCamera C++ implementation
at solutions/retail-vision/main/zone_metrics.cpp.
"""
from __future__ import annotations

from collections import deque

from .tracker import StateCount  # noqa: F401  (re-exported for callers)


class ZoneMetrics:
    SMOOTHING_WINDOW = 5

    def __init__(self, window_duration=60.0):
        self.window_duration = float(window_duration)
        self._occupancy_samples = deque()
        self._last_sample_time = 0.0
        self._removed = deque()
        self._sum_dwell = 0.0
        self._sum_engagement = 0.0
        self._sum_speed = 0.0
        self._removed_count = 0
        self._history = deque()
        self._counts = StateCount()
        self._smoothed = 0
        self._entry = 0
        self._exit = 0

    def _prune(self, now):
        cutoff = now - self.window_duration
        while self._occupancy_samples and self._occupancy_samples[0][0] < cutoff:
            self._occupancy_samples.popleft()
        while self._removed and self._removed[0].removal_time < cutoff:
            old = self._removed.popleft()
            self._sum_dwell -= old.dwell_time
            self._sum_engagement -= old.engagement_time
            self._sum_speed -= old.avg_speed
            self._removed_count -= 1

    def _smooth(self, raw):
        self._history.append(raw)
        while len(self._history) > self.SMOOTHING_WINDOW:
            self._history.popleft()
        return sorted(self._history)[len(self._history) // 2]

    def update(self, counts, entry_count, exit_count, now):
        self._counts = counts
        self._entry = entry_count
        self._exit = exit_count
        self._smoothed = self._smooth(counts.total)
        if now - self._last_sample_time >= 1.0:
            self._occupancy_samples.append((now, self._smoothed))
            self._last_sample_time = now
        self._prune(now)

    def on_track_removed(self, record):
        self._removed.append(record)
        self._sum_dwell += record.dwell_time
        self._sum_engagement += record.engagement_time
        self._sum_speed += record.avg_speed
        self._removed_count += 1

    def snapshot(self):
        n = self._removed_count
        return {
            "occupancy_count": self._smoothed,
            "browsing_count": self._counts.browsing,
            "engaged_count": self._counts.engaged,
            "assist_count": self._counts.assistance,
            "peak_customer": max((s[1] for s in self._occupancy_samples), default=0),
            "avg_dwell_time": round(self._sum_dwell / n, 1) if n else 0.0,
            "avg_engagement_time": round(self._sum_engagement / n, 1) if n else 0.0,
            "avg_velocity": round(self._sum_speed / n, 2) if n else 0.0,
            "entry_count": self._entry,
            "exit_count": self._exit,
        }
