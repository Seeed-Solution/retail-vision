"""Person tracker: two-pass matching, velocity EMA, edge-aware expiry.

Pure analytics logic: it consumes normalized detection boxes plus a timestamp
and nothing else, so every backend (RKNN, TensorRT, Hailo) shares it.

Behaviour authority is the reCamera C++ implementation at
solutions/retail-vision/main/person_tracker.cpp. Coordinates are model-space
normalized centre boxes in [0,1]; speeds are expressed in model-input pixels
per second, exactly as on reCamera, so the dwell thresholds carry over
unchanged regardless of sensor resolution.
"""
from __future__ import annotations

import math

from .dwell import (ASSISTANCE, DWELLING, ENGAGED, TRANSIENT, DwellThresholds,
                    update_dwell_state)
from .geometry import point_in_polygon, segment_crossing


class DetectionBox:
    """Centre-normalized detection in model input space."""
    __slots__ = ("x", "y", "w", "h", "score")

    def __init__(self, x, y, w, h, score):
        self.x, self.y, self.w, self.h, self.score = x, y, w, h, score

    def copy(self):
        return DetectionBox(self.x, self.y, self.w, self.h, self.score)

    def __repr__(self):
        return f"DetectionBox({self.x:.3f},{self.y:.3f},{self.w:.3f},{self.h:.3f},{self.score:.2f})"


class TrackerConfig(DwellThresholds):
    def __init__(self, **kw):
        super().__init__()
        self.iou_threshold = 0.2
        self.dist_threshold = 0.15
        self.max_lost_frames_center = 90     # ~3 s at 30 fps
        self.max_lost_frames_edge = 15       # ~0.5 s
        self.vel_alpha = 0.08
        self.vel_alpha_sudden = 0.6
        self.velocity_zero_threshold = 3.0
        self.frame_width = 640
        self.frame_height = 640
        self.edge_margin = 0.15
        self.min_frames_for_count = 10
        self.avg_person_height_m = 1.7
        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError(f"unknown tracker option: {k}")
            setattr(self, k, type(getattr(self, k))(v))


class TrackedPerson:
    __slots__ = ("track_id", "detection", "velocity_x", "velocity_y", "speed_px_s",
                 "speed_m_s", "dwell_state", "dwell_duration_sec", "first_seen_time",
                 "last_seen_time", "dwell_start_time", "engagement_start_time",
                 "frames_tracked", "stationary_frames", "lost_frames", "speed_sum",
                 "speed_samples", "last_near_edge")

    def __init__(self, track_id, detection, now):
        self.track_id = track_id
        self.detection = detection
        self.velocity_x = self.velocity_y = 0.0
        self.speed_px_s = self.speed_m_s = 0.0
        self.dwell_state = TRANSIENT
        self.dwell_duration_sec = 0.0
        self.first_seen_time = self.last_seen_time = now
        self.dwell_start_time = 0.0
        self.engagement_start_time = 0.0
        self.frames_tracked = 1
        self.stationary_frames = 0
        self.lost_frames = 0
        self.speed_sum = 0.0
        self.speed_samples = 0
        self.last_near_edge = False


class TrackRecord:
    __slots__ = ("track_id", "dwell_time", "engagement_time", "avg_speed",
                 "removal_time", "exited_at_edge")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


class StateCount:
    __slots__ = ("total", "browsing", "engaged", "assistance")

    def __init__(self):
        self.total = self.browsing = self.engaged = self.assistance = 0


def foot_point(d):
    """Bbox bottom-centre: a person is located by where they stand."""
    return d.x, d.y + d.h * 0.5


class PersonTracker:
    def __init__(self, config=None, on_track_removed=None):
        self.config = config or TrackerConfig()
        self.tracks = {}
        self._next_id = 0
        self._last_update = 0.0
        self.entry_count = 0
        self.exit_count = 0
        self.on_track_removed = on_track_removed
        self.count_zone = []
        self.line_enabled = False
        self.line_a = (0.0, 0.0)
        self.line_b = (0.0, 0.0)
        self.line_ab_in = True

    # Both spatial features are off unless the operator draws them: with no
    # zone the whole frame counts, with no line entries/exits come from track
    # appearance and disappearance.
    def set_count_zone(self, polygon):
        self.count_zone = list(polygon) if polygon and len(polygon) >= 3 else []

    def set_entry_line(self, a, b, ab_in=True):
        self.line_a, self.line_b, self.line_ab_in = tuple(a), tuple(b), bool(ab_in)
        self.line_enabled = True

    # -- geometry ---------------------------------------------------------
    def _near_edge(self, det):
        m = self.config.edge_margin
        return det.x < m or det.x > 1.0 - m or det.y < m or det.y > 1.0 - m

    def _moving_toward_edge(self, tr):
        cx, cy, vx, vy = tr.detection.x, tr.detection.y, tr.velocity_x, tr.velocity_y
        f = 0.3
        return ((vx < -20.0 and cx < f) or (vx > 20.0 and cx > 1.0 - f)
                or (vy < -40.0 and cy < f) or (vy > 40.0 and cy > 1.0 - f))

    def _edge_loss(self, tr):
        # Near the edge but moving toward the centre is an occlusion, not an
        # exit, so it keeps the long centre timeout.
        return self._near_edge(tr.detection) and self._moving_toward_edge(tr)

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a.x - a.w / 2, a.y - a.h / 2, a.x + a.w / 2, a.y + a.h / 2
        bx1, by1, bx2, by2 = b.x - b.w / 2, b.y - b.h / 2, b.x + b.w / 2, b.y + b.h / 2
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = a.w * a.h + b.w * b.h - inter
        return inter / union if union > 0 else 0.0

    def _predict(self, tr):
        pred = tr.detection.copy()
        if tr.lost_frames > 0 and tr.speed_px_s > 1.0:
            dt = tr.lost_frames / 15.0
            pred.x += (tr.velocity_x / self.config.frame_width) * dt
            pred.y += (tr.velocity_y / self.config.frame_height) * dt
        return pred

    def _match(self, detections):
        matches = []
        if not self.tracks or not detections:
            return matches
        track_ids = sorted(self.tracks, key=lambda t: -self.tracks[t].frames_tracked)
        used = [False] * len(detections)
        unmatched = []
        # Pass 1: IoU against velocity-predicted positions, older tracks first.
        for tid in track_ids:
            pred = self._predict(self.tracks[tid])
            best, best_idx = self.config.iou_threshold, -1
            for d, det in enumerate(detections):
                if used[d]:
                    continue
                v = self.iou(pred, det)
                if v > best:
                    best, best_idx = v, d
            if best_idx >= 0:
                matches.append((tid, best_idx))
                used[best_idx] = True
            else:
                unmatched.append(tid)
        # Pass 2: centre-distance fallback, recently lost tracks only.
        for tid in unmatched:
            tr = self.tracks[tid]
            if tr.lost_frames > 5:
                continue
            pred = self._predict(tr)
            best, best_idx = self.config.dist_threshold, -1
            for d, det in enumerate(detections):
                if used[d]:
                    continue
                dist = math.hypot(det.x - pred.x, det.y - pred.y)
                if dist < best:
                    best, best_idx = dist, d
            if best_idx >= 0:
                matches.append((tid, best_idx))
                used[best_idx] = True
        return matches

    def _update_velocity(self, tr, new_det, dt):
        if dt <= 0.001:
            return
        c = self.config
        ivx = (new_det.x - tr.detection.x) * c.frame_width / dt
        ivy = (new_det.y - tr.detection.y) * c.frame_height / dt
        instant = math.hypot(ivx, ivy)
        prev = tr.speed_px_s
        sudden_stop = prev > 10.0 and (prev - instant) / prev > 0.5
        sudden_start = prev < 5.0 and instant > 50.0
        if instant < c.velocity_zero_threshold:
            tr.velocity_x = tr.velocity_y = tr.speed_px_s = 0.0
        else:
            a = c.vel_alpha_sudden if (sudden_stop or sudden_start) else c.vel_alpha
            tr.velocity_x = (1 - a) * tr.velocity_x + a * ivx
            tr.velocity_y = (1 - a) * tr.velocity_y + a * ivy
            tr.speed_px_s = math.hypot(tr.velocity_x, tr.velocity_y)
        # px/s -> m/s using bbox height as the scale reference.
        bbox_h_px = new_det.h * c.frame_height
        tr.speed_m_s = (tr.speed_px_s / bbox_h_px) * c.avg_person_height_m if bbox_h_px > 1.0 else 0.0
        tr.speed_sum += tr.speed_m_s
        tr.speed_samples += 1

    def _remove(self, tid, now):
        tr = self.tracks.pop(tid, None)
        if tr is None:
            return
        # Exit is counted at removal. A loss in the centre of the frame is an
        # occlusion, not a departure, so it does not count. Skipped entirely
        # when an entry line drives the counts.
        if not self.line_enabled and tr.frames_tracked >= self.config.min_frames_for_count:
            if self._near_edge(tr.detection):
                self.exit_count += 1
        if self.on_track_removed:
            self.on_track_removed(TrackRecord(
                track_id=tr.track_id,
                dwell_time=tr.dwell_duration_sec,
                engagement_time=(now - tr.engagement_start_time) if tr.engagement_start_time > 0 else 0.0,
                avg_speed=(tr.speed_sum / tr.speed_samples) if tr.speed_samples else 0.0,
                removal_time=now,
                exited_at_edge=self._near_edge(tr.detection)))

    def update(self, detections, now):
        dt = (now - self._last_update) if self._last_update > 0 else 0.0
        self._last_update = now
        matches = self._match(detections)
        matched_ids = set()
        det_matched = [False] * len(detections)
        for tid, di in matches:
            det_matched[di] = True
            matched_ids.add(tid)
            tr = self.tracks[tid]
            new_det = detections[di]
            self._update_velocity(tr, new_det, dt)
            if self.line_enabled:
                p0x, p0y = foot_point(tr.detection)
                p1x, p1y = foot_point(new_det)
                d = segment_crossing(self.line_a[0], self.line_a[1],
                                     self.line_b[0], self.line_b[1],
                                     p0x, p0y, p1x, p1y)
                if d != 0:
                    if (d > 0) == self.line_ab_in:
                        self.entry_count += 1
                    else:
                        self.exit_count += 1
            tr.detection = new_det
            tr.last_seen_time = now
            tr.frames_tracked += 1
            tr.lost_frames = 0
            tr.last_near_edge = self._near_edge(new_det)
            update_dwell_state(tr, now, self.config)

        to_remove = []
        for tid, tr in self.tracks.items():
            if tid in matched_ids:
                continue
            tr.lost_frames += 1
            max_lost = (self.config.max_lost_frames_edge if self._edge_loss(tr)
                        else self.config.max_lost_frames_center)
            if tr.lost_frames > max_lost:
                to_remove.append(tid)
        for tid in to_remove:
            self._remove(tid, now)

        for i, det in enumerate(detections):
            if det_matched[i]:
                continue
            tr = TrackedPerson(self._next_id, det, now)
            self._next_id += 1
            tr.last_near_edge = self._near_edge(det)
            self.tracks[tr.track_id] = tr
            if not self.line_enabled:
                self.entry_count += 1

        return [t for t in self.tracks.values() if t.lost_frames == 0]

    def state_counts(self):
        counts = StateCount()
        for tr in self.tracks.values():
            if tr.lost_frames > 0:
                continue
            if self.count_zone:
                fx, fy = foot_point(tr.detection)
                if not point_in_polygon(fx, fy, self.count_zone):
                    continue
            counts.total += 1
            if tr.dwell_state in (TRANSIENT, DWELLING):
                counts.browsing += 1
            elif tr.dwell_state == ENGAGED:
                counts.engaged += 1
            else:
                counts.assistance += 1
        return counts
