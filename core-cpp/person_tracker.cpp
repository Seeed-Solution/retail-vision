#include "person_tracker.h"

#include <algorithm>
#include <cmath>

#define TAG "PersonTracker"

#include "core_log.h"

namespace retail_vision {

PersonTracker::PersonTracker() {}

void PersonTracker::setConfig(const TrackerConfig& config) {
    config_ = config;
}

void PersonTracker::setTrackRemovedCallback(TrackRemovedCallback cb) {
    on_track_removed_ = std::move(cb);
}

void PersonTracker::setCountZone(const std::vector<geom::Point>& polygon) {
    if (polygon.size() >= 3) {
        count_zone_ = polygon;
        MA_LOGI(TAG, "Counting zone enabled (%zu points)", count_zone_.size());
    } else {
        count_zone_.clear();
    }
}

void PersonTracker::setEntryLine(const geom::Point& a, const geom::Point& b, bool ab_in) {
    line_a_ = a;
    line_b_ = b;
    line_ab_in_ = ab_in;
    line_enabled_ = true;
    MA_LOGI(TAG, "Entry line enabled (%s)", ab_in ? "ab_in" : "ab_out");
}

// Foot point = bbox bottom-center (normalized). Used for zone membership and
// entry-line crossing so that a person is located by where they stand, not by
// their bbox center.
static inline void foot_point(const DetectionBox& d, float& fx, float& fy) {
    fx = d.x;
    fy = d.y + d.h * 0.5f;
}

bool PersonTracker::isNearEdge(const DetectionBox& det) const {
    float margin = config_.edge_margin;
    float cx = det.x;
    float cy = det.y;
    return (cx < margin || cx > (1.0f - margin) ||
            cy < margin || cy > (1.0f - margin));
}

// UA logic: check if velocity vector points toward nearest edge
bool PersonTracker::isMovingTowardEdge(const TrackedPerson& track) const {
    float cx = track.detection.x;
    float cy = track.detection.y;
    float vx = track.velocity_x;
    float vy = track.velocity_y;
    float fw = static_cast<float>(config_.frame_width);
    float fh = static_cast<float>(config_.frame_height);
    float margin_frac = 0.3f;  // UA checks within 30% for velocity direction

    // Moving left and in left portion
    if (vx < -20.0f && cx < margin_frac) return true;
    // Moving right and in right portion
    if (vx > 20.0f && cx > (1.0f - margin_frac)) return true;
    // Moving up and near top
    if (vy < -40.0f && cy < margin_frac) return true;
    // Moving down and near bottom
    if (vy > 40.0f && cy > (1.0f - margin_frac)) return true;

    (void)fw;
    (void)fh;
    return false;
}

// UA logic: EDGE loss = near edge AND moving away from center
// Near edge BUT moving toward center = CENTER (override, likely occlusion)
bool PersonTracker::classifyAsEdgeLoss(const TrackedPerson& track) const {
    if (!isNearEdge(track.detection)) {
        return false;  // Not near edge = CENTER loss
    }
    // Near edge: check velocity direction
    // If moving toward edge = true EDGE loss (likely exiting)
    // If moving toward center = CENTER override (likely occlusion at edge)
    return isMovingTowardEdge(track);
}

float PersonTracker::computeIoU(const DetectionBox& a, const DetectionBox& b) const {
    float ax1 = a.x - a.w / 2.0f;
    float ay1 = a.y - a.h / 2.0f;
    float ax2 = a.x + a.w / 2.0f;
    float ay2 = a.y + a.h / 2.0f;

    float bx1 = b.x - b.w / 2.0f;
    float by1 = b.y - b.h / 2.0f;
    float bx2 = b.x + b.w / 2.0f;
    float by2 = b.y + b.h / 2.0f;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float a_area = a.w * a.h;
    float b_area = b.w * b.h;
    float union_area = a_area + b_area - inter_area;

    return (union_area > 0) ? inter_area / union_area : 0.0f;
}

std::vector<std::pair<int, int>> PersonTracker::matchDetections(
    const std::vector<DetectionBox>& detections
) const {
    std::vector<std::pair<int, int>> matches;
    if (tracks_.empty() || detections.empty()) {
        return matches;
    }

    std::vector<int> track_ids;
    for (const auto& pair : tracks_) {
        track_ids.push_back(pair.first);
    }

    // Prefer older tracks
    std::sort(track_ids.begin(), track_ids.end(),
        [this](int a, int b) {
            return tracks_.at(a).frames_tracked > tracks_.at(b).frames_tracked;
        });

    std::vector<bool> det_used(detections.size(), false);

    // Pass 1: IOU matching with velocity-predicted positions
    std::vector<int> unmatched_tracks;
    for (int track_id : track_ids) {
        const auto& track = tracks_.at(track_id);

        // Predict position using velocity for lost tracks
        DetectionBox pred = track.detection;
        if (track.lost_frames > 0 && track.speed_px_s > 1.0f) {
            float dt = track.lost_frames / 15.0f;  // ~15fps
            pred.x += (track.velocity_x / config_.frame_width) * dt;
            pred.y += (track.velocity_y / config_.frame_height) * dt;
        }

        float best_iou = config_.iou_threshold;
        int best_det_idx = -1;

        for (size_t d = 0; d < detections.size(); d++) {
            if (det_used[d]) continue;
            float iou = computeIoU(pred, detections[d]);
            if (iou > best_iou) {
                best_iou = iou;
                best_det_idx = static_cast<int>(d);
            }
        }

        if (best_det_idx >= 0) {
            matches.push_back({track_id, best_det_idx});
            det_used[best_det_idx] = true;
        } else {
            unmatched_tracks.push_back(track_id);
        }
    }

    // Pass 2: Center-distance fallback for recently lost tracks
    for (int track_id : unmatched_tracks) {
        const auto& track = tracks_.at(track_id);
        if (track.lost_frames > 5) continue;

        DetectionBox pred = track.detection;
        if (track.lost_frames > 0 && track.speed_px_s > 1.0f) {
            float dt = track.lost_frames / 15.0f;
            pred.x += (track.velocity_x / config_.frame_width) * dt;
            pred.y += (track.velocity_y / config_.frame_height) * dt;
        }

        float best_dist = config_.dist_threshold;
        int best_det_idx = -1;

        for (size_t d = 0; d < detections.size(); d++) {
            if (det_used[d]) continue;
            float dx = detections[d].x - pred.x;
            float dy = detections[d].y - pred.y;
            float dist = std::hypot(dx, dy);
            if (dist < best_dist) {
                best_dist = dist;
                best_det_idx = static_cast<int>(d);
            }
        }

        if (best_det_idx >= 0) {
            matches.push_back({track_id, best_det_idx});
            det_used[best_det_idx] = true;
        }
    }

    return matches;
}

void PersonTracker::updateVelocity(TrackedPerson& track,
                                    const DetectionBox& new_det,
                                    float dt) {
    if (dt <= 0.001f) return;

    float dx = (new_det.x - track.detection.x) * config_.frame_width;
    float dy = (new_det.y - track.detection.y) * config_.frame_height;
    float ivx = dx / dt;
    float ivy = dy / dt;
    float instant_speed = std::hypot(ivx, ivy);

    float prev_speed = track.speed_px_s;

    // UA: detect sudden stop/start for adaptive alpha
    bool sudden_stop = (prev_speed > 10.0f) &&
                       ((prev_speed - instant_speed) / prev_speed > 0.5f);
    bool sudden_start = (prev_speed < 5.0f) && (instant_speed > 50.0f);

    if (instant_speed < config_.velocity_zero_threshold) {
        track.velocity_x = 0.0f;
        track.velocity_y = 0.0f;
        track.speed_px_s = 0.0f;
    } else {
        float alpha = (sudden_stop || sudden_start) ? config_.vel_alpha_sudden : config_.vel_alpha;
        track.velocity_x = (1.0f - alpha) * track.velocity_x + alpha * ivx;
        track.velocity_y = (1.0f - alpha) * track.velocity_y + alpha * ivy;
        track.speed_px_s = std::hypot(track.velocity_x, track.velocity_y);
    }

    // Convert to m/s: speed_m_s = (speed_px_s / bbox_height_px) * avg_person_height_m
    float bbox_h_px = new_det.h * config_.frame_height;
    if (bbox_h_px > 1.0f) {
        track.speed_m_s = (track.speed_px_s / bbox_h_px) * config_.avg_person_height_m;
    } else {
        track.speed_m_s = 0.0f;
    }

    // Accumulate for average
    track.speed_sum += track.speed_m_s;
    track.speed_samples++;
}

// UA logic: stationary frames with decay to tolerate brief gestures
void PersonTracker::updateStationaryFrames(TrackedPerson& track, bool is_stationary) {
    if (is_stationary) {
        track.stationary_frames++;
    } else {
        // Gradual decay instead of hard reset
        if (track.stationary_frames > config_.stationary_stable_threshold) {
            // Was stable for a while - slow decay
            track.stationary_frames = std::max(0, track.stationary_frames - config_.stationary_decay_slow);
        } else {
            // Not yet stable - fast decay
            track.stationary_frames = std::max(0, track.stationary_frames - config_.stationary_decay_fast);
        }
    }
}

void PersonTracker::updateDwellState(TrackedPerson& track, float current_time) {
    bool is_stationary = track.speed_px_s < config_.dwell_speed_threshold;

    // UA: update stationary frames with decay
    updateStationaryFrames(track, is_stationary);

    bool is_still = is_stationary && (track.stationary_frames >= config_.dwell_min_frames);

    if (!is_still) {
        // Not confirmed stationary
        if (track.stationary_frames == 0) {
            // Fully moving - reset dwell
            track.dwell_state = DwellState::TRANSIENT;
            track.dwell_start_time = 0.0f;
            track.dwell_duration_sec = 0.0f;
        }
        // If stationary_frames > 0 but < min_frames, keep current state (decay grace)
        return;
    }

    // Confirmed stationary
    if (track.dwell_start_time <= 0.0f) {
        track.dwell_start_time = current_time;
    }

    track.dwell_duration_sec = current_time - track.dwell_start_time;

    DwellState prev_state = track.dwell_state;

    if (track.dwell_duration_sec >= config_.dwell_assistance_threshold) {
        track.dwell_state = DwellState::ASSISTANCE;
    } else if (track.dwell_duration_sec >= config_.dwell_min_duration) {
        track.dwell_state = DwellState::ENGAGED;
    } else {
        track.dwell_state = DwellState::DWELLING;
    }

    // Track when engagement starts
    if (prev_state < DwellState::ENGAGED && track.dwell_state >= DwellState::ENGAGED) {
        track.engagement_start_time = current_time;
    }
}

void PersonTracker::removeTrack(int track_id, float current_time) {
    auto it = tracks_.find(track_id);
    if (it == tracks_.end()) return;

    const auto& track = it->second;

    // UA: exit counted at removal. Use velocity-aware edge classification.
    // Skipped when an entry line is configured — crossings drive the counts.
    if (!line_enabled_ && track.frames_tracked >= config_.min_frames_for_count) {
        // Classify loss zone using velocity direction (UA logic)
        bool is_edge_loss = classifyAsEdgeLoss(track);

        if (is_edge_loss) {
            // Near edge and moving toward edge → real exit
            exit_count_++;
        } else if (isNearEdge(track.detection)) {
            // Near edge but NOT moving toward edge → likely exit anyway (timeout)
            exit_count_++;
        }
        // Center loss → no exit count (occlusion)
    }

    // Emit TrackRecord for ZoneMetrics
    if (on_track_removed_) {
        TrackRecord record;
        record.track_id = track.track_id;
        record.dwell_time = track.dwell_duration_sec;
        record.engagement_time = (track.engagement_start_time > 0.0f)
            ? (current_time - track.engagement_start_time) : 0.0f;
        record.avg_speed = (track.speed_samples > 0)
            ? (track.speed_sum / track.speed_samples) : 0.0f;
        record.removal_time = current_time;
        record.exited_at_edge = isNearEdge(track.detection);

        on_track_removed_(record);
    }

    tracks_.erase(it);
}

std::vector<TrackedPerson> PersonTracker::update(
    const std::vector<DetectionBox>& detections,
    float current_time_sec
) {
    float dt = (last_update_time_ > 0) ? (current_time_sec - last_update_time_) : 0.0f;
    last_update_time_ = current_time_sec;

    // Filter person detections (target == 0 in COCO)
    std::vector<DetectionBox> person_detections;
    for (const auto& det : detections) {
        if (det.target == 0) {
            person_detections.push_back(det);
        }
    }

    auto matches = matchDetections(person_detections);

    std::vector<bool> det_matched(person_detections.size(), false);
    std::vector<int> matched_track_ids;

    for (const auto& match : matches) {
        int track_id = match.first;
        int det_idx = match.second;
        det_matched[det_idx] = true;
        matched_track_ids.push_back(track_id);

        TrackedPerson& track = tracks_[track_id];

        updateVelocity(track, person_detections[det_idx], dt);

        // Entry-line crossing: compare the track's previous foot point with the
        // new one. A sign change of the cross-product against the directed line
        // a->b is a genuine crossing; direction decides entry vs exit. Only
        // active when an entry line is configured (otherwise the default
        // appearance/disappearance counting below stays in effect).
        if (line_enabled_) {
            float p0x, p0y, p1x, p1y;
            foot_point(track.detection, p0x, p0y);
            foot_point(person_detections[det_idx], p1x, p1y);
            int dir = geom::segment_crossing(line_a_[0], line_a_[1], line_b_[0], line_b_[1],
                                             p0x, p0y, p1x, p1y);
            if (dir != 0) {
                bool is_in = (dir > 0) == line_ab_in_;
                if (is_in) {
                    entry_count_++;
                } else {
                    exit_count_++;
                }
                MA_LOGI(TAG, "Track %d crossed entry line: %s (entry=%d exit=%d)",
                        track_id, is_in ? "in" : "out", entry_count_, exit_count_);
            }
        }

        track.detection = person_detections[det_idx];
        track.last_seen_time = current_time_sec;
        track.frames_tracked++;
        track.lost_frames = 0;
        track.last_near_edge = isNearEdge(person_detections[det_idx]);

        updateDwellState(track, current_time_sec);
    }

    // Handle unmatched tracks
    std::vector<int> tracks_to_remove;
    for (auto& pair : tracks_) {
        int track_id = pair.first;
        if (std::find(matched_track_ids.begin(), matched_track_ids.end(), track_id)
            == matched_track_ids.end()) {
            pair.second.lost_frames++;

            // Edge-aware max lost frames (UA: buffer_edge_sec=0.5, buffer_center_sec=3.0)
            bool edge_loss = classifyAsEdgeLoss(pair.second);
            int max_lost = edge_loss
                ? config_.max_lost_frames_edge
                : config_.max_lost_frames_center;

            if (pair.second.lost_frames > max_lost) {
                tracks_to_remove.push_back(track_id);
            }
        }
    }

    for (int track_id : tracks_to_remove) {
        removeTrack(track_id, current_time_sec);
    }

    // Create new tracks for unmatched detections
    // UA: entry_count++ when track is first created
    for (size_t i = 0; i < person_detections.size(); i++) {
        if (!det_matched[i]) {
            TrackedPerson new_track;
            new_track.track_id = next_track_id_++;
            new_track.detection = person_detections[i];
            new_track.first_seen_time = current_time_sec;
            new_track.last_seen_time = current_time_sec;
            new_track.frames_tracked = 1;
            new_track.lost_frames = 0;
            new_track.dwell_state = DwellState::TRANSIENT;
            new_track.last_near_edge = isNearEdge(person_detections[i]);

            tracks_[new_track.track_id] = new_track;

            // UA: count entry when track is created. Skipped when an entry line
            // is configured — crossings drive the counts instead (see above).
            if (!line_enabled_) {
                entry_count_++;
            }
        }
    }

    // Build result (only visible tracks)
    std::vector<TrackedPerson> result;
    for (const auto& pair : tracks_) {
        if (pair.second.lost_frames == 0) {
            result.push_back(pair.second);
        }
    }

    return result;
}

StateCount PersonTracker::getStateCounts() const {
    StateCount counts;

    for (const auto& pair : tracks_) {
        const auto& track = pair.second;
        if (track.lost_frames > 0) continue;

        // With a counting zone configured, only persons whose FOOT point (bbox
        // bottom-center) is inside the polygon contribute to zone occupancy /
        // dwell counts. Empty zone = whole frame (default, unchanged).
        if (!count_zone_.empty()) {
            float fx, fy;
            foot_point(track.detection, fx, fy);
            if (!geom::point_in_polygon(fx, fy, count_zone_)) {
                continue;
            }
        }

        counts.total++;
        switch (track.dwell_state) {
            case DwellState::TRANSIENT:
            case DwellState::DWELLING:
                counts.browsing++;
                break;
            case DwellState::ENGAGED:
                counts.engaged++;
                break;
            case DwellState::ASSISTANCE:
                counts.assistance++;
                break;
        }
    }

    return counts;
}

}  // namespace retail_vision
