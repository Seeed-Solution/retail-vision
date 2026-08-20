#ifndef _PERSON_TRACKER_H_
#define _PERSON_TRACKER_H_

#include <vector>
#include <map>
#include <functional>
#include <cmath>

#include "detection_box.h"
#include "geometry.h"

namespace retail_vision {

enum class DwellState {
    TRANSIENT,
    DWELLING,
    ENGAGED,
    ASSISTANCE
};

inline const char* getDwellStateName(DwellState state) {
    switch (state) {
        case DwellState::TRANSIENT:  return "transient";
        case DwellState::DWELLING:   return "dwelling";
        case DwellState::ENGAGED:    return "engaged";
        case DwellState::ASSISTANCE: return "assistance";
        default: return "unknown";
    }
}

// Record emitted when a track is removed (for ZoneMetrics)
struct TrackRecord {
    int track_id;
    float dwell_time;           // Total time stationary (seconds)
    float engagement_time;      // Time in ENGAGED+ state (seconds)
    float avg_speed;            // Average speed_m_s over lifetime
    float removal_time;         // When removed (seconds since start)
    bool exited_at_edge;        // True if last seen near frame edge
};

struct TrackedPerson {
    int track_id;
    DetectionBox detection;

    // Velocity
    float velocity_x = 0.0f;       // Normalized velocity x per second
    float velocity_y = 0.0f;       // Normalized velocity y per second
    float speed_px_s = 0.0f;       // Speed in pixels/sec (640x640 frame)
    float speed_m_s = 0.0f;        // Estimated speed in meters/sec

    // Dwell state
    DwellState dwell_state = DwellState::TRANSIENT;
    float dwell_duration_sec = 0.0f;

    // Internal tracking data
    float first_seen_time = 0.0f;
    float last_seen_time = 0.0f;
    float dwell_start_time = 0.0f;
    float engagement_start_time = 0.0f;
    int frames_tracked = 0;
    int stationary_frames = 0;
    int lost_frames = 0;

    // Velocity accumulator for avg_speed
    float speed_sum = 0.0f;
    int speed_samples = 0;

    // Edge detection
    bool last_near_edge = false;
};

struct TrackerConfig {
    float iou_threshold = 0.2f;
    float dist_threshold = 0.15f;       // Max center-distance (normalized) for fallback matching
    int max_lost_frames_center = 90;    // ~3s at 30fps for center tracks
    int max_lost_frames_edge = 15;      // ~0.5s for edge tracks

    // Velocity EMA
    float vel_alpha = 0.08f;
    float vel_alpha_sudden = 0.6f;
    float velocity_zero_threshold = 3.0f;

    // Dwell detection
    float dwell_speed_threshold = 10.0f;
    float dwell_min_duration = 1.5f;
    float dwell_assistance_threshold = 20.0f;
    int dwell_min_frames = 5;

    // Frame dimensions
    int frame_width = 640;
    int frame_height = 640;

    // Edge margin (fraction of frame) - UA uses 0.15
    float edge_margin = 0.15f;

    // Min frames before counting entry/exit
    int min_frames_for_count = 10;

    // Stationary frame decay (UA tolerates brief gestures)
    int stationary_stable_threshold = 30;   // Frames before considered stable
    int stationary_decay_slow = 2;          // Decay rate when stable
    int stationary_decay_fast = 5;          // Decay rate when not stable

    // Person height for m/s estimation
    float avg_person_height_m = 1.7f;
};

// Zone occupancy counts
struct StateCount {
    int total = 0;
    int browsing = 0;
    int engaged = 0;
    int assistance = 0;
};

using TrackRemovedCallback = std::function<void(const TrackRecord&)>;

class PersonTracker {
public:
    PersonTracker();
    ~PersonTracker() = default;

    void setConfig(const TrackerConfig& config);
    void setTrackRemovedCallback(TrackRemovedCallback cb);

    // Optional spatial config. Both are OFF unless the operator draws them.
    //
    // Counting zone: when set, occupancy / dwell state counts (getStateCounts)
    // only include persons whose FOOT point (bbox bottom-center) is inside the
    // normalized polygon. Empty polygon (or <3 points) = whole-frame (default).
    void setCountZone(const std::vector<geom::Point>& polygon);
    bool hasCountZone() const { return !count_zone_.empty(); }

    // Entry line: when set, entry_count_/exit_count_ are driven by directed
    // crossings of the track's foot point over the segment a->b (instead of the
    // default appearance/disappearance counting). ab_in == true: a crossing
    // from the LEFT of a->b to the RIGHT counts as an entry.
    void setEntryLine(const geom::Point& a, const geom::Point& b, bool ab_in);
    bool hasEntryLine() const { return line_enabled_; }

    std::vector<TrackedPerson> update(
        const std::vector<DetectionBox>& detections,
        float current_time_sec
    );

    StateCount getStateCounts() const;
    int getTrackCount() const { return static_cast<int>(tracks_.size()); }
    int getEntryCount() const { return entry_count_; }
    int getExitCount() const { return exit_count_; }

private:
    float computeIoU(const DetectionBox& a, const DetectionBox& b) const;

    std::vector<std::pair<int, int>> matchDetections(
        const std::vector<DetectionBox>& detections
    ) const;

    void updateVelocity(TrackedPerson& track, const DetectionBox& new_det, float dt);
    void updateDwellState(TrackedPerson& track, float current_time);
    void updateStationaryFrames(TrackedPerson& track, bool is_stationary);
    bool isNearEdge(const DetectionBox& det) const;
    bool isMovingTowardEdge(const TrackedPerson& track) const;
    bool classifyAsEdgeLoss(const TrackedPerson& track) const;
    void removeTrack(int track_id, float current_time);

private:
    TrackerConfig config_;
    std::map<int, TrackedPerson> tracks_;
    int next_track_id_ = 0;
    float last_update_time_ = 0.0f;
    int entry_count_ = 0;
    int exit_count_ = 0;
    TrackRemovedCallback on_track_removed_;

    // Optional counting zone (empty = whole frame). Uses foot points.
    std::vector<geom::Point> count_zone_;

    // Optional entry line for directed crossing counts (foot points).
    bool line_enabled_ = false;
    geom::Point line_a_ { 0.0f, 0.0f };
    geom::Point line_b_ { 0.0f, 0.0f };
    bool line_ab_in_ = true;
};

}  // namespace retail_vision

#endif  // _PERSON_TRACKER_H_
