// Generic per-stream tracker (spec BASE-1 §6.2, M1.5).
//
// Matching reproduces retail-vision core-cpp/person_tracker.cpp (the
// authoritative implementation behind
// core-py/retail_core/tests/fixtures/tracker_sequence.json): two-pass
// IoU + centre-distance matching with velocity prediction for coasting
// tracks, older tracks preferred. Dwell / count-zone / entry-line / edge
// logic is intentionally absent; per-track business state lives in
// analyzers keyed by track_id.
//
// track_id semantics (design reference: esk contracts/MQTT.md): ids start
// at 1 and increase monotonically; 0 means "not associated".
#pragma once

#include <vector>

#include "vb/types.h"

namespace vb {

struct TrackerConfig {
    float iou_threshold = 0.2f;
    float dist_threshold = 0.15f;   // max centre distance (normalized) for fallback matching
    int max_misses = 15;            // remove a track when misses > max_misses
    float max_lost_s = 0.0f;        // 0 = do not remove by wall time; >0 also remove when
                                    // t - last_seen_s > max_lost_s
    int min_hits = 1;               // track appears in the returned alive list once hits >= min_hits
    bool class_aware = true;        // different class_id never matches
    enum Anchor { Center, BottomCenter } anchor = Center;
};

struct Track {
    uint32_t track_id = 0;
    Detection det;                  // last *associated* detection (stale while coasting)
    std::vector<Keypoint> kpts;
    float vx = 0, vy = 0;           // normalized units per second (EMA)
    uint32_t hits = 0, misses = 0;
    double first_seen_s = 0, last_seen_s = 0;
};

class Tracker {
public:
    explicit Tracker(TrackerConfig c = TrackerConfig());

    // Writes track_id in place into dets for associated/new detections
    // (every det gets a non-zero id). Returns this frame's alive tracks
    // (including coasting tracks with misses > 0; filtered by min_hits) in
    // ascending-id order. removed collects the ids deleted this frame.
    const std::vector<Track>& update(std::vector<Detection>& dets,
                                     const std::vector<Keypoint>& kpts,
                                     double t_mono_s,
                                     std::vector<uint32_t>& removed);

    void reset();

private:
    void update_velocity(Track& tr, float ncx, float ncy, double dt);

    TrackerConfig cfg_;
    std::vector<Track> tracks_;     // insertion order == ascending id
    std::vector<Track> alive_;
    uint32_t next_id_ = 1;
    double last_t_ = 0;
    bool has_t_ = false;
};

}  // namespace vb
