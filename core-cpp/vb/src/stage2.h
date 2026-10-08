#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "vb/backend.h"

namespace vb {

struct Stage2FilterConfig {
    float min_score = 0.0f;
    int max_per_track = 0;       // 0 = unlimited
    double min_gap_s = 0.0;
    int max_crops_per_frame = 0; // 0 = unlimited
    float expand = 0.0f;
    float roi[4] = {0, 0, 1, 1};
    std::vector<float> roi_polygon; // x,y pairs in source-normalized coordinates
    std::vector<int32_t> classes;
};

struct Stage2TrackState {
    uint64_t attempts = 0;
    double last_attempt_s = -1e30;
};

// Selects candidates in stable descending score order. State is advanced only
// for candidates that are actually returned, so skipped detections do not
// consume a track's retry budget.
std::vector<CropReq> stage2_select_crops(const std::vector<Detection>& dets,
                                         double t_mono_s,
                                         const Stage2FilterConfig& cfg,
                                         std::map<uint32_t, Stage2TrackState>& state);

} // namespace vb
