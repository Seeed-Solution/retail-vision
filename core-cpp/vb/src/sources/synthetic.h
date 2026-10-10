// Internal: synthetic frame source + inference backend (M1.7).
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// B3: add-time bounds for synthetic frame sizes. Each read() decodes into a
// host RGB888 buffer of w * h * 3 bytes, so the dimensions and their product
// are checked before anything is allocated; an oversized url is refused rather
// than clamped into something that overflows w * 3.
constexpr int kSyntheticMaxDim = 4096;
constexpr std::size_t kSyntheticMaxFrameBytes =
    static_cast<std::size_t>(kSyntheticMaxDim) *
    static_cast<std::size_t>(kSyntheticMaxDim) * 3;

// Returns false with err set when the size cannot be produced bounded.
bool synthetic_dims_ok(int w, int h, std::string& err);

// Deterministic test backend. backend_json keys (all optional):
//   max_batch (1..8, default 2), model_w/model_h (default 640),
//   boxes (detections per frame, default 3), infer_ms (simulated latency).
// Detections depend only on the frame seq (two streams at the same seq get
// identical boxes), so per-stream trackers assign ids starting from 1.
std::unique_ptr<Backend> make_synthetic_backend(const std::string& backend_json,
                                                std::string& err);

}  // namespace vb
