// Internal: synthetic frame source + inference backend (M1.7).
#pragma once

#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// Deterministic test backend. backend_json keys (all optional):
//   max_batch (1..8, default 2), model_w/model_h (default 640),
//   boxes (detections per frame, default 3), infer_ms (simulated latency).
// Detections depend only on the frame seq (two streams at the same seq get
// identical boxes), so per-stream trackers assign ids starting from 1.
std::unique_ptr<Backend> make_synthetic_backend(const std::string& backend_json,
                                                std::string& err);

}  // namespace vb
