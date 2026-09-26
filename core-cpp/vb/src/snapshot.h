// Internal: snapshot JPEG encoding (M1.8). Built without JPEG support this
// translation unit only provides the "unsupported" reply path.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vb/runtime.h"

namespace vb {

// Encode (optionally track-cropped, optionally downscaled to max_side) RGB888
// pixels from a snapshot-ring entry as JPEG. Returns false with err set when
// built without JPEG support ("snapshot unsupported") or on encode failure.
bool snapshot_encode_jpeg(const SnapshotRingEntry& e, uint32_t track_id, bool crop,
                          int max_side, std::vector<uint8_t>& jpeg, int& out_w,
                          int& out_h, std::string& err);

}  // namespace vb
