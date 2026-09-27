// Internal: snapshot JPEG encoding (M1.8). Built without JPEG support this
// translation unit only provides the "unsupported" reply path.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vb/runtime.h"

namespace vb {

// B2: clips a track box (source-pixel floats, inclusive-exclusive after
// rounding) to the [0,w] x [0,h] image area and writes the integer rect that
// the crop memcpy may read. Returns false when the intersection is empty
// (box entirely outside the frame, or a degenerate box): callers must not
// fall back to a 1-pixel crop, which reads past the image buffer.
bool snapshot_clip_box(float x0, float y0, float x1, float y1, int w, int h,
                       int& cx0, int& cy0, int& cw, int& ch);

// Encode (optionally track-cropped, optionally downscaled to max_side) RGB888
// pixels from a snapshot-ring entry as JPEG. Returns false with err set when
// built without JPEG support ("snapshot unsupported") or on encode failure.
bool snapshot_encode_jpeg(const SnapshotRingEntry& e, uint32_t track_id, bool crop,
                          int max_side, std::vector<uint8_t>& jpeg, int& out_w,
                          int& out_h, std::string& err);

}  // namespace vb
