#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "vb/types.h"
#include "vb/backend.h"

namespace vb {

// Bound packed RGB snapshots and the strided host backing store.
constexpr size_t kMaxHostRgbBytes = 64u * 1024u * 1024u;
bool host_rgb_layout_bytes(int w, int h, int stride, size_t& packed_bytes);

// Source-normalized crop rectangle, expressed in full-frame pixels.
struct CropRectPx {
    int x0 = 0, y0 = 0, w = 0, h = 0;
};

// Convert a validated [x0,y0,x1,y1] request to pixel bounds. The lower edge
// is floored and the upper edge ceiled so the requested source area is kept.
CropRectPx crop_rect_px(const float roi[4], int full_w, int full_h);

// Geometry for a crop that is physically represented as a local frame while
// detections remain normalized in the original full source. pad_x/pad_y may
// therefore be negative (crop origin correction).
LetterboxGeom crop_geom(int full_w, int full_h, const CropRectPx& crop,
                        int model_w, int model_h, Align align);

// Apply a host RGB/BGR crop in-place without copying. The backing allocation
// remains owned by FrameBuf::hold and the original geometry is retained.
bool host_crop_frame(FrameBuf& frame, const CropRectPx& crop, std::string& err);

// Stream timestamp limiter. It has no burst on recovery: after an accepted
// timestamp, next_due is at least accepted_timestamp + period.
class RateLimiter {
public:
    explicit RateLimiter(double max_fps = 0.0) { set_max_fps(max_fps); }
    bool set_max_fps(double max_fps);
    bool accept(double timestamp_s);
    double max_fps() const { return max_fps_; }

private:
    double max_fps_ = 0.0;
    double period_s_ = 0.0;
    double next_due_s_ = 0.0;
    double last_timestamp_s_ = 0.0;
    bool have_timestamp_ = false;
};

}  // namespace vb
