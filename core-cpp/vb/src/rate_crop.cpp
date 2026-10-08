#include "vb/rate_crop.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vb/letterbox.h"

namespace vb {

bool host_rgb_layout_bytes(int w, int h, int stride, size_t& packed_bytes) {
    packed_bytes = 0;
    if (w <= 0 || h <= 0 || stride <= 0 ||
        static_cast<size_t>(w) > kMaxHostRgbBytes / 3u) return false;
    const size_t row = static_cast<size_t>(w) * 3u;
    const size_t height = static_cast<size_t>(h);
    if (static_cast<size_t>(stride) < row ||
        height > kMaxHostRgbBytes / static_cast<size_t>(stride)) return false;
    packed_bytes = row * height;
    return true;
}

CropRectPx crop_rect_px(const float roi[4], int full_w, int full_h) {
    if (!roi || full_w <= 0 || full_h <= 0)
        throw std::invalid_argument("crop dimensions must be positive");
    for (int i = 0; i < 4; ++i)
        if (!std::isfinite(roi[i])) throw std::invalid_argument("crop must be finite");
    if (!(roi[0] >= 0 && roi[0] < roi[2] && roi[2] <= 1 &&
          roi[1] >= 0 && roi[1] < roi[3] && roi[3] <= 1))
        throw std::invalid_argument("crop bounds invalid");
    if (roi[2] - roi[0] < .05f || roi[3] - roi[1] < .05f)
        throw std::invalid_argument("crop span must be >= 0.05");
    // Keep the product in double until it is proven inside the int bounds;
    // float multiplication at INT_MAX can round 1.0*width to 2^31.
    const double dx0 = std::floor(static_cast<double>(roi[0]) * full_w);
    const double dy0 = std::floor(static_cast<double>(roi[1]) * full_h);
    const double dx1 = std::ceil(static_cast<double>(roi[2]) * full_w);
    const double dy1 = std::ceil(static_cast<double>(roi[3]) * full_h);
    if (dx0 < 0 || dy0 < 0 || dx1 > full_w || dy1 > full_h)
        throw std::invalid_argument("crop pixel bounds overflow");
    int x0 = static_cast<int>(dx0);
    int y0 = static_cast<int>(dy0);
    int x1 = static_cast<int>(dx1);
    int y1 = static_cast<int>(dy1);
    // Decoder surfaces require even origins and extents. Expand outward,
    // then clip at the source boundary (which may itself be odd).
    if (x0 & 1) --x0; if (y0 & 1) --y0;
    if ((x1 & 1) && x1 < full_w) ++x1;
    if ((y1 & 1) && y1 < full_h) ++y1;
    x0 = std::clamp(x0, 0, full_w - 1); y0 = std::clamp(y0, 0, full_h - 1);
    x1 = std::clamp(x1, x0 + 1, full_w); y1 = std::clamp(y1, y0 + 1, full_h);
    return {x0, y0, x1 - x0, y1 - y0};
}

LetterboxGeom crop_geom(int full_w, int full_h, const CropRectPx& crop,
                        int model_w, int model_h, Align align) {
    if (crop.x0 < 0 || crop.y0 < 0 || crop.x0 >= full_w || crop.y0 >= full_h ||
        crop.w <= 0 || crop.h <= 0 || crop.w > full_w - crop.x0 || crop.h > full_h - crop.y0)
        throw std::invalid_argument("crop rectangle outside source");
    LetterboxGeom local = letterbox_fit(crop.w, crop.h, model_w, model_h, align);
    local.src_w = full_w;
    local.src_h = full_h;
    const double px = static_cast<double>(local.pad_x) - static_cast<double>(crop.x0) * local.scale;
    const double py = static_cast<double>(local.pad_y) - static_cast<double>(crop.y0) * local.scale;
    local.pad_x = static_cast<float>(std::fabs(px - std::round(px)) < 1e-4 ? std::round(px) : px);
    local.pad_y = static_cast<float>(std::fabs(py - std::round(py)) < 1e-4 ? std::round(py) : py);
    return local;
}

bool host_crop_frame(FrameBuf& frame, const CropRectPx& crop, std::string& err) {
    if (frame.mem != Mem::Host || !frame.host ||
        (frame.fmt != PixFmt::RGB888 && frame.fmt != PixFmt::BGR888)) {
        err = "roi_crop requires host RGB888 or BGR888 source";
        return false;
    }
    const int full_w = frame.full_w > 0 ? frame.full_w : frame.w;
    const int full_h = frame.full_h > 0 ? frame.full_h : frame.h;
    const int full_stride = frame.full_stride > 0 ? frame.full_stride : frame.stride;
    size_t packed_bytes = 0;
    if (!host_rgb_layout_bytes(full_w, full_h, full_stride, packed_bytes)) {
        err = "roi_crop source layout invalid or exceeds 64 MiB";
        return false;
    }
    if (crop.x0 < 0 || crop.y0 < 0 || crop.w <= 0 || crop.h <= 0 ||
        crop.x0 > full_w - crop.w || crop.y0 > full_h - crop.h ||
        crop.x0 >= full_w || crop.y0 >= full_h) {
        err = "roi_crop rectangle or source stride invalid";
        return false;
    }
    if (!frame.full_host) {
        frame.full_host = frame.host;
    }
    frame.full_stride = full_stride;
    frame.host = frame.full_host + static_cast<size_t>(crop.y0) * frame.full_stride +
                 static_cast<size_t>(crop.x0) * 3;
    frame.w = crop.w;
    frame.h = crop.h;
    frame.stride = frame.full_stride;
    frame.full_w = full_w;
    frame.full_h = full_h;
    frame.crop_x0 = crop.x0;
    frame.crop_y0 = crop.y0;
    return true;
}

bool RateLimiter::set_max_fps(double max_fps) {
    if (!std::isfinite(max_fps) || max_fps < 0.0) return false;
    max_fps_ = max_fps;
    period_s_ = max_fps > 0.0 ? 1.0 / max_fps : 0.0;
    next_due_s_ = 0.0;
    last_timestamp_s_ = 0.0;
    have_timestamp_ = false;
    return true;
}

bool RateLimiter::accept(double t) {
    if (!std::isfinite(t)) return false;
    if (max_fps_ == 0.0) { last_timestamp_s_ = t; have_timestamp_ = true; return true; }
    if (have_timestamp_ && t < last_timestamp_s_) {
        have_timestamp_ = false; // source clock reset: accept this frame
    }
    constexpr double kTolerance = 0.002;
    if (!have_timestamp_) {
        have_timestamp_ = true; last_timestamp_s_ = t; next_due_s_ = t + period_s_; return true;
    }
    last_timestamp_s_ = t;
    if (t + kTolerance < next_due_s_) return false;
    // Keep the original deadline phase when a frame arrives late. Re-anchoring
    // at t + period_s_ makes small source-clock jitter accumulate into a much
    // lower effective rate. Use fmod instead of a loop so a long source gap
    // remains bounded while the next deadline stays on the same phase grid.
    const double threshold = t + kTolerance;
    const double lag = threshold - next_due_s_;
    const double remainder = std::fmod(lag, period_s_);
    double advance = period_s_ - remainder;
    if (!(advance > 0.0) || !std::isfinite(advance)) advance = period_s_;
    next_due_s_ = threshold + advance;
    return true;
}

}  // namespace vb
