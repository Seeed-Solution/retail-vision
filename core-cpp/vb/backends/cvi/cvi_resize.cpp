#include "cvi_resize.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vb {
namespace {
constexpr int kCoefBits = 11;
constexpr int kCoefScale = 1 << kCoefBits;
constexpr int kOutputShift = kCoefBits * 2;
constexpr int kOutputDelta = 1 << (kOutputShift - 1);

int quantize_coeff(float value) {
  return std::clamp(static_cast<int>(std::lrintf(value * kCoefScale)), 0,
                    kCoefScale);
}

void axis_coeff(float coordinate, int limit, bool clamp_fraction, int& offset,
                int& a0, int& a1) {
  const float floor_coordinate = std::floor(coordinate);
  offset = static_cast<int>(floor_coordinate);
  float fraction = coordinate - floor_coordinate;
  if (offset < 0) {
    offset = 0;
    if (clamp_fraction) fraction = 0.0f;
  } else if (offset >= limit - 1) {
    offset = limit - 1;
    if (clamp_fraction) fraction = 0.0f;
  }
  a1 = quantize_coeff(fraction);
  a0 = kCoefScale - a1;
}
}  // namespace

bool cvi_resize_u8_inter_linear(const uint8_t* src, int src_w, int src_h,
                                int src_stride, float x0, float y0, float x1,
                                float y1, int dst_w, int dst_h,
                                std::vector<uint8_t>& dst) {
  if (!src || src_w <= 0 || src_h <= 0 || src_stride < src_w * 3 ||
      dst_w <= 0 || dst_h <= 0 || !(x0 >= 0.0f && y0 >= 0.0f && x1 > x0 &&
                                    y1 > y0 && x1 <= src_w && y1 <= src_h))
    return false;
  dst.assign(static_cast<size_t>(dst_w) * dst_h * 3, 0);
  const double scale_x = (static_cast<double>(x1) - x0) / dst_w;
  const double scale_y = (static_cast<double>(y1) - y0) / dst_h;
  for (int dy = 0; dy < dst_h; ++dy) {
    int sy = 0, b0 = 0, b1 = 0;
    const float y_coordinate = static_cast<float>(static_cast<double>(y0) +
                                                   (dy + 0.5) * scale_y - 0.5);
    // OpenCV's generic resize clips the vertical source row through its
    // border-row lookup while retaining the computed beta coefficients.
    axis_coeff(y_coordinate, src_h, false, sy, b0, b1);
    const int raw_sy = static_cast<int>(std::floor(y_coordinate));
    const int sy1 = (raw_sy < 0 || raw_sy >= src_h - 1)
                        ? sy
                        : std::min(sy + 1, src_h - 1);
    const uint8_t* row0 = src + static_cast<size_t>(sy) * src_stride;
    const uint8_t* row1 = src + static_cast<size_t>(sy1) * src_stride;
    for (int dx = 0; dx < dst_w; ++dx) {
      int sx = 0, a0 = 0, a1 = 0;
      axis_coeff(static_cast<float>(static_cast<double>(x0) +
                                   (dx + 0.5) * scale_x - 0.5),
                 src_w, true, sx, a0, a1);
      const uint8_t* p00 = row0 + static_cast<size_t>(sx) * 3;
      const uint8_t* p01 = row0 + static_cast<size_t>(std::min(sx + 1, src_w - 1)) * 3;
      const uint8_t* p10 = row1 + static_cast<size_t>(sx) * 3;
      const uint8_t* p11 = row1 + static_cast<size_t>(std::min(sx + 1, src_w - 1)) * 3;
      uint8_t* out = dst.data() + (static_cast<size_t>(dy) * dst_w + dx) * 3;
      for (int c = 0; c < 3; ++c) {
        const int top = a0 * p00[c] + a1 * p01[c];
        const int bottom = a0 * p10[c] + a1 * p11[c];
        // OpenCV 4.8's optimized uint8 path first truncates each horizontal
        // accumulator by four bits, then applies the vertical fixed-point
        // multiply and the final two-bit rounding (resize.cpp:1827-1834).
        const int value = ((b0 * (top >> 4)) >> 16) +
                          ((b1 * (bottom >> 4)) >> 16);
        out[c] = static_cast<uint8_t>(std::clamp((value + 2) >> 2, 0, 255));
      }
    }
  }
  return true;
}
}  // namespace vb
