#pragma once

#include <cstdint>
#include <vector>

namespace vb {

// OpenCV 4.8 uint8 INTER_LINEAR-compatible resize for the CVI RGB/BGR paths.
// The source window is expressed in source-image coordinates and may be a
// fractional ROI; output is interleaved uint8 with three channels.
bool cvi_resize_u8_inter_linear(const uint8_t* src, int src_w, int src_h,
                                int src_stride, float x0, float y0, float x1,
                                float y1, int dst_w, int dst_h,
                                std::vector<uint8_t>& dst);

}  // namespace vb
