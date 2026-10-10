// Rockchip MPP frame source (spec BASE-1 §M2.1): RTSP -> hardware H.264/H.265
// decode -> NV12 DMA-BUF handed to the inference context without a CPU copy.
#pragma once

#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// Builds the pipeline eagerly so an unusable codec, a missing GStreamer element
// or an mppvideodec without the DMA-BUF feature fails at add() time instead of
// silently decoding on the CPU. There is deliberately no software-decode
// fallback (fall platforms/rknn `rknn.strict`).
std::unique_ptr<FrameSource> make_mpp_source(const StreamSpec& s, std::string& err);

}  // namespace vb
