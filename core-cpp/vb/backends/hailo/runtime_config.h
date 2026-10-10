// Hailo runtime environment parsing (spec BASE-1 §M2.3).
//
// Copied from fall-detection platforms/rpi-hailo/src/runtime_config.h
// (origin/main@eb72e1e, Apache-2.0; see NOTICE). Namespaced to
// vb::hailo_config; logic is unchanged. The queue-depth / drop-on-latency /
// codec-chain values feed hailo_source.cpp, which takes fall main.cpp's
// pipelinePrefix/sharedPipeline GStreamer pipeline with the a2d0e58
// codec-selected decode chain.
#pragma once

#include <charconv>
#include <stdexcept>
#include <string>

namespace vb {
namespace hailo_config {

inline int parseNonNegativeInt(const char* name, const std::string& value, int max_value = -1) {
  if (value.empty()) throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
  int parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 0 ||
      (max_value >= 0 && parsed > max_value)) {
    throw std::invalid_argument(std::string(name) + " must be a non-negative integer" +
                                (max_value >= 0 ? " within range" : ""));
  }
  return parsed;
}

inline int parseQueueDepth(const std::string& value) {
  const int parsed = parseNonNegativeInt("INFERENCE_QUEUE_DEPTH", value, 8);
  if (parsed < 1) throw std::invalid_argument("INFERENCE_QUEUE_DEPTH must be in range 1..8");
  return parsed;
}

inline bool parseDropOnLatency(const std::string& value) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::invalid_argument("RTSP_DROP_ON_LATENCY must be true, false, 1, or 0");
}

// Pi 5 has hardware decode for HEVC only (rpi-hevc-dec, /dev/video19); H.264 is
// decoded in software by avdec_h264. h265 pins the stateless V4L2 decoder so a
// missing device fails at startup instead of silently falling back to software.
// At 1080p the decoder only offers DMA_DRM (NV12 + Broadcom SAND128 modifier),
// which videoconvert cannot read, so the GPU converts and scales before download.
inline const char* rtspDepayChain(const std::string& codec) {
  if (codec == "h264") return "rtph264depay ! h264parse ! decodebin";
  if (codec == "h265")
    return "rtph265depay ! h265parse ! v4l2slh265dec ! glupload ! glcolorconvert ! glcolorscale ! "
           "video/x-raw(memory:GLMemory),format=RGBA,width=640,height=640 ! gldownload";
  throw std::invalid_argument("RTSP_CODEC must be h264 or h265");
}

}  // namespace hailo_config
}  // namespace vb
