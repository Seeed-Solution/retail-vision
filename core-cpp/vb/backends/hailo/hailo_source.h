// Hailo frame source: fall's GStreamer pipeline as a §6.1 FrameSource
// (spec BASE-1 §M2.3). Compiled only with VB_WITH_GST=ON.
//
// Takes fall main.cpp (origin/main@eb72e1e) pipelinePrefix / sharedPipeline:
// the appsink shared-runtime mode (fall main.cpp:89) and the codec-selected
// decode chain of a2d0e58 (rtspDepayChain in runtime_config.h). URL schemes
// follow fall: rtsp, plus the test:// / file:// branches of pipelinePrefix for
// smoke tests. Unlike the generic M1.9 source this pipeline is built with
// gst_parse_launch (fall's shape, needed for the pinned h265 GL chain); the
// url never reaches the parser unvalidated — see validate_source_url().
#pragma once

#include <string>

#include "runtime_config.h"
#include "vb/backend.h"

namespace vb {
namespace hailo_source {

// Backend-level rtsp settings shared by every stream (fall reads them from
// the environment; here they come from backend.json "rtsp").
struct RtspSettings {
    int latency_ms = 100;
    bool drop_on_latency = false;
    std::string codec = "h264";  // h264 | h265

    // Throws std::invalid_argument on values rtspDepayChain / the parsers
    // reject, so the backend factory can surface it as a config error.
    // Inline in the header: the backend factory validates before any GStreamer
    // code exists (the source itself is only built with VB_WITH_GST=ON).
    void validate() const {
        (void)hailo_config::rtspDepayChain(codec);
        if (latency_ms < 0) throw std::invalid_argument("rtsp.latency_ms must be >= 0");
    }
};

// '!' is excluded because the url is composed into a launch string; fall's
// pipelinePrefix has the same exposure, this just fails at config time.
bool validate_source_url(const std::string& url, std::string& err);

// The launch-string prefix for one stream (fall pipelinePrefix), sans the
// shared tail. Exposed for tests.
std::string pipeline_prefix(const std::string& url, const RtspSettings& rtsp);

}  // namespace hailo_source

// Registered by hailo_backend.cpp; compiled only when VB_WITH_GST=ON.
std::unique_ptr<FrameSource> make_hailo_source(const StreamSpec& s,
                                               const hailo_source::RtspSettings& rtsp,
                                               std::string& err);

}  // namespace vb
