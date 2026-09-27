// Generic GStreamer frame source (spec BASE-1 §8 M1.9).
// Compiled only when VB_WITH_GST=ON. Pipeline:
//   uridecodebin uri=<url> ! videoconvert ! videoscale !
//   video/x-raw,format=RGB ! appsink max-buffers=1 drop=true sync=false
//
// The url is only ever applied through the uridecodebin "uri" property; a
// verbatim launch string (e.g. "videotestsrc num-buffers=3", which selects
// arbitrary installed GStreamer elements) is a dev/test affordance gated on
// the §6.12 dev-mode switch (VB_PRODUCTION unset), not on the stream config.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// ---- url policy (§6.12 production gate; usable without GStreamer) ----

// Largest accepted decoded width/height. Longer sides are refused by the
// pipeline caps and re-checked in read() before allocating.
constexpr int kGstMaxDim = 4096;
// Largest RGB888 frame read() will allocate (kGstMaxDim^2 * 3 bytes).
constexpr std::size_t kGstMaxFrameBytes =
    static_cast<std::size_t>(kGstMaxDim) * static_cast<std::size_t>(kGstMaxDim) * 3;
// open() boundary: the pipeline must reach PLAYING (live sources: connected;
// file/HTTP sources: prerolled) within this many seconds. Matches the §6.6
// open_timeout_s default; the runtime applies its own add deadline around it.
constexpr double kGstOpenTimeoutS = 5.0;

// Accepted URI schemes for a stream url.
inline bool source_scheme_allowed(const std::string& scheme) {
    return scheme == "rtsp" || scheme == "rtsps" || scheme == "http" ||
           scheme == "https" || scheme == "file";
}

// §6.12: dev mode is off when the production image sets VB_PRODUCTION=1; the
// same switch gates both raw tensors and raw GStreamer launch strings.
inline bool source_dev_pipeline_allowed() {
    return std::getenv("VB_PRODUCTION") == nullptr;
}

enum class SourceUrlKind {
    Uri,          // a scheme:// url, applied through the element property
    DevPipeline,  // a verbatim launch string (dev mode only)
    Rejected,     // err explains why
};

// Classifies a stream url. Every url that reaches create_source/open() goes
// through this: unknown schemes and, in production, launch strings are
// refused instead of being handed to gst_parse_launch.
inline SourceUrlKind classify_source_url(const std::string& url, std::string& err) {
    err.clear();
    if (url.empty()) {
        err = "empty url";
        return SourceUrlKind::Rejected;
    }
    for (char c : url) {
        if (c < 0x20 || c == 0x7f) {
            err = "url contains control characters";
            return SourceUrlKind::Rejected;
        }
    }
    // A uri is "scheme://..." with an RFC 3986 scheme in front. Anything else
    // (including a launch string that merely embeds a uri) is a pipeline head.
    const std::size_t sep = url.find("://");
    std::string scheme;
    if (sep != std::string::npos) {
        scheme.reserve(sep);
        for (std::size_t i = 0; i < sep; ++i) {
            const unsigned char c = static_cast<unsigned char>(url[i]);
            const bool alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                               (c >= '0' && c <= '9');
            if (!alnum && c != '+' && c != '-' && c != '.') {
                scheme.clear();
                break;
            }
            scheme.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
        }
    }
    if (!scheme.empty()) {
        if (!source_scheme_allowed(scheme)) {
            err = "unsupported url scheme '" + scheme +
                  "': expected rtsp, rtsps, http, https or file";
            return SourceUrlKind::Rejected;
        }
        return SourceUrlKind::Uri;
    }
    // A verbatim launch string selects arbitrary installed GStreamer elements,
    // so it is a dev-mode affordance and never a production stream address.
    if (!source_dev_pipeline_allowed()) {
        err = "raw gstreamer pipeline strings are not allowed in production "
              "(VB_PRODUCTION is set); use a rtsp/rtsps/http/https/file url";
        return SourceUrlKind::Rejected;
    }
    return SourceUrlKind::DevPipeline;
}

// Returns a FrameSource reading RGB888 host frames via appsink. Returns
// nullptr with err set when the url is rejected, GStreamer is unavailable, or
// the pipeline cannot be built. The source is not started here: the caller
// calls open(), which only reports success once the pipeline is running.
std::unique_ptr<FrameSource> make_gst_source(const StreamSpec& s, std::string& err);

}  // namespace vb
