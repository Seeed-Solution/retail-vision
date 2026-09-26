// Generic GStreamer frame source (spec BASE-1 §8 M1.9).
// Compiled only when VB_WITH_GST=ON. Pipeline:
//   uridecodebin uri=... ! videoconvert ! videoscale !
//   video/x-raw,format=RGB ! appsink max-buffers=1 drop=true sync=false
#pragma once

#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// Returns a FrameSource reading RGB888 host frames via appsink.
// URLs containing "://" are opened with uridecodebin; anything else is used
// verbatim as the source element of the pipeline (dev/test pipelines such as
// "videotestsrc num-buffers=3").
// Returns nullptr with err set if GStreamer is not available or the pipeline
// cannot be built.
std::unique_ptr<FrameSource> make_gst_source(const StreamSpec& s, std::string& err);

}  // namespace vb
