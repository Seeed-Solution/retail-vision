// Internal: ContextPool definition shared by pool.cpp and runtime.cpp.
#pragma once

#include <cstddef>
#include "vb/runtime.h"

namespace vb {

// VBR1 declares kpt_per_det and attr_per_det as u8 (§6.3), so a frame whose
// counts exceed 255 cannot be encoded faithfully. Returns false with err set;
// the caller must drop the frame instead of sending the full payload under a
// clamped count (which would desynchronise the receiver).
bool vbr1_counts_ok(size_t kpt_per_det, size_t attr_per_det, std::string& err);

// Copies one analyzer's contiguous per-track attribute block (`scratch` holds
// n_tracks * n_attr floats, track-major) into the VBR1 attribute block, whose
// layout is [track][attribute] with the summed span of every analyzer as its
// stride (§6.3). Each analyzer writes into its own buffer, so two analyzers can
// never land in each other's slots.
void interleave_attrs(const float* scratch, size_t n_tracks, size_t n_attr,
                      size_t total_attrs, size_t attr_off, float* out);

class ContextPool {
public:
    explicit ContextPool(Runtime* rt) : rt_(rt) {}

    void start();
    void stop();

private:
    void worker(InferenceContext* ctx);
    void process(const std::shared_ptr<StreamState>& s, FrameBuf& f, InferenceContext* ctx);
    void apply_pending_controls(StreamState& s);
    void emit_events(StreamState& s, const FrameMeta& m,
                     const std::vector<AnalyzerEvent>& events, const char* analyzer_name);
    // Returns false (with err set) when the frame cannot be represented in
    // VBR1: the per-detection keypoint/attribute counts are u8 fields, and a
    // clamped count with a full payload would break record framing.
    bool encode_frame_record(StreamState& s, const FrameBuf& f,
                             const DetectionResult& res, const std::vector<Track>& alive,
                             const std::vector<float>& attrs, size_t attr_total,
                             std::string& err);

    Runtime* rt_;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_;
    size_t cursor_ = 0;
    bool stopping_ = false;
};

}  // namespace vb
