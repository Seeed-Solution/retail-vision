// Internal: ContextPool definition shared by pool.cpp and runtime.cpp.
#pragma once

#include "vb/runtime.h"

namespace vb {

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
    void encode_frame_record(StreamState& s, const FrameBuf& f,
                             const DetectionResult& res, const std::vector<Track>& alive,
                             const std::vector<float>& attrs, size_t attr_total);

    Runtime* rt_;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_;
    size_t cursor_ = 0;
    bool stopping_ = false;
};

}  // namespace vb
