// Per-stream frame queues feeding the shared Hailo batch (spec BASE-1 §M2.3).
//
// Copied from fall-detection platforms/rpi-hailo/src/frame_batcher.{h,cpp}
// (origin/main@eb72e1e, Apache-2.0; see NOTICE). Namespaced to vb::; logic is
// unchanged. The vb runtime's ContextPool gathers frames across streams
// without waiting, so production wiring passes up to caps.max_batch ready
// frames straight to the runner; this batcher stays for the fall-shaped
// deployment path and is exercised by tests/test_hailo_backend.cpp.
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace vb {
struct BatchFrame { int stream = 0; uint64_t seq = 0; std::chrono::steady_clock::time_point timestamp{}; std::vector<uint8_t> rgb; };
struct BatchStats { std::vector<uint64_t> drops; std::vector<uint64_t> histogram; };
class FrameBatcher {
 public:
  FrameBatcher(int streams, int batch_size, int wait_ms, size_t depth = 2);
  bool enqueue(BatchFrame frame);
  bool take(std::vector<BatchFrame> &out);
  void stop(bool discard = true);
  BatchStats stats() const;
 private:
  mutable std::mutex mutex_; std::condition_variable cv_; std::vector<std::deque<BatchFrame>> queues_;
  int batch_size_, wait_ms_; size_t depth_; size_t rr_ = 0; bool stopped_ = false; std::vector<uint64_t> drops_, hist_;
};
}
