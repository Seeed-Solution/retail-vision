// One-shot failure latch for the Hailo runner (spec BASE-1 §M2.3).
//
// Copied from fall-detection platforms/rpi-hailo/src/runner_lifecycle.h
// (origin/main@eb72e1e, Apache-2.0; see NOTICE). Namespaced to vb::; logic is
// unchanged.
#pragma once

#include <atomic>
#include <utility>

namespace vb {

class RunnerLifecycle {
 public:
  template <typename PostQuit>
  void fail(PostQuit &&post_quit) {
    if (!failed_.exchange(true, std::memory_order_acq_rel))
      std::forward<PostQuit>(post_quit)();
  }

  bool shouldRunLoop() const {
    return !failed_.load(std::memory_order_acquire);
  }

  int exitCode(int current) const {
    return current == 0 && !shouldRunLoop() ? 4 : current;
  }

 private:
  std::atomic<bool> failed_{false};
};

}  // namespace vb
