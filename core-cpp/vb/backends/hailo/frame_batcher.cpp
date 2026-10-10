// Per-stream frame queues feeding the shared Hailo batch (spec BASE-1 §M2.3).
//
// Copied from fall-detection platforms/rpi-hailo/src/frame_batcher.cpp
// (origin/main@eb72e1e, Apache-2.0; see NOTICE). Namespaced to vb::; logic is
// unchanged.
#include "frame_batcher.h"
#include <algorithm>
#include <stdexcept>

namespace vb {
FrameBatcher::FrameBatcher(int n, int b, int wait, size_t d) : queues_(n), batch_size_(b), wait_ms_(wait), depth_(d), drops_(n), hist_(b + 1) {
  if (n < 1 || b < 1 || wait < 0 || d < 1) throw std::invalid_argument("invalid batcher configuration");
}
bool FrameBatcher::enqueue(BatchFrame f) {
  std::lock_guard<std::mutex> lk(mutex_); if (stopped_ || f.stream < 0 || size_t(f.stream) >= queues_.size()) return false;
  auto &q = queues_[f.stream]; if (q.size() >= depth_) { q.pop_front(); ++drops_[f.stream]; } q.push_back(std::move(f)); cv_.notify_one(); return true;
}
bool FrameBatcher::take(std::vector<BatchFrame> &out) {
  std::unique_lock<std::mutex> lk(mutex_); out.clear();
  const auto queued = [&] { size_t count=0; for (const auto &q : queues_) count+=q.size(); return count; };
  cv_.wait(lk, [&] { return queued()>0 || stopped_; });
  if (out.empty() && stopped_) return false;
  auto deadline = std::chrono::steady_clock::time_point::max();
  for (const auto &q : queues_) if (!q.empty()) deadline = std::min(deadline, q.front().timestamp + std::chrono::milliseconds(wait_ms_));
  if (wait_ms_ > 0 && std::chrono::steady_clock::now() < deadline && queued()<size_t(batch_size_))
    cv_.wait_until(lk, deadline, [&] { return stopped_ || queued()>=size_t(batch_size_); });
  const size_t start = rr_; rr_ = (rr_ + 1) % queues_.size();
  // Round one takes the oldest item from each stream; later rounds only fill leftovers.
  for (size_t k = 0; k < queues_.size() && out.size() < size_t(batch_size_); ++k) { auto &q = queues_[(start + k) % queues_.size()]; if (!q.empty()) out.push_back(std::move(q.front())), q.pop_front(); }
  for (size_t k = 0; k < queues_.size() && out.size() < size_t(batch_size_); ++k) { auto &q = queues_[(start + k) % queues_.size()]; if (!q.empty()) out.push_back(std::move(q.front())), q.pop_front(); }
  ++hist_[out.size()]; return true;
}
void FrameBatcher::stop(bool discard) { std::lock_guard<std::mutex> lk(mutex_); stopped_ = true; if (discard) for (auto &q : queues_) q.clear(); cv_.notify_all(); }
BatchStats FrameBatcher::stats() const { std::lock_guard<std::mutex> lk(mutex_); return {drops_, hist_}; }
}
