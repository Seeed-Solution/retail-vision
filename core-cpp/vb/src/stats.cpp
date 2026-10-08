// Per-stream metrics + stats-op JSON (spec BASE-1 §6.3 stats, M1.7).
#include <mutex>
#include "vb/runtime.h"

#include <algorithm>
#include <chrono>

namespace vb {

namespace {

// One-second FPS window, in the same epoch-ms unit as FrameBuf::wall_ms.
constexpr double kFpsWindowMs = 1000.0;

double wall_ms_now() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
               system_clock::now().time_since_epoch()).count();
}

double percentile(std::deque<float> sorted, double p) {
    if (sorted.empty()) return 0.0;
    std::sort(sorted.begin(), sorted.end());
    double idx = p * static_cast<double>(sorted.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, sorted.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

}  // namespace

void StreamMetrics::set_decode_path(const std::string& p) {
    std::lock_guard<std::mutex> lk(mu_);
    decode_ = p;
}

void StreamMetrics::record_state(const std::string& state, const std::string& error) {
    std::lock_guard<std::mutex> lk(mu_);
    state_ = state;
    error_ = error;
}

void StreamMetrics::record_frame(float inference_ms, float queue_delay_ms, double wall_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    ++processed_;
    inf_ms_.push_back(inference_ms);
    qd_ms_.push_back(queue_delay_ms);
    if (inf_ms_.size() > 256) inf_ms_.pop_front();
    if (qd_ms_.size() > 256) qd_ms_.pop_front();
    frame_walls_.push_back(wall_ms);
    while (!frame_walls_.empty() && wall_ms - frame_walls_.front() > kFpsWindowMs)
        frame_walls_.pop_front();
    last_wall_ = wall_ms;
}

void StreamMetrics::add_dropped(uint64_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    dropped_ += n;
}

void StreamMetrics::add_rate_skipped(uint64_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    rate_skipped_ += n;
}
void StreamMetrics::add_text_vote_dedup(uint64_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    text_vote_dedup_ += n;
}

void StreamMetrics::set_stage2_backend(const std::string& backend, bool fallback) {
    std::lock_guard<std::mutex> lk(mu_);
    stage2_backend_ = backend;
    fallback_active_ = fallback;
}

void StreamMetrics::record_stage2(float ms, bool failed) {
    std::lock_guard<std::mutex> lk(mu_);
    stage2_ms_.push_back(ms);
    if (stage2_ms_.size() > 256) stage2_ms_.pop_front();
    if (failed) ++stage2_errors_;
}

std::string StreamMetrics::state() const {
    std::lock_guard<std::mutex> lk(mu_);
    return state_;
}

uint64_t StreamMetrics::processed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return processed_;
}

double StreamMetrics::last_frame_wall_ms() const {
    std::lock_guard<std::mutex> lk(mu_);
    return last_wall_;
}

Json StreamMetrics::to_json(uint32_t stream_index) const {
    std::lock_guard<std::mutex> lk(mu_);
    Json j;
    j["stream_index"] = stream_index;
    j["state"] = state_;
    // The window is measured against the query time, not against the last
    // frame: a stream that stopped producing frames a while ago must report
    // 0 fps instead of the rate it had when it was still running.
    const double now = wall_ms_now();
    while (!frame_walls_.empty() && now - frame_walls_.front() > kFpsWindowMs)
        frame_walls_.pop_front();
    double fps = 0.0;
    if (!frame_walls_.empty()) {
        // This is a fixed one-second window. Using the shorter age of the
        // oldest retained frame as the denominator amplifies a single recent
        // frame into a misleading burst rate (for example, 1 frame 100 ms
        // ago becomes 10 fps). A stopped stream is still reported as zero
        // because stale frames were pruned above.
        fps = static_cast<double>(frame_walls_.size()) /
              (kFpsWindowMs / 1000.0);
    }
    j["fps"] = fps;
    j["decode"] = decode_;
    // Cumulative successful native processing count for this StreamState.
    // Read the member while mu_ is held; calling processed() here would
    // attempt to acquire the same non-recursive mutex a second time.
    j["processed_frames"] = processed_;
    j["inference_ms_p50"] = percentile(inf_ms_, 0.50);
    j["inference_ms_p95"] = percentile(inf_ms_, 0.95);
    j["queue_delay_ms_p95"] = percentile(qd_ms_, 0.95);
    j["dropped_frames"] = dropped_;
    j["event_backpressure"] = 0;
    j["rate_skipped"] = rate_skipped_;
    j["stage2_backend"] = stage2_backend_;
    j["stage2_ms_p95"] = percentile(stage2_ms_, .95);
    j["stage2_errors"] = stage2_errors_;
    j["fallback_active"] = fallback_active_;
    j["text_vote_dedup"] = text_vote_dedup_;
    if (!error_.empty()) j["error"] = error_;
    return j;
}

}  // namespace vb
