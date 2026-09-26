// Per-stream metrics + stats-op JSON (spec BASE-1 §6.3 stats, M1.7).
#include <mutex>
#include "vb/runtime.h"

#include <algorithm>

namespace vb {

namespace {

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
    while (!frame_walls_.empty() && wall_ms - frame_walls_.front() > 1000.0)
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
    double fps = 0.0;
    if (!frame_walls_.empty()) {
        double span = (last_wall_ - frame_walls_.front()) / 1000.0;
        fps = span > 0.05 ? static_cast<double>(frame_walls_.size()) / span
                          : static_cast<double>(frame_walls_.size());
    }
    j["fps"] = fps;
    j["decode"] = decode_;
    j["inference_ms_p50"] = percentile(inf_ms_, 0.50);
    j["inference_ms_p95"] = percentile(inf_ms_, 0.95);
    j["queue_delay_ms_p95"] = percentile(qd_ms_, 0.95);
    j["dropped_frames"] = dropped_;
    j["event_backpressure"] = 0;
    j["rate_skipped"] = rate_skipped_;
    j["stage2_backend"] = "";
    j["stage2_ms_p95"] = 0.0;
    j["stage2_errors"] = 0;
    if (!error_.empty()) j["error"] = error_;
    return j;
}

}  // namespace vb
