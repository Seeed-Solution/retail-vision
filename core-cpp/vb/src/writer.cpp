// Writer: bounded frame queue (drop-oldest) + never-dropped event queue with
// backpressure blocking (spec BASE-1 §5.4, M1.7).
#include <mutex>
#include <thread>
#include "vb/runtime.h"

#include <algorithm>
#include <chrono>

namespace vb {

namespace {

uint64_t steady_ms() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

}  // namespace

Writer::Writer(WriterConfig cfg) : cfg_(cfg) {}

Writer::~Writer() { stop(); }

void Writer::start(Sink sink) {
    sink_ = std::move(sink);
    {
        std::lock_guard<std::mutex> lk(mu_);
        running_ = true;
        stopping_ = false;
        stop_deadline_ms_ = 0;
    }
    thread_ = std::thread([this] { run(); });
}

void Writer::request_stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopping_ = true;
        // The flush window is armed once; repeated calls leave it alone.
        if (stop_deadline_ms_.load() == 0) {
            stop_deadline_ms_ = steady_ms() +
                                static_cast<uint64_t>(std::max(0, cfg_.stop_flush_ms));
        }
    }
    // Release both producer waits: cv_full_ (event queue over the backpressure
    // limit) and cv_pop_ (the writer thread).
    cv_pop_.notify_all();
    cv_full_.notify_all();
}

bool Writer::flush_expired() const {
    uint64_t deadline = stop_deadline_ms_.load(std::memory_order_relaxed);
    return deadline != 0 && steady_ms() >= deadline;
}

void Writer::stop() {
    request_stop();
    if (thread_.joinable()) thread_.join();
    // Drain what is left on the caller thread (flush semantics). The sink
    // still sees flush_expired(), so a peer that is gone cannot block here.
    std::lock_guard<std::mutex> lk(mu_);
    while (!q_.empty()) {
        Entry& e = q_.front();
        if (sink_) sink_(e.bytes.data(), e.bytes.size());
        q_.pop_front();
    }
    running_ = false;
}

void Writer::push_frame(std::vector<uint8_t> rec) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!running_) return;
    while (frames_queued_ >= cfg_.frame_queue) {
        // Drop the oldest frame record.
        bool dropped_one = false;
        for (auto it = q_.begin(); it != q_.end(); ++it) {
            if (it->is_frame) {
                q_.erase(it);
                --frames_queued_;
                ++frames_dropped_;
                dropped_one = true;
                break;
            }
        }
        if (!dropped_one) break;  // nothing droppable; exceed cap
    }
    ++frames_queued_;
    q_.push_back(Entry{true, std::move(rec)});
    cv_pop_.notify_one();
}

void Writer::push_event(std::vector<uint8_t> rec) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!running_) return;
    if (events_queued_ >= cfg_.event_backpressure_limit) {
        // Block the producing (context) thread until space frees up.
        ++event_backpressure_;
        cv_full_.wait(lk, [this] { return stopping_ || events_queued_ < cfg_.event_backpressure_limit; });
        if (stopping_) return;
    }
    ++events_queued_;
    q_.push_back(Entry{false, std::move(rec)});
    cv_pop_.notify_one();
}

void Writer::run() {
    writer_thread_ = std::this_thread::get_id();
    std::vector<uint8_t> buf;
    for (;;) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_pop_.wait(lk, [this] { return stopping_ || !q_.empty(); });
        if (q_.empty() && stopping_) break;
        Entry e = std::move(q_.front());
        q_.pop_front();
        if (e.is_frame) --frames_queued_;
        else {
            --events_queued_;
            cv_full_.notify_one();
        }
        lk.unlock();
        if (sink_) sink_(e.bytes.data(), e.bytes.size());
    }
}

uint64_t Writer::frames_dropped() const {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(mu_));
    return frames_dropped_;
}

uint64_t Writer::event_backpressure() const {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(mu_));
    return event_backpressure_;
}

}  // namespace vb
