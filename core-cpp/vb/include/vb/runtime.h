// Runtime core: LatestFrame, Writer, stream metrics, ContextPool, Runtime
// (control ops, stats, stream_state) and the fd control-line server
// (spec BASE-1 §5.4 native thread model, §6.3 control protocol, M1.7/M1.8).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "vb/analyzer.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/tracker.h"
#include "vb/wire.h"

namespace vb {

constexpr const char* runtime_version() { return "0.1.0"; }

// ---- LatestFrame (§5.4: keep only the newest frame; drop-old counts) ----
class LatestFrame {
public:
    void put(FrameBuf f);              // drops (releases) the old frame if present
    bool take(FrameBuf& out);
    uint64_t dropped() const;

private:
    mutable std::mutex mu_;
    std::optional<FrameBuf> slot_;
    uint64_t dropped_ = 0;
};

// ---- Writer (§5.4: frame records drop-oldest, event records never) ----
struct WriterConfig {
    size_t frame_queue = 256;
    size_t event_backpressure_limit = 4096;
};

class Writer {
public:
    explicit Writer(WriterConfig cfg = WriterConfig());

    using Sink = std::function<void(const uint8_t* data, size_t len)>;

    void start(Sink sink);
    // Flushes queued records (sink called on the caller thread) and joins.
    void stop();

    void push_frame(std::vector<uint8_t> rec);  // may drop the oldest frame record
    void push_event(std::vector<uint8_t> rec);  // never dropped; blocks past the limit

    uint64_t frames_dropped() const;
    uint64_t event_backpressure() const;

private:
    struct Entry {
        bool is_frame;
        std::vector<uint8_t> bytes;
    };
    void run();

    WriterConfig cfg_;
    Sink sink_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_pop_;   // space freed / new data
    std::condition_variable cv_full_;  // event queue over limit -> block pusher
    std::deque<Entry> q_;
    size_t frames_queued_ = 0, events_queued_ = 0;
    uint64_t frames_dropped_ = 0, event_backpressure_ = 0;
    bool running_ = false, stopping_ = false;
    std::thread::id writer_thread_{};
};

// ---- Per-stream metrics (§6.3 stats op) ----
class StreamMetrics {
public:
    void set_decode_path(const std::string& p);
    void record_state(const std::string& state, const std::string& error);
    void record_frame(float inference_ms, float queue_delay_ms, double wall_ms);
    void add_dropped(uint64_t n);
    void add_rate_skipped(uint64_t n);

    std::string state() const;
    // One entry of the stats op "streams" array (mutex-guarded snapshot).
    Json to_json(uint32_t stream_index) const;
    // Convenience counters for tests.
    uint64_t processed() const;
    double last_frame_wall_ms() const;

private:
    mutable std::mutex mu_;
    std::string state_ = "starting", error_, decode_;
    uint64_t processed_ = 0, dropped_ = 0, rate_skipped_ = 0;
    std::deque<float> inf_ms_, qd_ms_;
    std::deque<double> frame_walls_;
    double last_wall_ = 0;
};

// ---- Runtime config (§6.9 rule 4: no stream list in this file) ----
struct RuntimeConfig {
    std::string backend_name = "synthetic";
    std::string backend_json = "{}";
    int contexts = 1;
    double open_timeout_s = 5.0;
    double reconnect_delay_s = 1.0;
    double status_interval_s = 1.0;
    TrackerConfig tracker;
    std::vector<std::string> plugin_paths;  // preloaded analyzer plugins
    int snapshot_ring = 2;

    static RuntimeConfig load(const std::string& path, std::string& err);
    static RuntimeConfig from_json(const Json& j, std::string& err);
};

class ContextPool;  // below

struct PendingAnalyzerCfg {
    std::string name, json;
    bool done = false, ok = false;
    std::string err;
};

// Per-frame snapshot-ring entry: host RGB copy plus this frame's track
// boxes in source-pixel coords (for track crop requests).
struct SnapBox {
    uint32_t track_id;
    float x0, y0, x1, y1;  // source pixels (inclusive-exclusive)
};
struct SnapshotRingEntry {
    uint64_t seq = 0;
    int w = 0, h = 0;
    std::vector<uint8_t> pixels;  // RGB888, stride = w*3
    std::vector<SnapBox> boxes;
};

// Per-stream state. The busy flag guarantees the tracker/analyzer chain is
// held by at most one context thread at a time (§5.4, pool pseudocode).
struct StreamState {
    uint32_t index = 0;
    StreamSpec spec;
    LatestFrame latest;
    Tracker tracker;
    std::vector<std::unique_ptr<Analyzer>> analyzers;
    std::vector<std::string> analyzer_names;
    StreamMetrics metrics;

    // Holder exclusion (busy set) + remove handshake.
    std::mutex holder_mu;
    std::condition_variable holder_cv;
    bool busy = false, removing = false;

    // Pending control items applied by the holder thread before the next
    // frame (configure_analyzer semantics, §6.3).
    std::mutex ctl_mu;
    std::condition_variable ctl_cv;
    std::optional<float> pending_score;
    std::optional<PendingAnalyzerCfg> pending_cfg;  // single in-flight config
    uint64_t drop_baseline = 0;

    // Snapshot ring: last N host frames.
    int ring_cap = 2;
    std::deque<SnapshotRingEntry> ring;

    // Set once the first frame after (re)connect was produced.
    std::atomic<bool> got_first_frame{false};
};

class Runtime {
public:
    Runtime(std::unique_ptr<Backend> backend, RuntimeConfig cfg, Writer& writer);
    ~Runtime();

    // Starts context threads + the stats thread. Emits nothing; the owner
    // sends hello via emit_hello().
    bool start(std::string& err);
    // Idempotent; joins all threads, removes all streams.
    void stop();

    void emit_hello();
    void emit_stats();

    // Handles one decoded control line (§6.3). Returns true when a stop was
    // requested (caller replies, finishes within 1 s). Replies are pushed
    // onto the writer by these handlers.
    bool handle_line(const Json& line);

    // Optional hook for parity mode: called with every encoded frame record.
    std::function<void(const WireFrameRec&)> on_frame_rec;

    // Optional hook (standalone): called with every control-plane record
    // (reply/hello/stream_state/stats) before it is queued on the writer.
    std::function<void(const Json&)> on_reply_record;

    // Streams listed in the config file (parity mode convenience; shard
    // configs have none, §6.9 rule 4).
    std::vector<Json> config_streams;

    Backend& backend() { return *backend_; }
    const RuntimeConfig& config() const { return cfg_; }
    size_t stream_count();
    std::shared_ptr<StreamState> stream(uint32_t index);  // null if absent

private:
    friend class ContextPool;
    friend void serve_fd(int conn_fd, Runtime& rt);

    // Control op implementations. Each pushes the reply record.
    void op_add(const Json& line);
    void op_remove(const Json& line);
    void op_set_threshold(const Json& line);
    void op_configure_analyzer(const Json& line);
    void op_snapshot(const Json& line);
    void op_stop(const Json& line);
    void reply(const std::string& req, bool ok, const Json& applied,
               const std::string& error);

    void push_reply_record(const Json& j);
    void emit_stream_state(std::shared_ptr<StreamState> s, const std::string& state,
                           const std::string& error);
    std::shared_ptr<StreamState> add_stream_locked(const StreamSpec& spec,
                                                   const Json& analyzers,
                                                   std::string& err);
    void source_thread(std::shared_ptr<StreamState> s,
                       std::unique_ptr<FrameSource> src);
    void stats_thread();

    std::unique_ptr<Backend> backend_;
    RuntimeConfig cfg_;
    Writer& writer_;
    Caps caps_;
    int max_batch_ = 1;

    std::mutex streams_mu_;  // guards streams_ and all user-facing add/remove
    std::map<uint32_t, std::shared_ptr<StreamState>> streams_;
    std::vector<std::unique_ptr<InferenceContext>> contexts_;
    std::vector<std::pair<uint32_t, std::thread>> source_threads_;
    std::unique_ptr<ContextPool> pool_;
    std::thread stats_thread_;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> stop_requested_{false};
};

// Serves one connected fd: reads newline-delimited JSON control lines
// (single-line cap 2 MiB, §6.3), dispatches to rt.handle_line. Returns when
// the peer closes, a line exceeds the cap, or stop was requested.
void serve_fd(int conn_fd, Runtime& rt);

// Write a full buffer to a fd (handles partial writes; returns false on EOF).
bool write_all_fd(int fd, const uint8_t* data, size_t len);

}  // namespace vb
