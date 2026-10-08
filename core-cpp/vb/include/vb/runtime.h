// Runtime core: LatestFrame, Writer, stream metrics, ContextPool, Runtime
// (control ops, stats, stream_state) and the fd control-line server
// (spec BASE-1 §5.4 native thread model, §6.3 control protocol, M1.7/M1.8).
#pragma once

#include <cstddef>
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

#include "dev_tensor.h"
#include "vb/analyzer.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/tracker.h"
#include "vb/vb_analyzer_abi.h"
#include "vb/wire.h"
#include "vb/post.h"
#include "vb/rate_crop.h"
#include "stage2.h"

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
    // B8: how long stop() lets the writer thread flush queued records before
    // the flush deadline expires and a sink blocked on a peer that stopped
    // reading abandons the record. Bounds how long stop() can block.
    int stop_flush_ms = 500;
};

class Writer {
public:
    explicit Writer(WriterConfig cfg = WriterConfig());
    // Stops and joins if the owner forgot; a joinable std::thread reaching the
    // destructor would terminate the process (B4).
    ~Writer();

    using Sink = std::function<void(const uint8_t* data, size_t len)>;

    void start(Sink sink);
    // Flushes queued records (sink called on the caller thread) and joins.
    void stop();

    // B8: releases everything that stop() would release, without joining.
    // Producers parked in push_event() (event queue over the backpressure
    // limit) return, and the sink's flush deadline starts. Runtime::stop()
    // calls this before joining the context pool, which would otherwise wait
    // on producers blocked behind a stalled sink.
    void request_stop();

    // B8: true once request_stop() has been called and the flush window has
    // elapsed. A sink whose peer keeps the connection open but stops reading
    // polls this between write slices and abandons the record instead of
    // blocking the writer thread (and therefore stop()'s join) forever.
    bool flush_expired() const;

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
    // Steady-clock ms at which the stop-time flush gives up; 0 = not stopping.
    std::atomic<uint64_t> stop_deadline_ms_{0};
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
    void add_text_vote_dedup(uint64_t n);
    void set_stage2_backend(const std::string& backend, bool fallback);
    void record_stage2(float ms, bool failed);

    std::string state() const;
    // One entry of the stats op "streams" array (mutex-guarded snapshot).
    Json to_json(uint32_t stream_index) const;
    // Convenience counters for tests.
    uint64_t processed() const;
    double last_frame_wall_ms() const;

private:
    mutable std::mutex mu_;
    std::string state_ = "starting", error_, decode_;
    uint64_t processed_ = 0, dropped_ = 0, rate_skipped_ = 0, text_vote_dedup_ = 0;
    std::deque<float> inf_ms_, qd_ms_;
    std::string stage2_backend_;
    bool fallback_active_ = false;
    uint64_t stage2_errors_ = 0;
    std::deque<float> stage2_ms_;
    // Frame wall-clock stamps of the last second. Pruned against the current
    // time by to_json() as well as by record_frame(): a stream that stopped
    // producing frames must not keep reporting its last FPS (the window has to
    // reflect "now", not "the last frame").
    mutable std::deque<double> frame_walls_;
    double last_wall_ = 0;
};

// ---- Runtime config (§6.9 rule 4: no stream list in this file) ----
struct RuntimeConfig {
    std::string backend_name = "synthetic";
    std::string backend_json = "{}";
    std::string stage2_json;
    int contexts = 1;
    double open_timeout_s = 5.0;
    double reconnect_delay_s = 1.0;
    double status_interval_s = 1.0;
    TrackerConfig tracker;
    // §6.6 `tracker.enabled` (default true). When false the runtime builds no
    // per-stream tracking: every detection is passed to the analyzer chain
    // with track_id 0 and an analyzer whose needs_tracks() is true is rejected
    // at add(). Read from the same key config.py validates (§6.11 classify
    // requires it to be false), so validation and execution agree (A4).
    bool tracker_enabled = true;
    std::vector<std::string> plugin_paths;  // preloaded analyzer plugins
    int snapshot_ring = 2;
    DevConfig dev;                          // §6.12 dev-mode tensor passthrough

    // allow_dev: config dev.raw_tensors=true is accepted only when the
    // process was started with --dev (§6.12).
    static RuntimeConfig load(const std::string& path, std::string& err,
                              bool allow_dev = false);
    static RuntimeConfig from_json(const Json& j, std::string& err,
                                   bool allow_dev = false);
};

// ---- Analyzer plugins (§6.2) ----
// A configured plugin .so, loaded and validated once at Runtime::start().
// `id` is the plugin's own api.name: the only handle a control line may use
// to select it. The path itself never comes from a request (C2).
struct AnalyzerPlugin {
    std::string id;
    std::string path;
    std::shared_ptr<void> handle;           // dlopen handle; dlclose on last release
    const vb_analyzer_api* api = nullptr;
};

// plugin_loader.cpp (vb_algo): dlopen `so_path` and validate the API table.
// On success returns the handle (kept open while any reference lives) and sets
// *api; on failure returns nullptr with err set. Initialisation code in the
// library runs inside dlopen, so callers must pass only protected config paths.
std::shared_ptr<void> load_plugin_api(const std::string& so_path,
                                      const vb_analyzer_api** api,
                                      std::string& err);

// plugin_loader.cpp: validate the mandatory callbacks and attribute metadata
// of a plugin API table. A table that passes is safe to call into; one that
// fails must never be used (create/destroy/on_frame/on_track_removed null).
bool validate_plugin_api(const vb_analyzer_api* api, std::string& err);

// plugin_loader.cpp: wrap an already validated (handle, api) pair as an
// Analyzer instance for one stream (create() runs on configure()).
std::unique_ptr<Analyzer> make_plugin_analyzer(std::shared_ptr<void> handle,
                                               const vb_analyzer_api* api);

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
    // §6.6 tracker.enabled for this stream. When false `tracker` is never
    // updated: detections are wrapped as track_id=0, hits=1, misses=0 tracks.
    bool track_enabled = true;
    std::vector<std::unique_ptr<Analyzer>> analyzers;
    std::vector<std::string> analyzer_names;
    std::unique_ptr<Stage2Context> stage2;
    Stage2Spec stage2_spec;
    std::string stage2_backend, stage2_fallback_reason;
    std::vector<std::string> stage2_charset;
    CtcLayout stage2_layout = CtcLayout::CT;
    Stage2FilterConfig stage2_filter;
    std::map<uint32_t, Stage2TrackState> stage2_tracks;
    StreamMetrics metrics;
    RateLimiter rate_limiter;

    // Holder exclusion (busy set) + remove handshake.
    std::mutex holder_mu;
    std::condition_variable holder_cv;
    bool busy = false;  // guarded by holder_mu
    // Removal handshake: atomic so remove() can stop new claims without
    // waiting for the frame lock, which a context thread holds for a whole
    // frame (a 6 s inference would otherwise make the removal wait for it and
    // report success for a stream it never stopped).
    std::atomic<bool> removing{false};

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

    // Dev-mode VBT1 rate limiter (§6.12 dev.max_fps).
    DevRateLimiter dev_limiter;
};

struct AddRequest {
    std::string req;
    StreamSpec spec;
    Json analyzers;
    std::atomic<bool> cancelled{false};
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
    void op_infer_image(const Json& line);
    void op_stop(const Json& line);
    void reply(const std::string& req, bool ok, const Json& applied,
               const std::string& error);

    void push_reply_record(const Json& j);
    void emit_stream_state(std::shared_ptr<StreamState> s, const std::string& state,
                           const std::string& error);
    std::shared_ptr<StreamState> add_stream_locked(const StreamSpec& spec,
                                                   const Json& analyzers,
                                                   const std::shared_ptr<AddRequest>& request,
                                                   std::string& err);
    void add_worker();
    // C2: resolves a control-plane analyzer name to an analyzer. Accepts only
    // built-in names and "plugin:<id>" ids registered from analyzers.plugins;
    // a name that carries a path is rejected, never dlopen'd.
    std::unique_ptr<Analyzer> make_stream_analyzer(const std::string& name,
                                                   std::string& err) const;
    // Loads and validates every analyzers.plugins path (§6.2: a plugin that
    // fails to load aborts startup rather than being skipped silently).
    bool load_configured_plugins(std::string& err);
    // §6.12: when dev mode is active (dev.raw_tensors + backend.decoder.type
    // == "raw"), rate-limit and push a VBT1 record for this frame.
    void maybe_send_dev_tensors(StreamState& s, const FrameBuf& f,
                                const DetectionResult& res, InferenceContext* ctx);
    void source_thread(std::shared_ptr<StreamState> s,
                       std::unique_ptr<FrameSource> src,
                       std::optional<FrameBuf> first = std::nullopt);
    void stats_thread();
    void image_thread();
    bool configure_stage2(StreamState& s, const Json& config, std::string& err, bool stream = true);

    std::unique_ptr<Backend> backend_;
    RuntimeConfig cfg_;
    Writer& writer_;
    Caps caps_;
    int max_batch_ = 1;

    std::mutex streams_mu_;  // guards streams_ and all user-facing add/remove
    std::map<uint32_t, std::shared_ptr<StreamState>> streams_;
    std::map<uint32_t, std::shared_ptr<AddRequest>> opening_;
    std::mutex add_mu_;
    std::condition_variable add_cv_;
    std::deque<std::shared_ptr<AddRequest>> add_queue_;
    std::thread add_thread_;
    bool add_stopping_ = false;
    size_t add_inflight_ = 0;
    std::vector<std::unique_ptr<InferenceContext>> contexts_;
    std::vector<std::pair<uint32_t, std::thread>> source_threads_;
    std::unique_ptr<ContextPool> pool_;
    std::thread stats_thread_;
    std::thread image_thread_;
    std::mutex image_mu_;
    std::condition_variable image_cv_;
    std::deque<Json> image_queue_;
    static constexpr size_t kImageQueueLimit = 16;
    bool image_stopping_ = false;
    std::unique_ptr<StreamState> image_stage2_;
    std::atomic<bool> image_ready_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<bool> stop_requested_{false};
    // Plugin whitelist: built once from the protected config, read-only after.
    std::vector<AnalyzerPlugin> plugins_;

    bool dev_active_ = false;  // dev.raw_tensors && decoder.type == "raw"
    std::atomic<uint64_t> dev_tensor_oversize_{0};
};

// Serves one connected fd: reads newline-delimited JSON control lines
// (single-line cap 2 MiB, §6.3), dispatches to rt.handle_line. Returns when
// the peer closes, a line exceeds the cap, or stop was requested.
void serve_fd(int conn_fd, Runtime& rt);

// Writes a full buffer to a fd (handles partial writes; returns false on EOF).
//
// B8: the write is a bounded, cancellable wait rather than a blocking send().
// The buffer is written with poll() slices of at most 50 ms; `cancel` is
// polled between slices and a true result abandons the write (returns false),
// and `deadline_ms` > 0 abandons it once the whole call has taken that long.
// A peer that keeps the connection open but stops reading therefore stalls
// neither the writer thread nor stop()'s join.
bool write_all_fd(int fd, const uint8_t* data, size_t len,
                  const std::function<bool()>& cancel = {},
                  int deadline_ms = 0);

}  // namespace vb
