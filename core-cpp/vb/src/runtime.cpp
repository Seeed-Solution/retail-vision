// Runtime: stream lifecycle, control ops (§6.3), hello/stats/stream_state
// records, source threads (reconnect per §5.4), parity hook (spec M1.7/M1.8).
#include <mutex>
#include <thread>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include "pool.h"
#include "snapshot.h"
#include "vb/letterbox.h"

namespace vb {

namespace {

// Bounded waits at the control boundary: how long the application of a
// control item may take before the reply reports failure, and how long a
// remove waits for the context pool to release the stream.
constexpr int kControlApplyTimeoutMs = 1000;
constexpr int kRemoveBusyTimeoutMs = 2000;

double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

std::string json_get_str(const Json& j, const char* key, const std::string& dflt = "") {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return dflt;
    if (!it->is_string()) return dflt;
    return it->get<std::string>();
}

// ---- Required-field accessors (B1) ----
// A control line is untrusted input (it may arrive over the MQTT control
// plane). Handlers validate their required fields with these instead of
// nlohmann's at(), whose out_of_range would escape to the fd server and
// terminate the process.

bool need_object(const Json& j, const char* key, const Json*& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_object()) {
        err = std::string("missing or invalid field: ") + key;
        return false;
    }
    out = &*it;
    return true;
}

bool need_u32(const Json& j, const char* key, uint32_t& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_number_integer()) {
        err = std::string("missing or invalid field: ") + key;
        return false;
    }
    int64_t v = it->get<int64_t>();
    if (v < 0 || v > static_cast<int64_t>(UINT32_MAX)) {
        err = std::string("out of range field: ") + key;
        return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

bool need_u64(const Json& j, const char* key, uint64_t& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_number_integer()) {
        err = std::string("missing or invalid field: ") + key;
        return false;
    }
    int64_t v = it->get<int64_t>();
    if (v < 0) {
        err = std::string("out of range field: ") + key;
        return false;
    }
    out = static_cast<uint64_t>(v);
    return true;
}

bool need_int(const Json& j, const char* key, int& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_number_integer()) {
        err = std::string("missing or invalid field: ") + key;
        return false;
    }
    out = it->get<int>();
    return true;
}

bool need_number(const Json& j, const char* key, double& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_number()) {
        err = std::string("missing or invalid field: ") + key;
        return false;
    }
    out = it->get<double>();
    return true;
}

bool need_string(const Json& j, const char* key, std::string& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_string()) {
        err = std::string("missing or invalid field: ") + key;
        return false;
    }
    out = it->get<std::string>();
    return true;
}

bool need_bool(const Json& j, const char* key, bool& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || !it->is_boolean()) {
        err = std::string("missing or invalid field: ") + key;
        return false;
    }
    out = it->get<bool>();
    return true;
}

// Reads an optional field: absent/null keeps the default, a present field of
// the wrong type is an error rather than a silent fallback.
bool opt_u64(const Json& j, const char* key, uint64_t dflt, uint64_t& out,
             std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        out = dflt;
        return true;
    }
    return need_u64(j, key, out, err);
}

bool opt_u32(const Json& j, const char* key, uint32_t dflt, uint32_t& out,
             std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        out = dflt;
        return true;
    }
    return need_u32(j, key, out, err);
}

bool opt_int(const Json& j, const char* key, int dflt, int& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        out = dflt;
        return true;
    }
    return need_int(j, key, out, err);
}

bool opt_bool(const Json& j, const char* key, bool dflt, bool& out, std::string& err) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        out = dflt;
        return true;
    }
    return need_bool(j, key, out, err);
}

bool has_so_suffix(const std::string& name) {
    return name.size() > 3 && name.compare(name.size() - 3, 3, ".so") == 0;
}

}  // namespace

// ---- RuntimeConfig ----

RuntimeConfig RuntimeConfig::from_json(const Json& j, std::string& err,
                                       bool allow_dev) {
    RuntimeConfig c;
    try {
        if (!dev_config_from_json(j, allow_dev, c.dev, err)) return c;
        const Json& backend = j.at("backend");
        c.backend_name = backend.at("name").get<std::string>();
        c.backend_json = json_dump(backend);
        auto read_int = [&](const char* key, int dflt) {
            auto it = j.find(key);
            if (it != j.end() && !it->is_null()) return it->get<int>();
            // also accept runtime.<key>
            auto rt = j.find("runtime");
            if (rt != j.end() && rt->is_object()) {
                auto it2 = rt->find(key);
                if (it2 != rt->end() && !it2->is_null()) return it2->get<int>();
            }
            return dflt;
        };
        c.contexts = std::max(1, read_int("contexts_per_worker", 1));
        auto read_dbl = [&](const char* key, double dflt) {
            auto it = j.find(key);
            if (it != j.end() && !it->is_number()) {
                if (it == j.end() || it->is_null()) return dflt;
            }
            if (it != j.end() && it->is_number()) return it->get<double>();
            auto rt = j.find("runtime");
            if (rt != j.end() && rt->is_object()) {
                auto it2 = rt->find(key);
                if (it2 != rt->end() && it2->is_number()) return it2->get<double>();
            }
            return dflt;
        };
        c.open_timeout_s = read_dbl("open_timeout_s", 5.0);
        c.reconnect_delay_s = std::max(0.0, read_dbl("reconnect_delay_s", 1.0));
        c.status_interval_s = std::max(0.2, read_dbl("status_interval_s", 1.0));
        auto tr = j.find("tracker");
        if (tr != j.end() && tr->is_object()) {
            // §6.6: tracker.enabled is the single key config.py validates
            // (§6.11 classify requires it false) and the runtime executes on;
            // the type rule matches config.py:285.
            auto en = tr->find("enabled");
            if (en != tr->end() && !en->is_null()) {
                if (!en->is_boolean()) {
                    err = "tracker.enabled: must be a boolean";
                    return c;
                }
                c.tracker_enabled = en->get<bool>();
            }
            auto rd = [&](const char* k, double d) {
                auto it = tr->find(k);
                return (it != tr->end() && it->is_number()) ? it->get<double>() : d;
            };
            auto ri = [&](const char* k, int d) {
                auto it = tr->find(k);
                return (it != tr->end() && it->is_number()) ? it->get<int>() : d;
            };
            c.tracker.iou_threshold = static_cast<float>(rd("iou_threshold", 0.2));
            c.tracker.dist_threshold = static_cast<float>(rd("dist_threshold", 0.15));
            c.tracker.max_misses = ri("max_misses", 15);
            c.tracker.max_lost_s = static_cast<float>(rd("max_lost_s", 0.0));
            c.tracker.min_hits = ri("min_hits", 1);
            c.tracker.class_aware = ri("class_aware", 1) != 0;
            std::string anchor = json_get_str(*tr, "anchor", "center");
            c.tracker.anchor = (anchor == "bottom_center") ? TrackerConfig::BottomCenter
                                                           : TrackerConfig::Center;
        }
        auto an = j.find("analyzers");
        if (an != j.end() && an->is_object()) {
            auto pl = an->find("plugins");
            if (pl != an->end() && pl->is_array())
                for (const auto& p : *pl) c.plugin_paths.push_back(p.get<std::string>());
        }
        auto ring = j.find("snapshot_ring");
        if (ring != j.end()) {
            if (ring->is_number())
                c.snapshot_ring = std::max(0, ring->get<int>());
            else if (ring->is_object()) {
                auto fr = ring->find("frames");
                if (fr != ring->end() && fr->is_number())
                    c.snapshot_ring = std::max(0, fr->get<int>());
                else if (fr != ring->end() && fr->is_string())
                    c.snapshot_ring = std::max(0, std::stoi(fr->get<std::string>()));
            } else if (ring->is_string()) {
                c.snapshot_ring = std::max(0, std::stoi(ring->get<std::string>()));
            }
        }
        return c;
    } catch (const std::exception& e) {
        err = std::string("runtime config: ") + e.what();
        return c;
    }
}

RuntimeConfig RuntimeConfig::load(const std::string& path, std::string& err,
                                  bool allow_dev) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open config: " + path;
        return RuntimeConfig();
    }
    std::stringstream ss;
    ss << f.rdbuf();
    try {
        return from_json(json_parse(ss.str()), err, allow_dev);
    } catch (const std::exception& e) {
        err = "bad config json: " + std::string(e.what());
        return RuntimeConfig();
    }
}

// ---- Runtime ----

Runtime::Runtime(std::unique_ptr<Backend> backend, RuntimeConfig cfg, Writer& writer)
    : backend_(std::move(backend)), cfg_(std::move(cfg)), writer_(writer) {
    caps_ = backend_->caps();
    max_batch_ = std::max(1, caps_.max_batch);
    cfg_.contexts = std::min(cfg_.contexts, std::max(1, caps_.max_contexts));
    // §6.12: VBT1 is sent only when dev.raw_tensors=true (already gated by
    // --dev at config load) and the backend decoder is "raw".
    if (cfg_.dev.raw_tensors) {
        std::string decoder_type;
        try {
            Json bj = json_parse(cfg_.backend_json);
            auto d = bj.find("decoder");
            if (d != bj.end() && d->is_object())
                decoder_type = d->value("type", std::string());
        } catch (...) {
        }
        dev_active_ = decoder_type == "raw";
    }
}

Runtime::~Runtime() { stop(); }

bool Runtime::load_configured_plugins(std::string& err) {
    for (const auto& path : cfg_.plugin_paths) {
        const vb_analyzer_api* api = nullptr;
        std::shared_ptr<void> handle = load_plugin_api(path, &api, err);
        if (!handle) return false;
        // §6.2: the plugin's own api.name is the id control lines select it
        // by. Two paths registering the same id would make that ambiguous.
        for (const auto& e : plugins_) {
            if (e.id == api->name) {
                err = "duplicate analyzer plugin id '" + std::string(api->name) +
                      "' (" + e.path + ", " + path + ")";
                return false;
            }
        }
        AnalyzerPlugin e;
        e.id = api->name;
        e.path = path;
        e.handle = std::move(handle);
        e.api = api;
        plugins_.push_back(std::move(e));
    }
    return true;
}

std::unique_ptr<Analyzer> Runtime::make_stream_analyzer(const std::string& name,
                                                       std::string& err) const {
    // C2: a request-supplied name that carries a path is rejected outright and
    // never reaches dlopen. Only ids declared in analyzers.plugins (protected
    // config, loaded at start) can select a plugin.
    if (name.find('/') != std::string::npos || has_so_suffix(name)) {
        err = "analyzer name must not be a path: " + name;
        return nullptr;
    }
    static const char kPluginPrefix[] = "plugin:";
    if (name.compare(0, sizeof(kPluginPrefix) - 1, kPluginPrefix) == 0) {
        std::string id = name.substr(sizeof(kPluginPrefix) - 1);
        for (const auto& p : plugins_) {
            if (p.id == id) return make_plugin_analyzer(p.handle, p.api);
        }
        err = "unknown plugin: " + id + " (not registered in analyzers.plugins)";
        return nullptr;
    }
    return create_analyzer(name, err);
}

bool Runtime::start(std::string& err) {
    if (!backend_) {
        err = "no backend";
        return false;
    }
    // §6.2: analyzers.plugins is the plugin whitelist; loading it here means a
    // plugin that fails to load (or an incomplete API table) aborts startup
    // instead of being skipped silently, and no add() can name an unknown path.
    if (!load_configured_plugins(err)) return false;
    for (int i = 0; i < cfg_.contexts; ++i) {
        auto ctx = backend_->create_context(i, err);
        if (!ctx) return false;
        contexts_.push_back(std::move(ctx));
    }
    pool_ = std::make_unique<ContextPool>(this);
    pool_->start();
    stats_thread_ = std::thread([this] { stats_thread(); });
    return true;
}

void Runtime::stop() {
    if (stopping_.exchange(true)) return;
    stop_requested_ = true;
    // B8: release the writer's queue waits before joining the context pool.
    // With the output peer gone, producers park in Writer::push_event once the
    // event queue is over the backpressure limit, and the pool join would wait
    // for them forever.
    writer_.request_stop();
    if (pool_) pool_->stop();
    // Stop source threads: mark removing, join, then drop the streams.
    std::vector<std::pair<uint32_t, std::thread>> threads;
    {
        std::lock_guard<std::mutex> lk(streams_mu_);
        threads = std::move(source_threads_);
        source_threads_.clear();
        for (auto& kv : streams_) {
            // atomic: no frame lock, so a long frame cannot delay shutdown
            kv.second->removing = true;
            kv.second->holder_cv.notify_all();
        }
    }
    for (auto& t : threads)
        if (t.second.joinable()) t.second.join();
    if (stats_thread_.joinable()) stats_thread_.join();
    std::lock_guard<std::mutex> lk(streams_mu_);
    streams_.clear();
}

void Runtime::push_reply_record(const Json& j) {
    if (on_reply_record) on_reply_record(j);
    std::vector<uint8_t> rec;
    wire_encode_json_record("VBC1", json_dump(j), rec);
    writer_.push_event(std::move(rec));
}

void Runtime::reply(const std::string& req, bool ok, const Json& applied,
                    const std::string& error) {
    Json j;
    j["op"] = "reply";
    j["req"] = req;
    j["ok"] = ok;
    if (ok) j["applied"] = applied;
    else j["error"] = error;
    push_reply_record(j);
}

void Runtime::emit_stream_state(std::shared_ptr<StreamState> s, const std::string& state,
                                const std::string& error) {
    std::string prev = s->metrics.state();
    s->metrics.record_state(state, error);
    Json j;
    j["op"] = "stream_state";
    j["stream_index"] = s->index;
    j["state"] = state;
    if (!error.empty()) j["error"] = error;
    push_reply_record(j);
    (void)prev;
}

void Runtime::emit_hello() {
    auto hw = backend_->model_hw();
    Json j;
    j["op"] = "hello";
    j["runtime_version"] = runtime_version();
    j["abi"] = 1;
    j["backend"] = backend_->name();
    Json caps;
    caps["max_contexts"] = caps_.max_contexts;
    caps["max_batch"] = caps_.max_batch;
    caps["keypoints"] = caps_.keypoints;
    caps["exclusive_device"] = caps_.exclusive_device;
    j["caps"] = caps;
    j["model_hw"] = Json::array({hw.first, hw.second});
    j["model_sha256"] = backend_->model_sha256();
    j["attr_names"] = Json::array();
    j["stage2_ready"] = false;
    j["pid"] = static_cast<int>(::getpid());
    push_reply_record(j);
}

void Runtime::emit_stats() {
    struct rusage ru;
    Json j;
    j["op"] = "stats";
    if (::getrusage(RUSAGE_SELF, &ru) == 0) {
        long rss_kb;
#if defined(__APPLE__)
        rss_kb = ru.ru_maxrss / 1024;  // macOS reports bytes
#else
        rss_kb = ru.ru_maxrss;         // Linux reports KiB
#endif
        j["rss_kb"] = rss_kb;
        j["cpu_s"] = static_cast<double>(ru.ru_utime.tv_sec) + ru.ru_utime.tv_usec / 1e6 +
                     static_cast<double>(ru.ru_stime.tv_sec) + ru.ru_stime.tv_usec / 1e6;
    } else {
        j["rss_kb"] = 0;
        j["cpu_s"] = 0.0;
    }
    Json streams = Json::array();
    std::lock_guard<std::mutex> lk(streams_mu_);
    for (auto& kv : streams_) streams.push_back(kv.second->metrics.to_json(kv.first));
    j["streams"] = streams;
    if (dev_active_) {  // §6.12: status records flag dev mode
        j["dev_mode"] = true;
        j["dev_tensor_oversize"] = dev_tensor_oversize_.load();
    }
    push_reply_record(j);
}

void Runtime::stats_thread() {
    double next = now_s() + cfg_.status_interval_s;
    for (;;) {
        if (stopping_) return;
        double now = now_s();
        if (now >= next) {
            emit_stats();
            next = now + cfg_.status_interval_s;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

bool Runtime::handle_line(const Json& line) {
    std::string op = json_get_str(line, "op");
    std::string req = json_get_str(line, "req");
    bool stop = false;
    try {
        if (op == "add") op_add(line);
        else if (op == "remove") op_remove(line);
        else if (op == "set_threshold") op_set_threshold(line);
        else if (op == "configure_analyzer") op_configure_analyzer(line);
        else if (op == "snapshot") op_snapshot(line);
        else if (op == "stop") op_stop(line);
        else reply(req, false, Json(), "unknown op: " + (op.empty() ? "?" : op));
        stop = (op == "stop");
    } catch (const std::exception& e) {
        // B1: a malformed control line fails its request. The handlers check
        // their required fields, this is the backstop for anything they miss;
        // letting the exception escape would terminate the whole runtime (the
        // control plane is reachable from MQTT).
        reply(req, false, Json(),
              std::string("bad ") + (op.empty() ? "command" : op) + ": " + e.what());
    }
    return stop;
}

void Runtime::op_add(const Json& line) {
    std::string req = json_get_str(line, "req");
    std::string err;
    const Json* sj = nullptr;
    if (!need_object(line, "stream", sj, err)) {
        reply(req, false, Json(), err);
        return;
    }
    StreamSpec spec;
    std::string url;
    if (!need_u32(*sj, "index", spec.index, err) ||
        !need_string(*sj, "url", url, err)) {
        reply(req, false, Json(), err);
        return;
    }
    spec.url = url;
    spec.id = json_get_str(*sj, "id");
    spec.name = json_get_str(*sj, "name");
    spec.transport = json_get_str(*sj, "transport", "tcp");
    auto sth = sj->find("score_threshold");
    if (sth != sj->end() && !sth->is_null()) {
        if (!sth->is_number()) {
            reply(req, false, Json(), "invalid field: score_threshold");
            return;
        }
        spec.score_threshold = sth->get<float>();
    }
    auto opts = sj->find("options");
    spec.options_json = (opts != sj->end() && !opts->is_null()) ? json_dump(*opts) : "{}";

    static const Json kNoAnalyzers = Json::array();
    const Json& analyzers = line.contains("analyzers") ? line.at("analyzers") : kNoAnalyzers;
    if (!analyzers.is_array()) {
        reply(req, false, Json(), "invalid field: analyzers");
        return;
    }
    std::shared_ptr<StreamState> s = add_stream_locked(spec, analyzers, err);
    if (!s) {
        reply(req, false, Json(), err);
        return;
    }
    Json applied;
    applied["stream_index"] = spec.index;
    reply(req, true, applied, "");
}

std::shared_ptr<StreamState> Runtime::add_stream_locked(const StreamSpec& spec,
                                                        const Json& analyzers,
                                                        std::string& err) {
    {
        std::lock_guard<std::mutex> lk(streams_mu_);
        if (streams_.count(spec.index)) {
            err = "stream index already in use: " + std::to_string(spec.index);
            return nullptr;
        }
        // §6.12: dev mode is limited to dev.max_streams (validated == 1).
        if (cfg_.dev.raw_tensors &&
            streams_.size() >= static_cast<size_t>(cfg_.dev.max_streams)) {
            err = "dev mode allows " + std::to_string(cfg_.dev.max_streams) + " stream";
            return nullptr;
        }
    }
    auto src = backend_->create_source(spec, err);
    if (!src) return nullptr;

    auto s = std::make_shared<StreamState>();
    s->index = spec.index;
    s->spec = spec;
    s->track_enabled = cfg_.tracker_enabled;
    s->tracker = Tracker(cfg_.tracker);
    s->ring_cap = cfg_.snapshot_ring;
    s->metrics.record_state("starting", "");
    s->metrics.set_decode_path(src->decode_path());
    s->dev_limiter.set_max_fps(cfg_.dev.max_fps);

    // Analyzers: built-in names, or "plugin:<id>" for a plugin registered from
    // analyzers.plugins. The name is what configure_analyzer matches against,
    // so it is stored as given.
    for (const auto& aj : analyzers) {
        if (!aj.is_object()) {
            err = "analyzer entry must be an object";
            return nullptr;
        }
        std::string name;
        if (!need_string(aj, "name", name, err)) return nullptr;
        std::string cfg_json = "{}";
        auto cj = aj.find("config");
        if (cj != aj.end() && !cj->is_null()) {
            if (!cj->is_object()) {
                err = "analyzer config must be an object";
                return nullptr;
            }
            cfg_json = json_dump(*cj);
        }
        std::unique_ptr<Analyzer> a = make_stream_analyzer(name, err);
        if (!a) return nullptr;
        // §6.2: with tracker.enabled=false detections arrive as track_id=0
        // single-frame tracks, so an analyzer that needs a track lifecycle
        // cannot be enabled on this runtime.
        if (!s->track_enabled && a->needs_tracks()) {
            err = name + " requires tracks (tracker.enabled=false)";
            return nullptr;
        }
        // §6.2.5: pose_angle-class analyzers need a model with keypoints
        // (checked after configure: min_keypoints depends on the joints).
        if (!a->configure(cfg_json, err)) return nullptr;
        if (a->min_keypoints() > 0 &&
            static_cast<uint32_t>(caps_.keypoints) < a->min_keypoints()) {
            err = name + " requires keypoints";
            return nullptr;
        }
        s->analyzer_names.push_back(name);
        s->analyzers.push_back(std::move(a));
    }

    emit_stream_state(s, "starting", "");

    // Open the source (add-phase deadline: open_timeout_s), §6.3/§6.5.2.
    double deadline = now_s() + cfg_.open_timeout_s;
    for (;;) {
        if (src->open(err)) break;
        if (stopping_ || now_s() >= deadline) {
            // §6.3: an add-phase open timeout is reported only through the
            // add reply (ok:false); no stream_state "error" record.
            return nullptr;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    {
        std::lock_guard<std::mutex> lk(streams_mu_);
        if (stopping_) {
            err = "runtime stopping";
            return nullptr;
        }
        streams_[spec.index] = s;
        source_threads_.emplace_back(spec.index,
                                     std::thread([this, s, src = std::move(src)]() mutable {
                                         source_thread(s, std::move(src));
                                     }));
    }
    return s;
}

void Runtime::source_thread(std::shared_ptr<StreamState> s,
                            std::unique_ptr<FrameSource> src) {
    FrameBuf f;
    for (;;) {
        if (stopping_) break;
        if (s->removing) break;
        int r = src->read(f, 100);
        if (r == 1) {
            f.stream_index = s->index;
            if (!s->got_first_frame.exchange(true)) {
                s->metrics.record_state("running", "");
                Json j;
                j["op"] = "stream_state";
                j["stream_index"] = s->index;
                j["state"] = "running";
                push_reply_record(j);
            }
            s->latest.put(std::move(f));
        } else if (r == 0) {
            continue;
        } else {
            // Lost stream: reconnecting, retry after reconnect_delay_s.
            s->got_first_frame = false;
            s->metrics.record_state("reconnecting", "read failed");
            Json j;
            j["op"] = "stream_state";
            j["stream_index"] = s->index;
            j["state"] = "reconnecting";
            j["error"] = "read failed";
            push_reply_record(j);
            src->close();
            double until = now_s() + cfg_.reconnect_delay_s;
            while (!stopping_ && now_s() < until)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            std::string err;
            while (!stopping_) {
                if (s->removing) return;
                if (src->open(err)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
    }
}

void Runtime::op_remove(const Json& line) {
    std::string req = json_get_str(line, "req");
    uint32_t idx = 0;
    std::string err;
    if (!need_u32(line, "stream_index", idx, err)) {
        reply(req, false, Json(), err);
        return;
    }
    std::shared_ptr<StreamState> s;
    {
        std::lock_guard<std::mutex> lk(streams_mu_);
        auto it = streams_.find(idx);
        if (it == streams_.end()) {
            reply(req, false, Json(), "no stream " + std::to_string(idx));
            return;
        }
        s = it->second;
    }
    // 1) Stop new work on this stream. `removing` is atomic, so this does not
    //    wait for the frame lock a context thread holds for a whole frame.
    s->removing = true;
    s->holder_cv.notify_all();
    // 2) Wait for the in-flight frame, if any, to finish. The pool skips
    //    removing streams both when claiming and before each batch item, so
    //    this is the last frame that can touch the stream. Reporting success
    //    after a timeout would be wrong twice over: the peer would see frames
    //    and events after the "stopped" record, and a later add() reusing the
    //    index would have them attributed to the new stream.
    //
    //    try_lock rather than wait_for: the holder keeps holder_mu for the
    //    whole frame, so blocking on the lock would silently wait for the
    //    frame and then report success — the timeout would never be observed.
    const double deadline = now_s() + kRemoveBusyTimeoutMs / 1000.0;
    bool released = false;
    for (;;) {
        {
            std::unique_lock<std::mutex> hlk(s->holder_mu, std::try_to_lock);
            if (hlk.owns_lock() && !s->busy) {
                released = true;
                break;
            }
        }
        if (now_s() >= deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!released) {
        // The reply says failed, so the stream must still be there and usable.
        s->removing = false;
        reply(req, false, Json(),
              "remove timeout: stream " + std::to_string(idx) + " still busy");
        return;
    }
    // 3) Detach the stream and join its source thread.
    {
        std::lock_guard<std::mutex> lk(streams_mu_);
        auto it = streams_.find(idx);
        if (it == streams_.end() || it->second != s) {
            reply(req, false, Json(), "no stream " + std::to_string(idx));
            return;
        }
        streams_.erase(it);
        for (auto tt = source_threads_.begin(); tt != source_threads_.end(); ++tt) {
            if (tt->first == idx) {
                if (tt->second.joinable()) tt->second.join();
                source_threads_.erase(tt);
                break;
            }
        }
    }
    emit_stream_state(s, "stopped", "");
    Json applied;
    applied["stream_index"] = idx;
    reply(req, true, applied, "");
}

void Runtime::op_set_threshold(const Json& line) {
    std::string req = json_get_str(line, "req");
    uint32_t idx = 0;
    double value = 0.0;
    std::string err;
    if (!need_u32(line, "stream_index", idx, err) ||
        !need_number(line, "value", value, err)) {
        reply(req, false, Json(), err);
        return;
    }
    auto s = stream(idx);
    if (!s) {
        reply(req, false, Json(), "no stream " + std::to_string(idx));
        return;
    }
    float applied_value = 0.0f;
    bool applied = false;
    {
        std::unique_lock<std::mutex> lk(s->ctl_mu);
        s->pending_score = static_cast<float>(value);
        // Applied by the holder thread before the next frame (§6.3). The
        // result and the value are read under ctl_mu, the same lock the holder
        // writes spec.score_threshold with, so the reply cannot carry a stale
        // or racy value.
        applied = s->ctl_cv.wait_for(lk, std::chrono::milliseconds(kControlApplyTimeoutMs),
                                     [&] { return !s->pending_score.has_value(); });
        if (applied) {
            applied_value = s->spec.score_threshold;
        } else {
            // No frame within the window (idle or reconnecting stream): drop
            // the pending value so the failed reply matches the real state.
            s->pending_score.reset();
        }
    }
    if (!applied) {
        reply(req, false, Json(), "stream has no frames; threshold not applied");
        return;
    }
    Json applied_json;
    applied_json["stream_index"] = s->index;
    applied_json["value"] = applied_value;
    reply(req, true, applied_json, "");
}

void Runtime::op_configure_analyzer(const Json& line) {
    std::string req = json_get_str(line, "req");
    uint32_t idx = 0;
    std::string name;
    std::string err;
    if (!need_u32(line, "stream_index", idx, err) ||
        !need_string(line, "name", name, err)) {
        reply(req, false, Json(), err);
        return;
    }
    auto s = stream(idx);
    if (!s) {
        reply(req, false, Json(), "no stream " + std::to_string(idx));
        return;
    }
    std::string cfg_json = "{}";
    auto cj = line.find("config");
    if (cj != line.end() && !cj->is_null()) {
        if (!cj->is_object()) {
            reply(req, false, Json(), "invalid field: config");
            return;
        }
        cfg_json = json_dump(*cj);
    }
    {
        std::unique_lock<std::mutex> lk(s->ctl_mu);
        if (s->pending_cfg.has_value() && !s->pending_cfg->done) {
            // Wait for the in-flight one first.
            s->ctl_cv.wait_for(lk, std::chrono::seconds(1),
                               [&] { return !s->pending_cfg.has_value() || s->pending_cfg->done; });
        }
        PendingAnalyzerCfg c;
        c.name = name;
        c.json = cfg_json;
        s->pending_cfg = c;
        bool done = s->ctl_cv.wait_for(lk, std::chrono::seconds(1),
                                       [&] { return s->pending_cfg->done || stopping_; });
        bool ok = false;
        std::string err = "stream has no frames; configure not applied";
        if (done && s->pending_cfg.has_value() && s->pending_cfg->done) {
            ok = s->pending_cfg->ok;
            err = s->pending_cfg->err;
        } else {
            // No frame for 1 s: apply directly under the busy-set protection.
            lk.unlock();
            {
                std::lock_guard<std::mutex> hlk(s->holder_mu);
                Analyzer* a = nullptr;
                for (size_t i = 0; i < s->analyzers.size(); ++i)
                    if (s->analyzer_names[i] == name) a = s->analyzers[i].get();
                if (!a) {
                    ok = false;
                    err = "no analyzer " + name + " on stream " + std::to_string(s->index);
                } else {
                    ok = a->configure(cfg_json, err);
                }
            }
            lk.lock();
        }
        s->pending_cfg.reset();
        if (ok) {
            Json applied;
            applied["stream_index"] = s->index;
            applied["name"] = name;
            reply(req, true, applied, "");
        } else {
            reply(req, false, Json(), err);
        }
    }
}

void Runtime::op_snapshot(const Json& line) {
    std::string req = json_get_str(line, "req");
    uint32_t idx = 0, track_id = 0;
    uint64_t seq = 0;
    bool crop = true;
    int max_side = 640;
    std::string err;
    if (!need_u32(line, "stream_index", idx, err) ||
        !opt_u64(line, "seq", 0, seq, err) ||
        !opt_u32(line, "track_id", 0, track_id, err) ||
        !opt_bool(line, "crop", true, crop, err) ||
        !opt_int(line, "max_side", 640, max_side, err)) {
        reply(req, false, Json(), err);
        return;
    }
    auto s = stream(idx);
    if (!s) {
        reply(req, false, Json(), "no stream " + std::to_string(idx));
        return;
    }

    std::vector<uint8_t> jpeg;
    int w = 0, h = 0;
    bool ok = false;
    uint64_t used_seq = 0;
    {
        std::lock_guard<std::mutex> hlk(s->holder_mu);
        const SnapshotRingEntry* pick = nullptr;
        if (s->ring.empty()) {
            err = "no frame available";
        } else if (seq == 0) {
            pick = &s->ring.back();
        } else {
            for (const auto& e : s->ring)
                if (e.seq == seq) pick = &e;
            if (!pick) pick = &s->ring.back();  // fall back to newest
        }
        if (pick) {
            used_seq = pick->seq;
            ok = snapshot_encode_jpeg(*pick, track_id, crop, max_side, jpeg, w, h, err);
        }
    }
    if (!ok) {
        reply(req, false, Json(), err.empty() ? "snapshot unsupported" : err);
        return;
    }
    Json meta;
    meta["req"] = req;
    meta["stream_index"] = s->index;
    meta["seq"] = used_seq;
    meta["track_id"] = track_id;
    meta["w"] = w;
    meta["h"] = h;
    meta["mime"] = "image/jpeg";
    std::vector<uint8_t> rec;
    wire_encode_vbs1(json_dump(meta), jpeg.data(), jpeg.size(), rec);
    writer_.push_event(std::move(rec));
}

void Runtime::op_stop(const Json& line) {
    std::string req = json_get_str(line, "req");
    Json applied;
    applied["stopped"] = true;
    reply(req, true, applied, "");
    stop_requested_ = true;
}

size_t Runtime::stream_count() {
    std::lock_guard<std::mutex> lk(streams_mu_);
    return streams_.size();
}

void Runtime::maybe_send_dev_tensors(StreamState& s, const FrameBuf& f,
                                     const DetectionResult& res,
                                     InferenceContext* ctx) {
    if (!dev_active_) return;
    if (!s.dev_limiter.try_send(now_s())) return;  // dev.max_fps per stream
    auto* raw = dynamic_cast<RawTensorSource*>(ctx);
    if (!raw) return;  // backend does not expose raw outputs
    DevTensorFrame tf;
    tf.stream_index = s.index;
    tf.seq = f.seq;
    tf.wall_ms = f.wall_ms;
    tf.src_w = f.w;
    tf.src_h = f.h;
    tf.model_w = res.geom.model_w;
    tf.model_h = res.geom.model_h;
    tf.scale = res.geom.scale;
    tf.pad_x = res.geom.pad_x;
    tf.pad_y = res.geom.pad_y;
    tf.align = static_cast<uint8_t>(res.geom.align);
    if (!raw->last_raw_tensors(tf.tensors)) return;
    std::vector<uint8_t> rec;
    std::string err;
    if (!wire_encode_vbt1(tf, rec, err)) {  // > 16 MiB: drop + count
        ++dev_tensor_oversize_;
        return;
    }
    writer_.push_event(std::move(rec));  // dev records are never dropped
}

std::shared_ptr<StreamState> Runtime::stream(uint32_t index) {
    std::lock_guard<std::mutex> lk(streams_mu_);
    auto it = streams_.find(index);
    return it == streams_.end() ? nullptr : it->second;
}

// ---- fd serving ----

bool write_all_fd(int fd, const uint8_t* data, size_t len,
                  const std::function<bool()>& cancel, int deadline_ms) {
    // The send timeout, not MSG_DONTWAIT, is what makes this cancellable: on
    // macOS a unix-domain send() whose peer has stopped reading blocks even
    // with MSG_DONTWAIT set, whereas SO_SNDTIMEO returns EAGAIN after the
    // timeout. The timeout bounds a single send() call, so the loop gets back
    // to the cancel predicate and the deadline every 50 ms. ENOTSOCK (a
    // non-socket fd) is left alone; its send() then fails on its own.
    constexpr int kSendTimeoutMs = 50;
    struct timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = kSendTimeoutMs * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    const double deadline = deadline_ms > 0 ? now_s() + deadline_ms / 1000.0 : 0.0;
    size_t off = 0;
    while (off < len) {
        if (cancel && cancel()) return false;
        if (deadline > 0.0 && now_s() >= deadline) return false;
        ssize_t n = ::send(fd, data + off, len - off, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            // Timed out mid-write: re-check cancel/deadline and resume.
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return false;
        }
        if (n == 0) return false;
        off += static_cast<size_t>(n);
    }
    return true;
}

void serve_fd(int conn_fd, Runtime& rt) {
    int flag = 1;
#if defined(SO_NOSIGPIPE)
    ::setsockopt(conn_fd, SOL_SOCKET, SO_NOSIGPIPE, &flag, sizeof(flag));
#endif
    std::string buf;
    buf.reserve(4096);
    char chunk[4096];
    const size_t kMaxLine = 2u * 1024 * 1024;  // §6.3 single-line cap
    for (;;) {
        ssize_t n = ::recv(conn_fd, chunk, sizeof(chunk), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;  // EOF
        buf.append(chunk, static_cast<size_t>(n));
        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line.size() > kMaxLine) return;  // close, record error via stderr
            Json j;
            try {
                j = json_parse(line);
            } catch (const std::exception& e) {
                Json r;
                r["op"] = "reply";
                r["req"] = "";
                r["ok"] = false;
                r["error"] = std::string("bad json: ") + e.what();
                std::vector<uint8_t> rec;
                wire_encode_json_record("VBC1", json_dump(r), rec);
                rt.writer_.push_event(std::move(rec));
                continue;
            }
            if (rt.handle_line(j)) return;  // stop: exit within 1 s
        }
        if (buf.size() > kMaxLine) return;
    }
}

}  // namespace vb
