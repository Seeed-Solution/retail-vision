// Runtime acceptance test for the BASE-1 review-fix batch (B1, A4, B4, B8, C2
// and the pool/stats defects). Deterministic: every wait is bounded and every
// assertion is on a value the test drives itself (no sleep-until-lucky).
//
//   usage: test_runtime <fixtures-dir>
//
// Sections:
//   1  B1  a control line with missing fields fails its request, no exit
//   2  A4  tracker.enabled is parsed, honoured (track_id 0), and gates
//          needs_tracks() analyzers
//   3  C2  analyzer names can never carry a load path
//   4      plugin API table validation (null mandatory callbacks)
//   5      plugin load failure reports one usable message (no null string)
//   6      configured plugin end-to-end + attribute interleaving (needs cc)
//   7      attribute interleaving unit
//   8      remove semantics: no frame after the reply, timeout != success
//   9      set_threshold: applied value on success, failure when not applied
//   10 B8  cancellable/bounded writes and stop() with a stalled peer
//   11     stats FPS window follows the query time
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "check.h"
#include "pool.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/runtime.h"

using namespace vb;

namespace {

// ---- capture sink: keeps every wire record the runtime emits --------------

struct Rec {
    std::string magic;
    std::vector<uint8_t> bytes;  // whole record (header included)
    uint32_t stream_index = 0;
};

struct Capture {
    std::mutex mu;
    std::vector<Rec> recs;

    void install(Writer& w) {
        w.start([this](const uint8_t* data, size_t /*len*/) { add(data); });
    }
    void add(const uint8_t* data) {
        uint32_t body_len;
        std::memcpy(&body_len, data + 4, 4);
        Rec r;
        r.magic.assign(data, data + 4);
        r.bytes.assign(data, data + 8 + body_len);
        if (r.magic == "VBR1" || r.magic == "VBE1")
            std::memcpy(&r.stream_index, data + 8, 4);
        std::lock_guard<std::mutex> lk(mu);
        recs.push_back(std::move(r));
    }
    size_t count(const char* magic) {
        std::lock_guard<std::mutex> lk(mu);
        size_t n = 0;
        for (auto& r : recs)
            if (r.magic == magic) ++n;
        return n;
    }
    // Control records (VBC1) with the given op.
    std::vector<Json> control(const char* op) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<Json> out;
        for (auto& r : recs) {
            if (r.magic != "VBC1") continue;
            Json j = json_parse(std::string(r.bytes.begin() + 8, r.bytes.end()));
            if (j.value("op", std::string()) == op) out.push_back(j);
        }
        return out;
    }
    Json reply_for(const std::string& req) {
        for (auto& j : control("reply"))
            if (j.value("req", std::string()) == req) return j;
        return Json();
    }
    size_t frames(uint32_t stream) {
        std::lock_guard<std::mutex> lk(mu);
        size_t n = 0;
        for (auto& r : recs) {
            if (r.magic != "VBR1" || r.stream_index != stream) continue;
            ++n;
        }
        return n;
    }
    // Last VBR1 record of a stream, or nullptr.
    bool last_frame(uint32_t stream, std::vector<uint8_t>& out) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = recs.rbegin(); it != recs.rend(); ++it) {
            if (it->magic == "VBR1" && it->stream_index == stream) {
                out = it->bytes;
                return true;
            }
        }
        return false;
    }
    bool has_control(const char* op, const char* key, const char* value) {
        for (auto& j : control(op)) {
            if (j.dump().find(std::string("\"") + key + "\":\"" + value + "\"") !=
                std::string::npos)
                return true;
        }
        return false;
    }
    // Record order: nothing of `stream` may follow its "stopped" record.
    bool stopped_after_last_data(uint32_t stream) {
        std::lock_guard<std::mutex> lk(mu);
        size_t stopped_at = size_t(-1), last_data = 0;
        for (size_t i = 0; i < recs.size(); ++i) {
            if ((recs[i].magic == "VBR1" || recs[i].magic == "VBE1") &&
                recs[i].stream_index == stream) {
                last_data = i;
            } else if (recs[i].magic == "VBC1") {
                Json j = json_parse(
                    std::string(recs[i].bytes.begin() + 8, recs[i].bytes.end()));
                if (j.value("op", std::string()) == "stream_state" &&
                    j.value("stream_index", -1) == static_cast<int>(stream) &&
                    j.value("state", std::string()) == "stopped") {
                    stopped_at = i;
                }
            }
        }
        return stopped_at != size_t(-1) && last_data < stopped_at;
    }
};

// ---- VBR1 body decoding (spec §6.3) --------------------------------------
// `body` starts at the record's byte 8.

struct FrameView {
    uint32_t stream_index = 0, n_det = 0, kpt_per_det = 0, attr_per_det = 0;
    std::vector<uint32_t> track_ids;
    std::vector<float> attrs;
    size_t body_len = 0;

    static FrameView parse(const std::vector<uint8_t>& rec) {
        FrameView v;
        std::memcpy(&v.body_len, rec.data() + 4, 4);
        const uint8_t* b = rec.data() + 8;
        std::memcpy(&v.stream_index, b, 4);
        v.kpt_per_det = b[49];
        uint16_t n_det = 0;
        std::memcpy(&n_det, b + 50, 2);
        v.n_det = n_det;
        v.attr_per_det = b[60];
        size_t off = 64;
        for (uint32_t i = 0; i < v.n_det; ++i) {
            uint32_t tid;
            std::memcpy(&tid, b + off + 24, 4);
            v.track_ids.push_back(tid);
            off += 28;
        }
        off += size_t(v.n_det) * v.kpt_per_det * 3 * 4;
        for (uint32_t i = 0; i < v.n_det * v.attr_per_det; ++i) {
            float f;
            std::memcpy(&f, b + off, 4);
            v.attrs.push_back(f);
            off += 4;
        }
        return v;
    }
};

// ---- runtime helpers -----------------------------------------------------

Json add_line(uint32_t index, const std::string& url, const Json& analyzers,
              float score = 0.35f) {
    Json j;
    j["op"] = "add";
    j["req"] = "r-" + std::to_string(index);
    Json st;
    st["index"] = index;
    st["id"] = "syn-" + std::to_string(index);
    st["url"] = url;
    st["transport"] = "tcp";
    st["score_threshold"] = score;
    st["options"] = Json::object();
    j["stream"] = st;
    j["analyzers"] = analyzers;
    return j;
}

RuntimeConfig cfg_for(const Json& extra) {
    Json base{{"backend", {{"name", "synthetic"}, {"max_batch", 1}, {"infer_ms", 0}}},
              {"contexts_per_worker", 1},
              {"open_timeout_s", 3.0},
              {"reconnect_delay_s", 1.0},
              {"tracker", Json::object()},
              {"analyzers", Json::object({{"plugins", Json::array()}})}};
    for (auto it = extra.begin(); it != extra.end(); ++it) base[it.key()] = it.value();
    std::string ignore;
    return RuntimeConfig::from_json(base, ignore);
}

std::unique_ptr<Runtime> make_runtime(Writer& w, const Json& extra = Json::object()) {
    RuntimeConfig cfg = cfg_for(extra);
    std::string err;
    auto backend = create_backend(cfg.backend_name, cfg.backend_json, err);
    CHECK(backend != nullptr);
    return std::make_unique<Runtime>(std::move(backend), cfg, w);
}

// Bounded polling helper: returns false instead of blocking when `pred` never
// becomes true.
template <typename Pred>
bool wait_until(Pred pred, int timeout_ms = 5000) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return pred();
}

// True while a context thread holds the stream: either it is inside
// process() (holder_mu is held for the whole frame) or it claimed the stream
// and has not released it yet. Never blocks, so a 3 s frame cannot stall the
// poll loop on the lock.
bool busy(const std::shared_ptr<StreamState>& s) {
    if (!s) return false;
    std::unique_lock<std::mutex> lk(s->holder_mu, std::try_to_lock);
    if (!lk.owns_lock()) return true;
    return s->busy;
}

// Replies travel through the Writer thread, so wait for the record to land.
Json wait_reply(Capture& cap, const std::string& req, int timeout_ms = 4000) {
    Json r;
    wait_until(
        [&] {
            r = cap.reply_for(req);
            return !r.is_null();
        },
        timeout_ms);
    return r;
}

// ---- test plugin sources -------------------------------------------------
// Compiled at test time with the system compiler against the real ABI header,
// derived from this file's path so no extra CMake plumbing is needed.

const char* kPluginSource = R"PLUGIN(
#include "vb/vb_analyzer_abi.h"
#include <stdlib.h>
#include <string.h>

#if !defined(TP_ATTRS)
#define TP_ATTRS 1
#endif

static const char* const kNames[] = {TP_ATTRS == 2 ? "tp.a0" : "tp.a",
                                     TP_ATTRS == 2 ? "tp.a1" : "tp.a"};

static void* tp_create(const char* cfg, char* err, size_t errlen) {
    (void)cfg;
    if (err && errlen) err[0] = 0;
    return calloc(1, 8);
}
static void tp_destroy(void* self) { free(self); }

static int tp_on_frame(void* self, const vb_frame_meta* m, const vb_track* tracks,
                       size_t n, float* attrs_out, vb_emit_fn emit, void* sink) {
    (void)self; (void)m; (void)emit; (void)sink;
    for (size_t i = 0; i < n; i++) {
        if (!attrs_out) continue;
#if TP_ATTRS == 2
        attrs_out[i * 2 + 0] = TP_BASE + (float)tracks[i].track_id;
        attrs_out[i * 2 + 1] = TP_BASE + 100.0f + (float)tracks[i].track_id;
#else
        attrs_out[i] = TP_BASE + (float)tracks[i].track_id;
#endif
    }
    return 0;
}
static void tp_removed(void* self, uint32_t id, double t) { (void)self; (void)id; (void)t; }

static const vb_analyzer_api kApi = {
    VB_ANALYZER_ABI,
    TP_NAME,
    TP_ATTRS,
    kNames,
#if TP_NULL_CREATE
    NULL,
#else
    tp_create,
#endif
    tp_destroy,
    tp_on_frame,
    tp_removed,
};

const vb_analyzer_api* vb_analyzer_entry(void) { return &kApi; }
)PLUGIN";

// Dummy callbacks for the synthetic API tables in section 4.
void* dummy_create(const char*, char*, size_t) { return nullptr; }
void dummy_destroy(void*) {}
int dummy_on_frame(void*, const vb_frame_meta*, const vb_track*, size_t, float*,
                   vb_emit_fn, void*) {
    return 0;
}
void dummy_on_track_removed(void*, uint32_t, double) {}

std::string dir_of_this_file() {
    std::string f = __FILE__;
    size_t slash = f.find_last_of('/');
    return slash == std::string::npos ? std::string(".") : f.substr(0, slash);
}

// Compiles one test plugin; empty string when the compiler is unavailable.
std::string build_plugin(const std::string& dir, const std::string& name,
                         const char* plugin_name, int attrs, float base,
                         bool null_create) {
    std::string src = dir + "/" + name + ".c";
    std::string so = dir + "/" + name + ".so";
    {
        std::ofstream f(src);
        f << kPluginSource;
    }
    std::string cmd = "cc -shared -fPIC -o " + so + " " + src + " -I" + dir_of_this_file() +
                      "/../include -DTP_NAME=\\\"" + plugin_name + "\\\" -DTP_ATTRS=" +
                      std::to_string(attrs) + " -DTP_BASE=" + std::to_string(base) +
                      "f -DTP_NULL_CREATE=" + (null_create ? "1" : "0") + " 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) return "";
    if (!std::filesystem::exists(so)) return "";
    return so;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep progress visible on a crash
    const std::string fixtures = argc > 1 ? argv[1] : ".";
    // Directory of this test binary: the vb-runtime executable is its sibling.
    std::string prog_dir = ".";
    {
        std::string a0 = argc > 0 ? argv[0] : "";
        size_t slash = a0.find_last_of('/');
        if (slash != std::string::npos) prog_dir = a0.substr(0, slash);
    }

    // ---- 1) B1: malformed control lines fail, the process survives --------
    {
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w, Json{{"contexts_per_worker", 1}});
        std::string err;
        CHECK(rt->start(err));
        Json ok_add = add_line(0, "synthetic://?w=64&h=48&fps=30", Json::array());
        rt->handle_line(ok_add);
        CHECK(wait_until([&] { return cap.frames(0) > 0; }));

        struct Case {
            const char* line;
            const char* req;
        };
        // Every field a handler dereferences, with the field missing.
        const Case cases[] = {
            {R"({"op":"snapshot","req":"b1-1"})", "b1-1"},
            {R"({"op":"snapshot","req":"b1-2","stream_index":"x"})", "b1-2"},
            {R"({"op":"remove","req":"b1-3"})", "b1-3"},
            {R"({"op":"add","req":"b1-4"})", "b1-4"},
            {R"({"op":"add","req":"b1-5","stream":{"url":"synthetic://"}})", "b1-5"},
            {R"({"op":"add","req":"b1-6","stream":{"index":9}})", "b1-6"},
            {R"({"op":"add","req":"b1-7","stream":{"index":9,"url":"synthetic://"},"analyzers":7})", "b1-7"},
            {R"({"op":"set_threshold","req":"b1-8","stream_index":0})", "b1-8"},
            {R"({"op":"set_threshold","req":"b1-9","value":0.5})", "b1-9"},
            {R"({"op":"configure_analyzer","req":"b1-10","stream_index":0})", "b1-10"},
            {R"({"op":"configure_analyzer","req":"b1-11","name":"line_cross"})", "b1-11"},
            {R"({"op":"nonsense","req":"b1-12"})", "b1-12"},
            {R"({"op":12345})", ""},
        };
        for (const auto& c : cases) {
            rt->handle_line(json_parse(c.line));
            Json r = wait_reply(cap, c.req);
            CHECK(!r.is_null());
            CHECK(r.at("ok").get<bool>() == false);
            CHECK(r.contains("error"));
        }
        // Still alive and still producing frames.
        size_t before = cap.frames(0);
        CHECK(wait_until([&] { return cap.frames(0) > before; }));
        // A valid command still works afterwards.
        Json rm;
        rm["op"] = "remove";
        rm["req"] = "b1-rm";
        rm["stream_index"] = 0;
        rt->handle_line(rm);
        CHECK(wait_reply(cap, "b1-rm").at("ok").get<bool>() == true);
        rt->stop();
        w.stop();
        std::printf("B1 missing-field replies: OK\n");
    }

    // ---- 2) A4: tracker.enabled -------------------------------------------
    {
        std::string err;
        // Type rule matches config.py:285 (must be a boolean).
        std::string e1;
        RuntimeConfig bad = RuntimeConfig::from_json(
            Json{{"backend", {{"name", "synthetic"}}}, {"tracker", {{"enabled", "yes"}}}},
            e1);
        CHECK(!e1.empty());
        CHECK(e1.find("tracker.enabled") != std::string::npos);
        std::string e2;
        RuntimeConfig rc = RuntimeConfig::from_json(
            Json{{"backend", {{"name", "synthetic"}}}, {"tracker", {{"enabled", false}}}},
            e2);
        CHECK(e2.empty());
        CHECK(rc.tracker_enabled == false);
        std::string e3;
        RuntimeConfig rc2 = RuntimeConfig::from_json(
            Json{{"backend", {{"name", "synthetic"}}}, {"tracker", Json::object()}}, e3);
        CHECK(rc2.tracker_enabled == true);  // default

        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w, Json{{"tracker", {{"enabled", false}}}});
        CHECK(rt->start(err));
        // A needs_tracks() analyzer is refused at add time.
        Json lines = Json::array();
        lines.push_back(Json{{"name", "line_cross"},
                             {"config", {{"lines", Json::array({Json{{"id", "l"},
                                                                    {"a", Json::array({0.0, 0.5})},
                                                                    {"b", Json::array({1.0, 0.5})}}})}}}});
        Json add = add_line(4, "synthetic://?w=64&h=48&fps=60", lines);
        rt->handle_line(add);
        Json r = wait_reply(cap, "r-4");
        CHECK(!r.is_null());
        CHECK(r.at("ok").get<bool>() == false);
        CHECK(r.at("error").get<std::string>().find("requires tracks") != std::string::npos);
        // A needs_tracks()==false analyzer is accepted.
        Json ct = Json::array();
        ct.push_back(Json{{"name", "count_threshold"}, {"config", {{"max_count", 5}}}});
        Json add2 = add_line(4, "synthetic://?w=64&h=48&fps=60", ct);
        add2["req"] = "r-4b";
        rt->handle_line(add2);
        Json r2 = wait_reply(cap, "r-4b");
        CHECK(r2.at("ok").get<bool>() == true);
        // Every detection carries track_id 0 (no tracker ran).
        std::vector<uint8_t> rec;
        CHECK(wait_until([&] { return cap.last_frame(4, rec); }));
        FrameView v = FrameView::parse(rec);
        CHECK(v.n_det == 3);
        for (uint32_t id : v.track_ids) CHECK(id == 0);
        CHECK(v.attr_per_det == 0);
        rt->stop();
        w.stop();
        std::printf("A4 tracker.enabled=false: OK\n");
    }

    // ---- 3) C2: analyzer names never carry a path -------------------------
    {
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w);
        std::string err;
        CHECK(rt->start(err));
        const char* names[] = {"evil.so", "/tmp/evil.so", "../../etc/evil.so",
                               "plugin:/tmp/evil.so", "sub/dir.so"};
        int i = 0;
        for (const char* n : names) {
            Json an = Json::array();
            an.push_back(Json{{"name", n}, {"config", Json::object()}});
            std::string req = "c2-" + std::to_string(i);
            Json add = add_line(7, "synthetic://?w=32&h=32&fps=5", an);
            add["req"] = req;
            rt->handle_line(add);
            Json r = wait_reply(cap, req);
            CHECK(!r.is_null());
            CHECK(r.at("ok").get<bool>() == false);
            std::string e = r.at("error").get<std::string>();
            // Rejected as a name, never attempted as a load path: a dlopen of
            // e.g. /tmp/evil.so would report "plugin load failed".
            CHECK(e.find("must not be a path") != std::string::npos);
            CHECK(e.find("plugin load failed") == std::string::npos);
            ++i;
        }
        // An unregistered plugin id is refused too.
        Json an = Json::array();
        an.push_back(Json{{"name", "plugin:count"}, {"config", Json::object()}});
        Json add = add_line(7, "synthetic://?w=32&h=32&fps=5", an);
        add["req"] = "c2-unreg";
        rt->handle_line(add);
        Json r = wait_reply(cap, "c2-unreg");
        CHECK(!r.is_null());
        CHECK(r.at("ok").get<bool>() == false);
        CHECK(r.at("error").get<std::string>().find("unknown plugin") != std::string::npos);
        CHECK(rt->stream_count() == 0);
        rt->stop();
        w.stop();
        std::printf("C2 plugin path rejection: OK (%d names)\n", i);
    }

    // ---- 4) plugin API table validation -----------------------------------
    {
        auto table = [](const char* name, uint32_t attrs, const char* const* names_) {
            vb_analyzer_api a{};
            a.abi = VB_ANALYZER_ABI;
            a.name = name;
            a.attr_count = attrs;
            a.attr_names = names_;
            a.create = dummy_create;
            a.destroy = dummy_destroy;
            a.on_frame = dummy_on_frame;
            a.on_track_removed = dummy_on_track_removed;
            return a;
        };
        static const char* const kOne[] = {"a.b"};
        std::string err;
        vb_analyzer_api good = table("tp", 1, kOne);
        CHECK(validate_plugin_api(&good, err));

        // Each mandatory callback, null in turn.
        {
            vb_analyzer_api t = good;
            t.create = nullptr;
            CHECK(!validate_plugin_api(&t, err));
            CHECK(err.find("create") != std::string::npos);
        }
        {
            vb_analyzer_api t = good;
            t.destroy = nullptr;
            CHECK(!validate_plugin_api(&t, err));
            CHECK(err.find("destroy") != std::string::npos);
        }
        {
            vb_analyzer_api t = good;
            t.on_frame = nullptr;
            CHECK(!validate_plugin_api(&t, err));
            CHECK(err.find("on_frame") != std::string::npos);
        }
        {
            vb_analyzer_api t = good;
            t.on_track_removed = nullptr;
            CHECK(!validate_plugin_api(&t, err));
            CHECK(err.find("on_track_removed") != std::string::npos);
        }
        {
            vb_analyzer_api t = good;
            t.name = nullptr;
            CHECK(!validate_plugin_api(&t, err));
        }
        {
            vb_analyzer_api t = good;
            t.attr_names = nullptr;  // attr_count 1, no names
            CHECK(!validate_plugin_api(&t, err));
            CHECK(err.find("attr_names") != std::string::npos);
        }
        {
            vb_analyzer_api t = good;
            t.attr_count = 256;  // cannot be expressed in VBR1
            CHECK(!validate_plugin_api(&t, err));
            CHECK(err.find("255") != std::string::npos);
        }
        {
            vb_analyzer_api t = good;
            t.attr_count = 0;
            t.attr_names = nullptr;  // fine: no attributes
            CHECK(validate_plugin_api(&t, err));
        }
        std::printf("plugin api validation: OK\n");
    }

    // ---- 5) dlopen failure reports one usable message ---------------------
    {
        std::string err;
        CHECK(load_plugin_analyzer("/nonexistent/dir/vb_missing_plugin.so", err) == nullptr);
        CHECK(!err.empty());
        CHECK(err.find("plugin load failed") != std::string::npos);
        // A loadable library without the entry symbol is rejected with a
        // message, not a null string.
        std::string err2;
        auto a = load_plugin_analyzer("libSystem.B.dylib", err2);
        CHECK(a == nullptr);
        CHECK(!err2.empty());
        std::printf("plugin load errors: OK (%s)\n", err.c_str());
    }

    // ---- 6) configured plugin end-to-end + attribute interleaving ---------
    bool cc_ok = true;
    {
        std::string base = std::filesystem::temp_directory_path().string() +
                           "/vb_test_runtime_" + std::to_string(::getpid());
        std::error_code ec;
        std::filesystem::create_directories(base, ec);
        std::string so_a = build_plugin(base, "tp_a", "tp_a", 2, 10.0f, false);
        std::string so_b = build_plugin(base, "tp_b", "tp_b", 1, 30.0f, false);
        if (so_a.empty() || so_b.empty()) {
            cc_ok = false;
            std::printf("plugin end-to-end: SKIP (no working cc)\n");
        } else {
            Writer w;
            Capture cap;
            cap.install(w);
            RuntimeConfig cfg = cfg_for(Json::object());
            cfg.plugin_paths = {so_a, so_b};
            std::string err;
            auto backend = create_backend(cfg.backend_name, cfg.backend_json, err);
            CHECK(backend != nullptr);
            Runtime rt(std::move(backend), cfg, w);
            CHECK(rt.start(err));

            Json an = Json::array();
            an.push_back(Json{{"name", "plugin:tp_a"}, {"config", Json::object()}});
            an.push_back(Json{{"name", "plugin:tp_b"}, {"config", Json::object()}});
            rt.handle_line(add_line(0, "synthetic://?w=64&h=48&fps=30", an));
            Json r = wait_reply(cap, "r-0");
            CHECK(!r.is_null());
            CHECK(r.at("ok").get<bool>() == true);

            std::vector<uint8_t> rec;
            CHECK(wait_until([&] { return cap.last_frame(0, rec); }));
            FrameView v = FrameView::parse(rec);
            CHECK(v.n_det == 3);
            CHECK(v.attr_per_det == 3);  // 2 from tp_a + 1 from tp_b
            CHECK(v.attrs.size() == v.n_det * 3u);
            // [track][attribute]: tp_a's two values, then tp_b's one, per track.
            bool layout_ok = true;
            for (uint32_t i = 0; i < v.n_det; ++i) {
                float id = static_cast<float>(v.track_ids[i]);
                CHECK(v.track_ids[i] != 0);
                float* row = &v.attrs[i * 3];
                if (row[0] != 10.0f + id) layout_ok = false;
                if (row[1] != 110.0f + id) layout_ok = false;
                if (row[2] != 30.0f + id) layout_ok = false;
            }
            CHECK(layout_ok);
            rt.stop();
            w.stop();

            // A configured plugin with a null create must abort start().
            std::string so_bad = build_plugin(base, "tp_bad", "tp_bad", 1, 0.0f, true);
            CHECK(!so_bad.empty());
            {
                Writer w2;
                RuntimeConfig cfg2 = cfg_for(Json::object());
                cfg2.plugin_paths = {so_bad};
                std::string err2;
                auto b2 = create_backend(cfg2.backend_name, cfg2.backend_json, err2);
                Runtime rt2(std::move(b2), cfg2, w2);
                CHECK(!rt2.start(err2));
                CHECK(err2.find("create") != std::string::npos);
            }
            std::printf("plugin end-to-end + attr layout: OK\n");
        }
        std::filesystem::remove_all(base, ec);
    }

    // ---- 7) attribute interleaving + u8 count guard -----------------------
    {
        // Two analyzers, tracks t0/t1, spans 2 and 1 -> [track][attribute].
        const float a[] = {11, 21, 12, 22};  // analyzer A: 2 tracks x 2 attrs
        const float b[] = {31, 32};          // analyzer B: 2 tracks x 1 attr
        float out[6] = {0, 0, 0, 0, 0, 0};
        interleave_attrs(a, 2, 2, 3, 0, out);
        interleave_attrs(b, 2, 1, 3, 2, out);
        const float want[6] = {11, 21, 31, 12, 22, 32};
        for (int i = 0; i < 6; ++i) CHECK(out[i] == want[i]);
        // VBR1's per-detection counts are u8: 255 is encodable, 256 is not, and
        // the frame has to be dropped rather than sent with a clamped count.
        std::string err;
        CHECK(vbr1_counts_ok(0, 0, err));
        CHECK(vbr1_counts_ok(17, 2, err));
        CHECK(vbr1_counts_ok(255, 255, err));
        CHECK(!vbr1_counts_ok(256, 0, err));
        CHECK(err.find("keypoint count 256") != std::string::npos);
        CHECK(!vbr1_counts_ok(17, 256, err));
        CHECK(err.find("attribute count 256") != std::string::npos);
        std::printf("attr interleave + count guard: OK\n");
    }

    // ---- 8) remove: no frame after the reply, timeout is not success ------
    {
        Writer w;
        Capture cap;
        cap.install(w);
        // The frame lasts far longer than the runtime's 2 s remove timeout,
        // and the stream is observed at its *first* claim (busy was false
        // before, so the poll lands at the start of the frame), so the timeout
        // is deterministic instead of racing the end of the frame.
        auto rt = make_runtime(w, Json{{"backend", {{"name", "synthetic"},
                                                    {"max_batch", 1},
                                                    {"infer_ms", 6000}}}});
        std::string err;
        CHECK(rt->start(err));
        rt->handle_line(add_line(0, "synthetic://?w=64&h=48&fps=30", Json::array()));
        CHECK(wait_until([&] { return rt->stream(0) != nullptr; }));
        auto s = rt->stream(0);
        CHECK(s != nullptr);
        // Wait until a context thread is inside process() for this stream.
        CHECK(wait_until([&] { return busy(s); }, 4000));
        Json rm;
        rm["op"] = "remove";
        rm["req"] = "rm-slow";
        rm["stream_index"] = 0;
        auto t0 = std::chrono::steady_clock::now();
        rt->handle_line(rm);
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        Json r = wait_reply(cap, "rm-slow");
        CHECK(!r.is_null());
        CHECK(r.at("ok").get<bool>() == false);
        CHECK(r.at("error").get<std::string>().find("timeout") != std::string::npos);
        CHECK(dt < 4000);
        CHECK(!cap.has_control("stream_state", "state", "stopped"));
        rt->stop();
        w.stop();
        std::printf("remove timeout (no success, %lld ms): OK\n", (long long)dt);
    }
    {
        // Mid-batch removal: the frame of the removed stream must never land
        // after the remove reply, and a removal that succeeds must be silent
        // for that stream from then on.
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w, Json{{"backend", {{"name", "synthetic"},
                                                    {"max_batch", 2},
                                                    {"infer_ms", 400}}}});
        std::string err;
        CHECK(rt->start(err));
        rt->handle_line(add_line(0, "synthetic://?w=64&h=48&fps=60", Json::array()));
        rt->handle_line(add_line(1, "synthetic://?w=64&h=48&fps=60", Json::array()));
        CHECK(wait_until([&] { return rt->stream(1) != nullptr; }));
        auto s1 = rt->stream(1);
        CHECK(s1 != nullptr);
        CHECK(wait_until([&] { return busy(s1); }, 4000));
        Json rm;
        rm["op"] = "remove";
        rm["req"] = "rm-1";
        rm["stream_index"] = 1;
        rt->handle_line(rm);
        size_t after = cap.frames(1);
        Json r = wait_reply(cap, "rm-1");
        CHECK(!r.is_null());
        CHECK(r.at("ok").get<bool>() == true);
        CHECK(cap.has_control("stream_state", "state", "stopped"));
        CHECK(cap.stopped_after_last_data(1));  // record order, not just counts
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        CHECK(cap.frames(1) == after);   // nothing after the reply
        CHECK(cap.frames(0) > 0);        // the other stream kept running
        rt->stop();
        w.stop();
        std::printf("remove mid-batch: OK\n");
    }

    // ---- 8b) a held stream does not block the streams after it -------------
    {
        // The claim loop must try_lock: while stream 0 is held (a long frame,
        // or a control op between frames), stream 1 in the same batch has to
        // keep running. With a blocking lock the single context thread would
        // park on stream 0 and neither stream would progress.
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w, Json{{"backend", {{"name", "synthetic"},
                                                    {"max_batch", 2},
                                                    {"infer_ms", 0}}}});
        std::string err;
        CHECK(rt->start(err));
        rt->handle_line(add_line(0, "synthetic://?w=64&h=48&fps=60", Json::array()));
        rt->handle_line(add_line(1, "synthetic://?w=64&h=48&fps=60", Json::array()));
        CHECK(wait_until([&] { return rt->stream(0) != nullptr; }));
        auto s0 = rt->stream(0);
        CHECK(s0 != nullptr);
        // Hold stream 0 the way a frame does, from outside the pool.
        std::unique_lock<std::mutex> held(s0->holder_mu);
        size_t frames0_before = cap.frames(0);
        size_t frames1_before = cap.frames(1);
        CHECK(wait_until([&] { return cap.frames(1) > frames1_before + 5; }, 3000));
        CHECK(cap.frames(0) == frames0_before);  // stream 0 really is held
        held.unlock();
        rt->stop();
        w.stop();
        std::printf("held stream does not block others: OK\n");
    }

    // ---- 9) set_threshold semantics ---------------------------------------
    {
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w);
        std::string err;
        CHECK(rt->start(err));
        rt->handle_line(add_line(0, "synthetic://?w=64&h=48&fps=60", Json::array()));
        CHECK(wait_until([&] { return cap.frames(0) > 0; }));
        Json st;
        st["op"] = "set_threshold";
        st["req"] = "th-ok";
        st["stream_index"] = 0;
        st["value"] = 0.44;
        rt->handle_line(st);
        Json r = wait_reply(cap, "th-ok");
        CHECK(!r.is_null());
        CHECK(r.at("ok").get<bool>() == true);
        CHECK_NEAR(r.at("applied").at("value").get<double>(), 0.44, 1e-6);
        auto s = rt->stream(0);
        CHECK_NEAR(s->spec.score_threshold, 0.44, 1e-6);
        rt->stop();
        w.stop();
    }
    {
        // No frame within the apply window: failure, and the pending value is
        // dropped instead of landing later. As above, the stream is observed
        // at its first claim so the next frame is seconds away, not
        // milliseconds.
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w, Json{{"backend", {{"name", "synthetic"},
                                                    {"max_batch", 1},
                                                    {"infer_ms", 6000}}}});
        std::string err;
        CHECK(rt->start(err));
        rt->handle_line(add_line(2, "synthetic://?w=64&h=48&fps=30", Json::array()));
        CHECK(wait_until([&] { return rt->stream(2) != nullptr; }));
        auto s = rt->stream(2);
        CHECK(s != nullptr);
        CHECK(wait_until([&] { return busy(s); }, 4000));
        Json st;
        st["op"] = "set_threshold";
        st["req"] = "th-slow";
        st["stream_index"] = 2;
        st["value"] = 0.9;
        rt->handle_line(st);
        Json r = wait_reply(cap, "th-slow");
        CHECK(!r.is_null());
        CHECK(r.at("ok").get<bool>() == false);
        CHECK(r.at("error").get<std::string>().find("not applied") != std::string::npos);
        CHECK_NEAR(s->spec.score_threshold, 0.35, 1e-6);  // unchanged
        rt->stop();
        w.stop();
        std::printf("set_threshold timeout: OK\n");
    }

    // ---- 10) B8: bounded/cancellable writes, stop() with a stalled peer ---
    {
        // write_all_fd: a peer that never reads must not block; the cancel
        // predicate and the deadline both end the call.
        int sv[2];
        CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        int sndbuf = 4096;
        ::setsockopt(sv[1], SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
        std::vector<uint8_t> bulk(1 << 20, 0x5a);
        // Cancel from another thread after 100 ms.
        std::atomic<bool> cancel{false};
        std::thread flagger([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            cancel = true;
        });
        auto t0 = std::chrono::steady_clock::now();
        bool ok = write_all_fd(sv[1], bulk.data(), bulk.size(),
                               [&] { return cancel.load(); });
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        flagger.join();
        CHECK(!ok);
        CHECK(dt < 1500);
        // The deadline bounds the call even when nothing cancels.
        t0 = std::chrono::steady_clock::now();
        ok = write_all_fd(sv[1], bulk.data(), bulk.size(), {}, 200);
        dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - t0).count();
        CHECK(!ok);
        CHECK(dt < 1500);
        // A readable peer still receives everything (nothing cancels here).
        std::atomic<bool> never{false};
        std::thread reader([&] {
            std::vector<uint8_t> got;
            uint8_t buf[8192];
            ssize_t n;
            while ((n = ::recv(sv[0], buf, sizeof(buf), 0)) > 0)
                got.insert(got.end(), buf, buf + n);
        });
        ok = write_all_fd(sv[1], bulk.data(), bulk.size(),
                          [&] { return never.load(); });
        CHECK(ok);
        ::shutdown(sv[1], SHUT_WR);
        reader.join();
        ::close(sv[0]);
        ::close(sv[1]);
        std::printf("write_all_fd bounded/cancellable: OK\n");
    }
    {
        // Writer: a sink blocked on a peer that stopped reading must not stop
        // stop() from returning, and producers parked on backpressure must be
        // released.
        WriterConfig cfg;
        cfg.event_backpressure_limit = 8;
        cfg.stop_flush_ms = 200;
        Writer w(cfg);
        std::atomic<int> sink_calls{0};
        std::atomic<bool> sink_blocked{false};
        w.start([&](const uint8_t*, size_t) {
            ++sink_calls;
            sink_blocked = true;
            // Stand-in for a socket write whose peer is gone.
            while (!w.flush_expired()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        });
        std::atomic<int> pushed{0};
        std::atomic<bool> producers_done{false};
        std::vector<std::thread> producers;
        for (int i = 0; i < 3; ++i) {
            producers.emplace_back([&] {
                for (int k = 0; k < 100; ++k) {
                    std::vector<uint8_t> rec{'V', 'B', 'E', '1', 0, 0, 0, 0};
                    w.push_event(std::move(rec));
                    ++pushed;
                }
                producers_done = true;
            });
        }
        CHECK(wait_until([&] { return sink_blocked.load(); }, 2000));
        // The precondition: producers really are parked on backpressure.
        CHECK(wait_until([&] { return w.event_backpressure() > 0; }, 2000));
        auto t0 = std::chrono::steady_clock::now();
        w.stop();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        for (auto& t : producers) t.join();
        CHECK(producers_done.load());
        CHECK(dt < 3000);
        CHECK(w.flush_expired());  // the window expired, the sink gave up
        std::printf("Writer stop with stalled sink: OK (%lld ms, pushed %d, sink %d)\n",
                    (long long)dt, pushed.load(), sink_calls.load());
    }
    {
        // Runtime level: the context threads park in push_event() behind the
        // blocked sink, and rt.stop() has to release them before joining the
        // pool (otherwise the join deadlocks). A dwell analyzer with a 1 ms
        // repeat emits on every frame, so the queue overflows on its own.
        WriterConfig cfg;
        cfg.event_backpressure_limit = 4;
        cfg.stop_flush_ms = 200;
        Writer w(cfg);
        std::atomic<bool> sink_blocked{false};
        w.start([&](const uint8_t*, size_t) {
            sink_blocked = true;
            while (!w.flush_expired()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        });
        auto rt = make_runtime(w, Json{{"contexts_per_worker", 2}});
        std::string err;
        CHECK(rt->start(err));
        Json an = Json::array();
        an.push_back(Json{{"name", "dwell"},
                          {"config", {{"min_dwell_s", 0.0}, {"repeat_s", 0.001}}}});
        // Once the queue is over the limit an add() blocks in push_event (the
        // specified backpressure), so the adds run on their own thread and the
        // test body only calls stop().
        std::atomic<bool> adds_done{false};
        std::thread adder([&] {
            for (uint32_t i = 0; i < 3; ++i)
                rt->handle_line(add_line(i, "synthetic://?w=64&h=48&fps=120", an));
            adds_done = true;
        });
        CHECK(wait_until([&] { return sink_blocked.load(); }, 4000));
        CHECK(wait_until([&] { return w.event_backpressure() > 0; }, 4000));
        auto t0 = std::chrono::steady_clock::now();
        rt->stop();   // must release the parked producers before joining
        w.stop();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        CHECK(wait_until([&] { return adds_done.load(); }, 4000));
        adder.join();
        CHECK(dt < 3000);
        std::printf("Runtime stop with stalled sink: OK (%lld ms)\n", (long long)dt);
    }

    // ---- 11) stats FPS window ---------------------------------------------
    {
        StreamMetrics m;
        using namespace std::chrono;
        double now = duration_cast<duration<double, std::milli>>(
                         system_clock::now().time_since_epoch()).count();
        // Frames recorded 5 s ago: outside the window at query time.
        for (int i = 0; i < 30; ++i)
            m.record_frame(10.0f, 0.0f, now - 5000.0 + i * 10.0);
        Json j = m.to_json(0);
        CHECK_NEAR(j.at("fps").get<double>(), 0.0, 1e-9);
        // A single frame anywhere in the active one-second window counts as
        // one frame per second, regardless of the query phase.
        for (double age_ms : {100.0, 500.0, 900.0}) {
            StreamMetrics phase;
            phase.record_frame(10.0f, 0.0f, now - age_ms);
            Json phase_json = phase.to_json(0);
            CHECK_NEAR(phase_json.at("fps").get<double>(), 1.0, 1e-9);
        }
        // A live stream still reports a rate.
        StreamMetrics m2;
        for (int i = 0; i < 30; ++i)
            m2.record_frame(10.0f, 0.0f, now - 900.0 + i * 30.0);
        Json j2 = m2.to_json(0);
        CHECK(j2.at("fps").get<double>() > 10.0);
        CHECK(j2.at("fps").get<double>() < 60.0);
        CHECK(j2.at("processed_frames").get<uint64_t>() == 30);
        CHECK(j2.at("inference_ms_p50").get<double>() == 10.0);
        std::printf("stats fps window: OK (%.2f)\n", j2.at("fps").get<double>());
    }

    // ---- 12) stats JSON carries process context/time observability ---------
    {
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w, Json{{"contexts_per_worker", 999}});
        std::string err;
        CHECK(rt->start(err));
        rt->emit_stats();
        CHECK(wait_until([&] { return !cap.control("stats").empty(); }));
        auto records = cap.control("stats");
        CHECK(!records.empty());
        const Json& stats = records.back();
        CHECK(stats.at("effective_contexts").is_number_unsigned());
        CHECK(stats.at("effective_contexts").get<size_t>() == rt->config().contexts);
        CHECK(stats.at("effective_contexts").get<size_t>() ==
              static_cast<size_t>(rt->backend().caps().max_contexts));
        CHECK(stats.at("stats_monotonic_ms").is_number_integer());
        CHECK(stats.at("stats_wall_ms").is_number_integer());
        CHECK(stats.at("stats_pid").get<int>() > 0);
        rt->stop();
        w.stop();
        std::printf("stats context/time JSON: OK (%zu contexts)\n",
                    stats.at("effective_contexts").get<size_t>());
    }

    // ---- 13) B4: a start failure after the writer is running exits cleanly --
    {
        // vb-runtime lives next to this test binary in the build tree. The
        // failure is triggered by a configured plugin that cannot be loaded,
        // which is where Runtime::start() fails *after* main() started the
        // writer thread: returning without stopping it used to run a joinable
        // std::thread into ~Writer and terminate (SIGABRT) instead of exiting 1.
        std::string exe = prog_dir + "/vb-runtime";
        if (!std::filesystem::exists(exe)) {
            std::printf("start-failure exit: SKIP (no %s)\n", exe.c_str());
        } else {
            std::string cfg_path = prog_dir + "/test_runtime_bad_plugin.json";
            {
                std::ofstream f(cfg_path);
                f << R"({"backend":{"name":"synthetic"},)"
                     R"("contexts_per_worker":1,)"
                     R"("analyzers":{"plugins":["/nonexistent/vb_no_such_plugin.so"]}})";
            }
            std::string cmd = exe + " --ipc-fd 0 --config " + cfg_path +
                              " >/dev/null 2>&1";
            int rc = std::system(cmd.c_str());
            std::filesystem::remove(cfg_path);
            CHECK(WIFEXITED(rc));
            CHECK(WEXITSTATUS(rc) == 1);
            std::printf("start-failure exit code: OK (%d)\n", WEXITSTATUS(rc));
        }
    }

    (void)fixtures;
    std::printf("test_runtime: all OK (plugin end-to-end: %s)\n",
                cc_ok ? "ran" : "SKIPPED (no working cc)");
    return 0;
}
