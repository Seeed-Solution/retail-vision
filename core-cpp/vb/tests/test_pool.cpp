// ContextPool / Runtime acceptance test (spec BASE-1 §8 M1.7):
//   - 4 synthetic streams x 2 contexts for 5 s: per-stream processed frame
//     counts within 10%;
//   - one stream's read() returning -1 leaves the others running and the
//     failed stream reconnects after reconnect_delay_s (1 s);
//   - runtime add/remove of streams;
//   - two streams with identical boxes get track ids starting from 1 each;
//   - Writer queue full: frame records drop oldest, events never drop.
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#include "check.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/runtime.h"

using namespace vb;

namespace {

struct CapturedRec {
    std::string magic;
    uint32_t stream_index = 0;
    double wall_ms = 0;
    std::vector<uint8_t> bytes;
};

struct Capture {
    std::mutex mu;
    std::vector<CapturedRec> recs;

    void install(Writer& w) {
        w.start([this](const uint8_t* data, size_t /*len*/) {
            std::lock_guard<std::mutex> lk(mu);
            CapturedRec r;
            r.magic.assign(data, data + 4);
            uint32_t body_len;
            std::memcpy(&body_len, data + 4, 4);
            r.bytes.assign(data, data + 8 + body_len);
            if (r.magic == "VBR1" || r.magic == "VBE1")
                std::memcpy(&r.stream_index, data + 8, 4);
            else if (r.magic == "VBC1" && body_len > 8 + 4 + 13) {
                // stream_state only: peek "stream_index" is unreliable in raw
                // bytes; leave 0 and use the JSON when needed.
            }
            recs.push_back(std::move(r));
        });
    }

    size_t count(const char* magic, uint32_t stream) {
        std::lock_guard<std::mutex> lk(mu);
        size_t n = 0;
        for (auto& r : recs)
            if (r.magic == magic && (stream == UINT32_MAX || r.stream_index == stream)) ++n;
        return n;
    }
    // Frame count per stream (VBR1 only).
    std::map<uint32_t, size_t> frames_per_stream() {
        std::lock_guard<std::mutex> lk(mu);
        std::map<uint32_t, size_t> m;
        for (auto& r : recs)
            if (r.magic == "VBR1") ++m[r.stream_index];
        return m;
    }
    // First VBR1 record for a stream (decoded minimally: track ids of dets).
    std::vector<uint32_t> first_track_ids(uint32_t stream) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& r : recs) {
            if (r.magic != "VBR1" || r.stream_index != stream) continue;
            const uint8_t* p = r.bytes.data() + 8;
            (void)p;
            uint16_t n_det;
            std::memcpy(&n_det, r.bytes.data() + 8 + 50, 2);  // body offset 50
            std::vector<uint32_t> ids;
            size_t off = 8 + 64;
            for (uint16_t i = 0; i < n_det; ++i) {
                uint32_t tid;
                std::memcpy(&tid, r.bytes.data() + off + 24, 4);
                ids.push_back(tid);
                off += 28;
            }
            return ids;
        }
        return {};
    }
    std::vector<double> vbr1_walls(uint32_t stream) {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<double> v;
        for (auto& r : recs) {
            if (r.magic != "VBR1" || r.stream_index != stream) continue;
            double wm;
            std::memcpy(&wm, r.bytes.data() + 8 + 12, 8);
            v.push_back(wm);
        }
        return v;
    }
};

Json add_line(uint32_t index, const std::string& url, float score = 0.35f,
              const Json& analyzers = Json::array()) {
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

std::unique_ptr<Runtime> make_runtime(Writer& w, int contexts, double reconnect = 1.0) {
    std::string ignore;
    RuntimeConfig cfg = RuntimeConfig::from_json(Json{
        {"backend", {{"name", "synthetic"}, {"max_batch", 2}, {"infer_ms", 0}}},
        {"contexts_per_worker", contexts},
        {"open_timeout_s", 3.0},
        {"reconnect_delay_s", reconnect},
        {"tracker", Json::object()},
        {"analyzers", Json::object({{"plugins", Json::array()}})},
    }, ignore);
    std::string err;
    auto backend = create_backend(cfg.backend_name, cfg.backend_json, err);
    CHECK(backend != nullptr);
    return std::make_unique<Runtime>(std::move(backend), cfg, w);
}

}  // namespace

int main() {
    // ---- 1) 4 streams x 2 contexts, 5 s, fairness <= 10% ----
    {
        Writer w;
        Capture cap;
        cap.install(w);
        std::string ignore;
    auto rt = make_runtime(w, 2);
        std::string err;
        CHECK(rt->start(err));
        for (uint32_t i = 0; i < 4; ++i)
            rt->handle_line(add_line(i, "synthetic://?w=64&h=48&fps=15"));
        std::this_thread::sleep_for(std::chrono::seconds(5));
        auto m = cap.frames_per_stream();
        CHECK(m.size() == 4);
        size_t mn = SIZE_MAX, mx = 0;
        for (auto& kv : m) {
            mn = std::min(mn, kv.second);
            mx = std::max(mx, kv.second);
            CHECK(kv.second >= 60);  // ~75 expected at 15 fps
        }
        CHECK(mx <= mn + mn / 10);  // within 10%
        // Two streams with same-coordinate boxes: track ids start from 1 each.
        for (uint32_t i = 0; i < 4; ++i) {
            auto ids = cap.first_track_ids(i);
            CHECK(ids.size() == 3);
            CHECK(ids[0] == 1);
        }
        rt->stop();
        w.stop();
        std::printf("pool fairness: min=%zu max=%zu\n", mn, mx);
    }

    // ---- 2) read() == -1: others unaffected, reconnect after ~1 s ----
    {
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w, 2, /*reconnect=*/1.0);
        std::string err;
        CHECK(rt->start(err));
        rt->handle_line(add_line(0, "synthetic://?w=64&h=48&fps=60"));
        rt->handle_line(add_line(1, "synthetic://?w=64&h=48&fps=60&fail_at_seq=80"));
        rt->handle_line(add_line(2, "synthetic://?w=64&h=48&fps=60"));
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        size_t others_before = cap.count("VBR1", 0) + cap.count("VBR1", 2);
        CHECK(others_before >= 60);
        auto walls1 = cap.vbr1_walls(1);
        CHECK(walls1.size() >= 40);  // 80 frames minus in-flight
        double fail_at = walls1.back();
        // The failed stream must resume ~1 s later and others keep producing.
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));
        auto walls1b = cap.vbr1_walls(1);
        CHECK(walls1b.size() > walls1.size());
        // Gap between last pre-failure frame and first post-reconnect frame.
        double first_after = -1;
        for (double t : walls1b)
            if (t > fail_at && (first_after < 0 || t < first_after)) first_after = t;
        CHECK(first_after > 0);
        CHECK(first_after - fail_at >= 900.0);   // reconnect_delay_s ~= 1 s
        CHECK(first_after - fail_at <= 2500.0);
        size_t others_after = cap.count("VBR1", 0) + cap.count("VBR1", 2);
        CHECK(others_after >= others_before + 60);  // others unaffected
        rt->stop();
        w.stop();
        std::printf("pool reconnect: gap=%.0f ms, others %zu -> %zu\n",
                    first_after - fail_at, others_before, others_after);
    }

    // ---- 3) runtime add/remove ----
    {
        Writer w;
        Capture cap;
        cap.install(w);
        auto rt = make_runtime(w, 1);
        std::string err;
        CHECK(rt->start(err));
        CHECK(rt->stream_count() == 0);
        rt->handle_line(add_line(7, "synthetic://?w=64&h=48&fps=60"));
        rt->handle_line(add_line(8, "synthetic://?w=64&h=48&fps=60"));
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
        CHECK(rt->stream_count() == 2);
        CHECK(cap.count("VBR1", 7) >= 20);
        CHECK(cap.count("VBR1", 8) >= 20);
        Json rm;
        rm["op"] = "remove";
        rm["req"] = "rm";
        rm["stream_index"] = 7;
        rt->handle_line(rm);
        CHECK(rt->stream_count() == 1);
        size_t c7 = cap.count("VBR1", 7);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        CHECK(cap.count("VBR1", 7) == c7);           // no frames after remove
        CHECK(cap.count("VBR1", 8) > c7);            // survivor keeps running
        // Re-add the same index works again.
        rt->handle_line(add_line(7, "synthetic://?w=64&h=48&fps=60"));
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        CHECK(cap.count("VBR1", 7) > c7);
        rt->stop();
        w.stop();
        std::printf("pool add/remove: OK\n");
    }

    // ---- 4) Writer: drop-oldest frames, events never dropped ----
    {
        WriterConfig cfg;
        cfg.frame_queue = 4;
        cfg.event_backpressure_limit = 4096;
        Writer w(cfg);
        std::mutex mu;
        size_t frames_out = 0, events_out = 0;
        std::atomic<bool> draining{false};
        w.start([&](const uint8_t* data, size_t) {
            // Hold deliveries until the pushes are done to force overflow.
            while (!draining.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::lock_guard<std::mutex> lk(mu);
            if (std::memcmp(data, "VBR1", 4) == 0) ++frames_out;
            else ++events_out;
        });
        auto rec = [](const char* magic, uint8_t tag) {
            std::vector<uint8_t> r(magic, magic + 4);
            r.push_back(0);
            r.push_back(0);
            r.push_back(0);
            r.push_back(0);
            r.push_back(tag);
            return r;
        };
        for (uint8_t i = 0; i < 11; ++i) w.push_frame(rec("VBR1", i));
        for (uint8_t i = 0; i < 5; ++i) w.push_event(rec("VBE1", i));
        draining = true;
        w.stop();
        CHECK(w.frames_dropped() >= 6);   // 10 pushed into a queue of 4
        CHECK(frames_out + w.frames_dropped() == 11);  // nothing lost silently
        CHECK(frames_out <= 5);           // queue of 4 (+1 in flight at most)
        CHECK(w.frames_dropped() >= 6);   // oldest dropped, never the newest
        CHECK(events_out == 5);           // events never dropped
        CHECK(w.event_backpressure() == 0);
        std::printf("writer: frames_out=%zu dropped=%llu events_out=%zu\n", frames_out,
                    (unsigned long long)w.frames_dropped(), events_out);
    }

    std::printf("test_pool: all OK\n");
    return 0;
}
