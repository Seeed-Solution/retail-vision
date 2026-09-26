// Synthetic frame source + inference backend (spec BASE-1 §8 M1.7).
//
// URL: synthetic://?w=1280&h=720&fps=15&boxes=3[&fail_at_seq=N]
//   - deterministic pixel pattern, seq increments monotonically and survives
//     close()/open() (so a forced failure at fail_at_seq happens once);
//   - fail_at_seq: read() returns -1 once when seq reaches that value
//     (used by test_pool to exercise the reconnect path).
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <thread>

#include "sources/synthetic.h"
#include "vb/json.h"
#include "vb/letterbox.h"

namespace vb {

namespace {

double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

double now_ms_epoch() {
    using namespace std::chrono;
    return duration<double, std::milli>(system_clock::now().time_since_epoch()).count();
}

// Minimal query-string parser for the synthetic URL (after '?').
std::map<std::string, std::string> parse_query(const std::string& url) {
    std::map<std::string, std::string> q;
    size_t pos = url.find('?');
    if (pos == std::string::npos) return q;
    size_t start = pos + 1;
    while (start <= url.size()) {
        size_t amp = url.find('&', start);
        if (amp == std::string::npos) amp = url.size();
        if (amp > start) {
            std::string kv = url.substr(start, amp - start);
            size_t eq = kv.find('=');
            if (eq == std::string::npos) q[kv] = "";
            else q[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
        start = amp + 1;
    }
    return q;
}

long query_int(const std::map<std::string, std::string>& q, const char* key, long dflt) {
    auto it = q.find(key);
    if (it == q.end() || it->second.empty()) return dflt;
    char* end = nullptr;
    long v = std::strtol(it->second.c_str(), &end, 10);
    return (end && *end == '\0') ? v : dflt;
}

struct SyntheticParams {
    int w = 1280, h = 720, fps = 15, boxes = 3;
    uint64_t fail_at_seq = 0;  // 0 = never fail

    static SyntheticParams from_url(const std::string& url) {
        auto q = parse_query(url);
        SyntheticParams p;
        p.w = static_cast<int>(query_int(q, "w", p.w));
        p.h = static_cast<int>(query_int(q, "h", p.h));
        p.fps = static_cast<int>(query_int(q, "fps", p.fps));
        p.boxes = static_cast<int>(query_int(q, "boxes", p.boxes));
        p.fail_at_seq = static_cast<uint64_t>(query_int(q, "fail_at_seq", 0));
        if (p.w <= 0) p.w = 1280;
        if (p.h <= 0) p.h = 720;
        if (p.fps <= 0) p.fps = 15;
        if (p.fps > 1000) p.fps = 1000;
        if (p.boxes < 0) p.boxes = 3;
        return p;
    }
};

class SyntheticSource : public FrameSource {
public:
    explicit SyntheticSource(SyntheticParams p) : p_(p) {}

    bool open(std::string& err) override {
        (void)err;
        std::lock_guard<std::mutex> lk(mu_);
        next_tick_s_ = now_s();
        return true;
    }

    int read(FrameBuf& out, int timeout_ms) override {
        (void)timeout_ms;  // pacing below already bounds the wait
        double target;
        {
            std::lock_guard<std::mutex> lk(mu_);
            target = next_tick_s_;
            next_tick_s_ += 1.0 / static_cast<double>(p_.fps);
        }
        double now = now_s();
        if (target > now) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(target - now));
        }
        uint64_t seq;
        {
            std::lock_guard<std::mutex> lk(mu_);
            seq = seq_++;
            if (p_.fail_at_seq != 0 && seq == p_.fail_at_seq) return -1;
        }
        auto pixels = std::make_shared<std::vector<uint8_t>>(
            static_cast<size_t>(p_.w) * static_cast<size_t>(p_.h) * 3);
        for (size_t i = 0; i < pixels->size(); ++i)
            (*pixels)[i] = static_cast<uint8_t>((seq * 31 + i) & 0xff);
        out.stream_index = 0;
        out.seq = seq;
        out.wall_ms = now_ms_epoch();
        out.t_mono_s = now_s();
        out.w = p_.w;
        out.h = p_.h;
        out.stride = p_.w * 3;
        out.fmt = PixFmt::RGB888;
        out.mem = Mem::Host;
        out.host = pixels->data();
        out.full_w = p_.w;
        out.full_h = p_.h;
        out.hold = pixels;
        return 1;
    }

    void close() override {}
    const char* decode_path() const override { return "synthetic"; }

private:
    SyntheticParams p_;
    std::mutex mu_;
    uint64_t seq_ = 0;
    double next_tick_s_ = 0;
};

class SyntheticContext : public InferenceContext {
public:
    SyntheticContext(int model_w, int model_h, int boxes, double infer_ms)
        : model_w_(model_w), model_h_(model_h), boxes_(boxes), infer_ms_(infer_ms) {}

    int infer(const FrameBuf* const* frames, size_t n, float score, float nms,
              DetectionResult* out, std::string& err) override {
        (void)score;
        (void)nms;
        (void)err;
        if (infer_ms_ > 0)
            std::this_thread::sleep_for(std::chrono::duration<double>(infer_ms_ / 1000.0));
        for (size_t i = 0; i < n; ++i) {
            const FrameBuf& f = *frames[i];
            DetectionResult& r = out[i];
            r.geom = LetterboxGeom::fit(f.w, f.h, model_w_, model_h_, Align::Center);
            r.dets.clear();
            r.kpts.clear();
            for (int k = 0; k < boxes_; ++k) {
                double s = static_cast<double>(f.seq);
                float cx = static_cast<float>(
                    0.10 + 0.003 * s + 0.25 * k - std::floor(0.10 + 0.003 * s + 0.25 * k));
                float cy = static_cast<float>(0.5 + 0.05 * std::sin(0.1 * s + k));
                Detection d;
                d.cx = cx;
                d.cy = cy;
                d.w = 0.12f;
                d.h = 0.20f;
                d.score = 0.90f;
                d.class_id = k;
                r.dets.push_back(d);
            }
            r.preprocess_ms = 0;
            r.inference_ms = static_cast<float>(infer_ms_);
            r.postprocess_ms = 0;
        }
        return 0;
    }

private:
    int model_w_, model_h_, boxes_;
    double infer_ms_;
};

class SyntheticBackend : public Backend {
public:
    explicit SyntheticBackend(const std::string& json) {
        if (!json.empty()) {
            std::string err;
            Json j = json_parse(json);
            if (j.contains("max_batch"))
                max_batch_ = std::min(8, std::max(1, j.at("max_batch").get<int>()));
            if (j.contains("model_w")) model_w_ = j.at("model_w").get<int>();
            if (j.contains("model_h")) model_h_ = j.at("model_h").get<int>();
            if (j.contains("boxes")) boxes_ = std::max(0, j.at("boxes").get<int>());
            if (j.contains("infer_ms")) infer_ms_ = j.at("infer_ms").get<double>();
        }
    }

    const char* name() const override { return "synthetic"; }
    Caps caps() const override {
        Caps c;
        c.max_contexts = 8;
        c.max_batch = max_batch_;
        c.keypoints = 0;
        c.exclusive_device = false;
        return c;
    }
    std::pair<int, int> model_hw() const override { return {model_w_, model_h_}; }
    std::string model_sha256() const override {
        return "0000000000000000000000000000000000000000000000000000000000synthetic";
    }
    std::unique_ptr<FrameSource> create_source(const StreamSpec& s,
                                               std::string& err) override {
        if (s.url.rfind("synthetic://", 0) != 0) {
            err = "synthetic backend cannot open url: " + s.url;
            return nullptr;
        }
        return std::make_unique<SyntheticSource>(SyntheticParams::from_url(s.url));
    }
    std::unique_ptr<InferenceContext> create_context(int index,
                                                     std::string& err) override {
        (void)index;
        (void)err;
        return std::make_unique<SyntheticContext>(model_w_, model_h_, boxes_, infer_ms_);
    }

private:
    int max_batch_ = 2, model_w_ = 640, model_h_ = 640, boxes_ = 3;
    double infer_ms_ = 0;
};

}  // namespace

std::unique_ptr<Backend> make_synthetic_backend(const std::string& backend_json,
                                                std::string& err) {
    (void)err;
    return std::make_unique<SyntheticBackend>(backend_json);
}

}  // namespace vb
