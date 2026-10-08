#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>

#include "check.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/rate_crop.h"
#include "vb/runtime.h"

namespace {

void test_rate_and_geometry() {
    vb::RateLimiter r(1.0);
    CHECK(r.accept(0.0));
    CHECK(!r.accept(0.997));
    CHECK(r.accept(1.0));
    CHECK(r.accept(1.999));          // within the 2 ms acceptance tolerance
    CHECK(r.accept(5.0));                 // long pause does not create a burst
    CHECK(!r.accept(5.001));
    CHECK(r.accept(5.999));               // 2 ms acceptance tolerance
    CHECK(!r.accept(6.997));
    CHECK(r.accept(7.0));
    CHECK(r.accept(2.0));                 // backwards source clock resets
    CHECK(!r.accept(std::numeric_limits<double>::quiet_NaN()));
    CHECK(!r.set_max_fps(std::numeric_limits<double>::infinity()));
    vb::RateLimiter unlimited;
    CHECK(unlimited.accept(0));
    CHECK(unlimited.accept(0));

    // A 30 FPS source with small arrival jitter must still deliver the 10 FPS
    // deadline grid. This catches phase reset after a late accepted frame.
    for (double jitter_ms : {1.0, 2.0, 5.0, 10.0}) {
        vb::RateLimiter jittered(10.0);
        int accepted = 0;
        double previous = -1.0;
        for (int i = 0; i < 3000; ++i) {
            double t = static_cast<double>(i) / 30.0 +
                       ((i % 2) ? jitter_ms : -jitter_ms) / 1000.0;
            if (t <= previous) t = std::nextafter(previous, std::numeric_limits<double>::infinity());
            previous = t;
            if (jittered.accept(t)) ++accepted;
        }
        CHECK(accepted >= 995 && accepted <= 1005);
    }

    // A slower 15 FPS source is still capped at 10 FPS, without a burst after
    // a long gap. The fmod-based phase advance must stay bounded for this gap.
    vb::RateLimiter slow(10.0);
    int slow_accepted = 0;
    for (int i = 0; i < 1500; ++i)
        if (slow.accept(static_cast<double>(i) / 15.0)) ++slow_accepted;
    CHECK(slow_accepted >= 995 && slow_accepted <= 1005);
    CHECK(slow.accept(1000000.0));
    CHECK(!slow.accept(1000000.001));

    float roi[4] = {.25f, .5f, .75f, 1.f};
    auto rect = vb::crop_rect_px(roi, 1920, 1080);
    CHECK(rect.x0 == 480 && rect.y0 == 540 && rect.w == 960 && rect.h == 540);
    float odd_roi[4] = {.10f, .10f, .50f, .50f};
    auto odd = vb::crop_rect_px(odd_roi, 101, 81);
    CHECK(odd.x0 == 10 && odd.y0 == 8 && odd.w == 42 && odd.h == 34);
    float edge_roi[4] = {0.f, 0.f, 1.f, 1.f};
    auto edge = vb::crop_rect_px(edge_roi, 101, 81);
    CHECK(edge.x0 == 0 && edge.y0 == 0 && edge.w == 101 && edge.h == 81);
    auto huge = vb::crop_rect_px(edge_roi, INT32_MAX, 100);
    CHECK(huge.x0 == 0 && huge.w == INT32_MAX);
    float bad_roi[4] = {0.f, 0.f, .01f, .5f};
    bool bad = false;
    try { (void)vb::crop_rect_px(bad_roi, 100, 100); } catch (const std::invalid_argument&) { bad = true; }
    CHECK(bad);
    bad = false;
    try { (void)vb::crop_rect_px(edge_roi, 0, 100); } catch (const std::invalid_argument&) { bad = true; }
    CHECK(bad);
    bad = false;
    try { (void)vb::crop_geom(100, 100, vb::CropRectPx{99, 0, 2, 2}, 640, 640, vb::Align::Center); } catch (const std::invalid_argument&) { bad = true; }
    CHECK(bad);
    auto g = vb::crop_geom(1920, 1080, rect, 640, 640, vb::Align::Center);
    CHECK_NEAR(g.scale, 2.0 / 3.0, 1e-5);
    CHECK_NEAR(g.pad_x, -320.0, 1e-5);
    CHECK_NEAR(g.pad_y, -220.0, 1e-5);
    float sx = 0, sy = 0;
    g.to_source_norm(.5f, .5f, sx, sy);
    CHECK_NEAR(sx, .5, 1e-5);
    CHECK_NEAR(sy, .75, 1e-5);
}

vb::Json add_line_url(uint32_t index, const std::string& url, const vb::Json& options) {
    return vb::Json{{"op", "add"}, {"req", "rate"}, {"stream", {
        {"index", index}, {"url", url},
        {"options", options}}}, {"analyzers", vb::Json::array()}};
}

vb::Json add_line(uint32_t index, const vb::Json& options) {
    return add_line_url(index, "synthetic://?w=64&h=48&fps=15", options);
}

void test_host_crop_stride_and_bgr() {
    std::vector<uint8_t> bytes(16 * 3, 0);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) {
            bytes[y * 16 + x * 3 + 0] = static_cast<uint8_t>(10 + y);
            bytes[y * 16 + x * 3 + 1] = static_cast<uint8_t>(20 + x);
            bytes[y * 16 + x * 3 + 2] = 30;
        }
    vb::FrameBuf f;
    f.w = f.full_w = 4; f.h = f.full_h = 3; f.stride = f.full_stride = 16;
    f.fmt = vb::PixFmt::BGR888; f.mem = vb::Mem::Host; f.host = bytes.data();
    f.full_host = f.host;
    f.full_stride = 0; // Explicit backing pointer may use the source stride.
    std::string err;
    CHECK(vb::host_crop_frame(f, vb::CropRectPx{1, 1, 2, 2}, err));
    CHECK(f.w == 2 && f.h == 2 && f.stride == 16 && f.full_w == 4 && f.full_h == 3);
    CHECK(f.host[0] == 11 && f.host[1] == 21 && f.host[2] == 30);
    CHECK(f.full_host[0] == 10 && f.full_host[16] == 11);
    const auto* saved = f.host;
    f.full_w = INT32_MAX;
    CHECK(!vb::host_crop_frame(f, vb::CropRectPx{0, 0, 1, 1}, err));
    CHECK(f.host == saved);
    f.full_w = 4;
    f.full_h = INT32_MAX;
    CHECK(!vb::host_crop_frame(f, vb::CropRectPx{0, 0, 1, 1}, err));
    CHECK(f.host == saved);
    size_t bytes_count = 1;
    CHECK(!vb::host_rgb_layout_bytes(100000000, 1, 300000000, bytes_count));
    CHECK(bytes_count == 0);
    CHECK(!vb::host_rgb_layout_bytes(4, 3, INT32_MAX, bytes_count));
    CHECK(vb::host_rgb_layout_bytes(1920, 1080, 5760, bytes_count));
    CHECK(bytes_count == 1920u * 1080u * 3u);
    std::printf("host BGR crop: stride/pointer/full backing preserved\n");
}

void test_native_add_validation_and_wiring() {
    vb::Writer writer;
    writer.start([](const uint8_t*, size_t) {});
    vb::RuntimeConfig cfg;
    cfg.backend_name = "synthetic";
    cfg.backend_json = R"({"name":"synthetic"})";
    cfg.contexts = 1;
    std::string err;
    auto backend = vb::create_backend("synthetic", cfg.backend_json, err);
    CHECK(backend != nullptr);
    vb::Runtime rt(std::move(backend), cfg, writer);
    std::atomic<int> roi_src_w{0}, roi_src_h{0}, roi_seq{-1};
    rt.on_frame_rec = [&](const vb::WireFrameRec& rec) {
        if (rec.stream_index == 2) {
            roi_src_w = rec.src_w;
            roi_src_h = rec.src_h;
            roi_seq = static_cast<int>(rec.seq);
        }
    };
    vb::Json last_reply;
    rt.on_reply_record = [&](const vb::Json& j) {
        if (j.value("op", std::string()) == "reply") last_reply = j;
    };
    CHECK(rt.start(err));

    rt.handle_line(add_line(0, vb::Json()));
    CHECK(!last_reply.value("ok", true));
    CHECK(last_reply.value("error", std::string()).find("options") != std::string::npos);
    rt.handle_line(add_line(0, vb::Json{{"max_fps", 1.0}}));
    for (int i = 0; i < 500 && !rt.stream(0); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto s = rt.stream(0);
    CHECK(s != nullptr);
    CHECK_NEAR(s->rate_limiter.max_fps(), 1.0, 1e-12);
    std::this_thread::sleep_for(std::chrono::seconds(10));
    auto limited = s->metrics.to_json(0);
    CHECK(s->metrics.processed() >= 9 && s->metrics.processed() <= 11);
    CHECK(limited["rate_skipped"].get<uint64_t>() >= 137 &&
          limited["rate_skipped"].get<uint64_t>() <= 143);
    CHECK(limited["dropped_frames"].get<uint64_t>() == 0);
    std::printf("runtime 10s synthetic 15fps max_fps=1: processed=%llu rate_skipped=%llu dropped=%llu\n",
                static_cast<unsigned long long>(s->metrics.processed()),
                static_cast<unsigned long long>(limited["rate_skipped"].get<uint64_t>()),
                static_cast<unsigned long long>(limited["dropped_frames"].get<uint64_t>()));

    rt.handle_line(vb::Json{{"op", "remove"}, {"req", "remove-rate"}, {"stream_index", 0}});
    rt.handle_line(add_line(1, vb::Json{{"max_fps", 0.0}}));
    for (int i = 0; i < 500 && !rt.stream(1); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto unlimited = rt.stream(1);
    CHECK(unlimited != nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(unlimited->metrics.processed() >= 3);
    CHECK(unlimited->metrics.to_json(1)["rate_skipped"].get<uint64_t>() == 0);
    std::printf("runtime 0 unlimited 0.3s: processed=%llu rate_skipped=0\n",
                static_cast<unsigned long long>(unlimited->metrics.processed()));

    rt.handle_line(add_line(2, vb::Json{{"roi_crop", vb::Json::array({0.25, 0.25, 0.75, 0.75})}}));
    for (int i = 0; i < 500 && !rt.stream(2); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(rt.stream(2) != nullptr);
    for (int i = 0; i < 1000 && roi_src_w.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(roi_src_w == 64 && roi_src_h == 48);
    CHECK(roi_seq == 0);  // handshake frame is processed exactly once
    std::printf("native host ROI: full source geometry %dx%d, first seq=0\n",
                roi_src_w.load(), roi_src_h.load());

    rt.stop();
    writer.stop();

    // The add queue has an independent bound from the writer queue. A
    // deterministic slow ROI read fills 1 inflight + 15 queued requests;
    // request 17 is rejected immediately.
    {
        vb::Writer qw;
        qw.start([](const uint8_t*, size_t) {});
        auto qb = vb::create_backend("synthetic", cfg.backend_json, err);
        CHECK(qb != nullptr);
        vb::Runtime qr(std::move(qb), cfg, qw);
        std::atomic<int> queue_full{0};
        qr.on_reply_record = [&](const vb::Json& j) {
            if (j.value("error", std::string()).find("queue full") != std::string::npos)
                ++queue_full;
        };
        CHECK(qr.start(err));
        const vb::Json slow_roi{{"roi_crop", vb::Json::array({0.1, 0.1, 0.9, 0.9})}};
        for (uint32_t i = 0; i < 17; ++i)
            qr.handle_line(add_line_url(i, "synthetic://?w=64&h=48&read_delay_ms=3000", slow_roi));
        CHECK(queue_full.load() == 1);
        qr.stop();
        qw.stop();
        std::printf("add queue bound: 16 accepted, 17th rejected\n");
    }

    // Cancellation releases the index reservation. The old slow worker must
    // not register or emit a frame after index 0 is reused by the new add.
    {
        vb::Writer rw;
        rw.start([](const uint8_t*, size_t) {});
        auto rb = vb::create_backend("synthetic", cfg.backend_json, err);
        CHECK(rb != nullptr);
        vb::Runtime rr(std::move(rb), cfg, rw);
        const vb::Json slow_roi{{"roi_crop", vb::Json::array({0.1, 0.1, 0.9, 0.9})}};
        std::atomic<int> frames{0};
        std::atomic<int> first_seq{-1};
        rr.on_frame_rec = [&](const vb::WireFrameRec& rec) {
            if (rec.stream_index == 0) { ++frames; first_seq = static_cast<int>(rec.seq); }
        };
        CHECK(rr.start(err));
        rr.handle_line(add_line_url(1, "synthetic://?w=64&h=48&fps=60", vb::Json::object()));
        for (int i = 0; i < 500 && !rr.stream(1); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        CHECK(rr.stream(1) != nullptr);
        rr.handle_line(add_line_url(0, "synthetic://?w=64&h=48&read_delay_ms=1000", slow_roi));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        auto ctl_begin = std::chrono::steady_clock::now();
        rr.handle_line(vb::Json{{"op", "set_threshold"}, {"req", "during-open"},
                                {"stream_index", 1}, {"value", 0.44}});
        auto ctl_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - ctl_begin).count();
        CHECK(ctl_ms < 250);
        rr.handle_line(vb::Json{{"op", "remove"}, {"req", "cancel-open"}, {"stream_index", 0}});
        rr.handle_line(add_line_url(0, "synthetic://?w=64&h=48&fps=60", vb::Json::object()));
        for (int i = 0; i < 1500 && frames.load() == 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        CHECK(frames.load() > 0 && first_seq.load() == 0);
        rr.stop();
        rw.stop();
        std::printf("cancel/reuse: index 0 reused, only new seq=0 frame observed; control=%lldms\n",
                    static_cast<long long>(ctl_ms));
    }
}

}  // namespace

int main() {
    test_rate_and_geometry();
    test_host_crop_stride_and_bgr();
    test_native_add_validation_and_wiring();
    return 0;
}
