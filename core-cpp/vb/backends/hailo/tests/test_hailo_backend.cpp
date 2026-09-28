// Hailo backend offline tests (spec BASE-1 §M2.3, VB_HAILO_STUB build):
// batch policy, frame batcher, runner lifecycle, runtime config parsing, and
// the stub runner decoded end-to-end through the shared yolo_pose decoder
// with hand-computed DFL and keypoint expectations. The keypoint expectations
// pin the parameterisation of the HEF's raw kx/ky against the shared
// decoder's `(2*k + g - 0.5) * stride` (the RK/Ultralytics form); if the
// on-device fixed-input measurement ever shows the HEF needs the other half
// cell, the compensation goes in hailo_backend.cpp's dequantiser and this
// file's kStubKxBias changes with it.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "batch_policy.h"
#include "batched_hailo_runner.h"
#include "check.h"
#include "frame_batcher.h"
#include "hailo_backend.h"
#include "runner_lifecycle.h"
#include "runtime_config.h"

#include "vb/backend.h"
#include "vb/post.h"

using namespace vb;

// ---- batch policy ---------------------------------------------------------

static void test_batch_policy() {
    BatchConfig auto_cfg = parseBatchMode("auto");
    BatchConfig off_cfg = parseBatchMode("off");
    CHECK(batchModeName(auto_cfg) == std::string("auto"));
    CHECK(batchModeName(off_cfg) == std::string("off"));
    CHECK(batchModeName(parseBatchMode("8")) == std::string("fixed"));
    CHECK(parseBatchMode("4").fixed_size == 4);
    CHECK(parseBatchWaitMs("0") == 0);
    CHECK(parseBatchWaitMs("20") == 20);

    bool threw = false;
    try {
        (void)parseBatchMode("3");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    // Off never batches; fixed keeps the HEF multi-context flag.
    BatchDecision d = chooseBatch(off_cfg, 1, true, 16);
    CHECK(!d.shared && d.batch_size == 1);
    d = chooseBatch(parseBatchMode("4"), 1, true, 16);
    CHECK(d.shared && d.multi_context && d.batch_size == 4);
    // Auto falls back to the legacy (per-stream hailonet) shape unless the
    // HEF is one multi-context network group.
    d = chooseBatch(auto_cfg, 2, false, 16);
    CHECK(!d.shared && d.batch_size == 1);
    d = chooseBatch(auto_cfg, 1, false, 16);
    CHECK(!d.shared && d.batch_size == 1);
    d = chooseBatch(auto_cfg, 1, true, 1);
    CHECK(d.shared && d.batch_size == 1);
    d = chooseBatch(auto_cfg, 1, true, 3);
    CHECK(d.shared && d.batch_size == 1);
    d = chooseBatch(auto_cfg, 1, true, 4);
    CHECK(d.shared && d.batch_size == 4);
    d = chooseBatch(auto_cfg, 1, true, 8);
    CHECK(d.shared && d.batch_size == 8);

    threw = false;
    try {
        (void)chooseBatch(auto_cfg, 0, true, 1);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

// ---- frame batcher --------------------------------------------------------

static void test_frame_batcher() {
    FrameBatcher b(2, 2, 0, 2);
    BatchFrame f0;
    f0.stream = 0;
    f0.seq = 1;
    CHECK(b.enqueue(std::move(f0)));
    std::vector<BatchFrame> out;
    CHECK(b.take(out));
    CHECK(out.size() == 1 && out[0].stream == 0 && out[0].seq == 1);

    // One frame per stream, then the fill-up round. take() advances the
    // round-robin cursor on every call; the earlier single-frame take left it
    // at stream 1, so this take starts there (fall's fairness rule:
    // consecutive takes rotate the start stream).
    for (int round = 0; round < 2; ++round) {
        for (int s : {0, 1}) {
            BatchFrame f;
            f.stream = s;
            f.seq = static_cast<uint64_t>(round * 2 + s);
            CHECK(b.enqueue(std::move(f)));
        }
    }
    CHECK(b.take(out));
    CHECK(out.size() == 2);
    CHECK(out[0].stream == 1 && out[1].stream == 0);
    CHECK(b.take(out));
    CHECK(out.size() == 2);
    CHECK(out[0].stream == 0 && out[1].stream == 1);

    // Depth overflow drops the oldest and counts it.
    FrameBatcher d(1, 1, 0, 2);
    for (uint64_t i = 0; i < 4; ++i) {
        BatchFrame f;
        f.stream = 0;
        f.seq = i;
        CHECK(d.enqueue(std::move(f)));
    }
    CHECK(d.take(out));
    CHECK(out.size() == 1 && out[0].seq == 2);  // 0 and 1 were dropped
    BatchStats stats = d.stats();
    CHECK(stats.drops.size() == 1 && stats.drops[0] == 2);
    CHECK(stats.histogram[1] == 1);

    // stop() wakes take() and rejects later enqueues.
    FrameBatcher w(1, 1, 0, 1);
    w.stop(true);
    CHECK(!w.take(out));
    BatchFrame late;
    late.stream = 0;
    CHECK(!w.enqueue(std::move(late)));
}

// ---- runner lifecycle -----------------------------------------------------

static void test_runner_lifecycle() {
    RunnerLifecycle lc;
    int quits = 0;
    lc.fail([&] { ++quits; });
    lc.fail([&] { ++quits; });  // one-shot: the second fail is a no-op
    CHECK(quits == 1);
    CHECK(!lc.shouldRunLoop());
    CHECK(lc.exitCode(0) == 4);
    CHECK(lc.exitCode(3) == 3);
}

// ---- runtime config -------------------------------------------------------

static void test_runtime_config() {
    CHECK(hailo_config::parseQueueDepth("1") == 1);
    CHECK(hailo_config::parseQueueDepth("8") == 8);
    bool threw = false;
    try {
        (void)hailo_config::parseQueueDepth("0");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)hailo_config::parseQueueDepth("9");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    CHECK(hailo_config::parseDropOnLatency("true"));
    CHECK(hailo_config::parseDropOnLatency("1"));
    CHECK(!hailo_config::parseDropOnLatency("false"));
    threw = false;
    try {
        (void)hailo_config::parseDropOnLatency("maybe");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    CHECK_STREQ(hailo_config::rtspDepayChain("h264"),
                "rtph264depay ! h264parse ! decodebin");
    const char* h265 = hailo_config::rtspDepayChain("h265");
    CHECK(std::string(h265).find("v4l2slh265dec") != std::string::npos);
    CHECK(std::string(h265).find("gldownload") != std::string::npos);
    threw = false;
    try {
        (void)hailo_config::rtspDepayChain("mpeg4");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

// ---- stub runner through the shared decoder (hand-computed outputs) -------

namespace {

constexpr int kSide = 80;      // stride 8 level
constexpr int kPlane = kSide * kSide;
constexpr int kStride = 8;

struct LevelBytes {
    std::vector<uint8_t> box;    // [64, 80, 80] u8, scale 1 zp 0
    std::vector<uint8_t> score;  // [1, 80, 80] u8, scale 1/255 zp 0
    std::vector<uint16_t> kpt;   // [51, 80, 80] u16, scale 0.01 zp 0
};

// dist_bins[d] = the DFL bin that carries all the mass for direction d.
LevelBytes make_level(const int dist_bins[4], int cell_a, int cell_b,
                      float kx0, float ky0, float kc0, float kx1, float ky1,
                      float kc1) {
    LevelBytes lv;
    lv.box.assign(64 * kPlane, 0);
    lv.score.assign(kPlane, 0);
    lv.kpt.assign(51 * kPlane, 0);
    for (int cell : {cell_a, cell_b}) {
        for (int d = 0; d < 4; ++d)
            lv.box[(static_cast<size_t>(d) * 16 + dist_bins[d]) * kPlane + cell] = 200;
        lv.score[cell] = 220;  // 220/255 = 0.8627
        auto put = [&](int j, float kx, float ky, float kc) {
            lv.kpt[(static_cast<size_t>(j) * 3 + 0) * kPlane + cell] =
                static_cast<uint16_t>(kx * 100.0f + 0.5f);
            lv.kpt[(static_cast<size_t>(j) * 3 + 1) * kPlane + cell] =
                static_cast<uint16_t>(ky * 100.0f + 0.5f);
            lv.kpt[(static_cast<size_t>(j) * 3 + 2) * kPlane + cell] =
                static_cast<uint16_t>(kc * 100.0f + 0.5f);
        };
        put(0, kx0, ky0, kc0);
        put(1, kx1, ky1, kc1);
    }
    return lv;
}

std::vector<HailoVStreamInfo> pose_infos() {
    std::vector<HailoVStreamInfo> infos;
    for (int32_t side : {20, 40, 80}) {
        HailoVStreamInfo box, score, kpt;
        box.features = 64; box.height = side; box.width = side;
        box.format_type = kHailoFormatUint8; box.qp_scale = 1.0f; box.qp_zp = 0;
        score.features = 1; score.height = side; score.width = side;
        score.format_type = kHailoFormatUint8;
        score.qp_scale = 1.0f / 255.0f; score.qp_zp = 0;
        kpt.features = 51; kpt.height = side; kpt.width = side;
        kpt.format_type = kHailoFormatUint16;
        kpt.qp_scale = 0.01f; kpt.qp_zp = 0;
        infos.push_back(box);
        infos.push_back(score);
        infos.push_back(kpt);
    }
    return infos;
}

std::vector<std::vector<uint8_t>> pose_bytes(const LevelBytes& s80) {
    LevelBytes zero;
    zero.box.assign(64 * (20 * 20), 0);
    zero.score.assign(20 * 20, 0);
    zero.kpt.assign(51 * (20 * 20), 0);
    LevelBytes s40 = zero;
    s40.box.resize(64 * (40 * 40));
    s40.score.resize(40 * 40);
    s40.kpt.resize(51 * (40 * 40));

    auto to_bytes_u8 = [](const std::vector<uint8_t>& v) {
        return v;
    };
    auto to_bytes_u16 = [](const std::vector<uint16_t>& v) {
        std::vector<uint8_t> out(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            out[i * 2] = static_cast<uint8_t>(v[i] & 0xff);
            out[i * 2 + 1] = static_cast<uint8_t>(v[i] >> 8);
        }
        return out;
    };

    std::vector<std::vector<uint8_t>> out;
    out.push_back(to_bytes_u8(zero.box));      // 20: box
    out.push_back(to_bytes_u8(zero.score));    // 20: score
    out.push_back(to_bytes_u16(zero.kpt));     // 20: kpt
    out.push_back(to_bytes_u8(s40.box));       // 40: box
    out.push_back(to_bytes_u8(s40.score));     // 40: score
    out.push_back(to_bytes_u16(s40.kpt));      // 40: kpt
    out.push_back(to_bytes_u8(s80.box));       // 80: box
    out.push_back(to_bytes_u8(s80.score));     // 80: score
    out.push_back(to_bytes_u16(s80.kpt));      // 80: kpt
    return out;
}

const Detection* find_det(const DetectionResult& res, float cx, float cy) {
    for (const auto& d : res.dets)
        if (std::fabs(d.cx - cx) < 1e-4f && std::fabs(d.cy - cy) < 1e-4f)
            return &d;
    return nullptr;
}

}  // namespace

static void test_stub_backend_end_to_end() {
    // Cell (10,10): x0=52, x1=100, y0=76, y1=108 (dist l=4, r=2, t=1, b=3).
    const int bins_a[4] = {4, 1, 2, 3};
    // Cell (20,5): x0=132, x1=180, y0=36, y1=68.
    const int bins_b[4] = {4, 1, 2, 3};
    // Raw kx/ky in grid units, conf as the raw logit the decoder sigmoids.
    // cell = gy * side + gx: (10,10) and (20,5).
    LevelBytes lv = make_level(bins_a, 10 * kSide + 10, 5 * kSide + 20,
                               3.0f, 2.5f, 0.9f, 2.0f, 3.5f, 0.0f);
    hailo_stub::set_outputs(pose_infos(), pose_bytes(lv));

    std::string err;
    // fixed batch 8, so one infer() can carry both test frames.
    auto backend = create_backend(
        "hailo",
        R"({"model_path":"stub.hef","batch":{"mode":"8"}})", err);
    CHECK(backend != nullptr);
    const Caps caps = backend->caps();
    CHECK(caps.max_contexts == 1);
    CHECK(caps.max_batch == 8);
    CHECK(caps.keypoints == 17);
    CHECK(caps.exclusive_device);

    auto ctx = backend->create_context(0, err);
    CHECK(ctx != nullptr);
    ctx = nullptr;
    CHECK(backend->create_context(1, err) == nullptr);  // max_contexts is 1

    ctx = backend->create_context(0, err);
    CHECK(ctx != nullptr);

    // Two RGB888 frames on the model canvas.
    std::vector<std::vector<uint8_t>> pixels(2, std::vector<uint8_t>(640 * 640 * 3, 12));
    std::vector<FrameBuf> frames(2);
    std::vector<FrameBuf*> frame_ptrs(2);
    for (size_t i = 0; i < 2; ++i) {
        frames[i].w = 640;
        frames[i].h = 640;
        frames[i].stride = 640 * 3;
        frames[i].fmt = PixFmt::RGB888;
        frames[i].mem = Mem::Host;
        frames[i].host = pixels[i].data();
        frame_ptrs[i] = &frames[i];
    }
    std::vector<DetectionResult> results(2);
    CHECK(ctx->infer(frame_ptrs.data(), 2, 0.25f, 0.7f, results.data(), err) == 0);

    // Hand-computed boxes (model-canvas normalised): dist l=4 r=2 t=1 b=3.
    // cell (10,10): x0=52 x1=100 y0=76 y1=108 -> cx=76 cy=92 w=48 h=32
    const float eps = 1e-4f;
    const Detection* a = find_det(results[0], 76.0f / 640, 92.0f / 640);
    CHECK(a != nullptr);
    CHECK_NEAR(a->w, 48.0f / 640, eps);
    CHECK_NEAR(a->h, 32.0f / 640, eps);
    CHECK_NEAR(a->score, 220.0f / 255.0f, eps);
    CHECK(a->kpt_count == 17);
    // cell (20,5): x0=132 x1=180 y0=36 y1=68 -> cx=156 cy=52
    const Detection* b = find_det(results[0], 156.0f / 640, 52.0f / 640);
    CHECK(b != nullptr);

    // Keypoints: shared decoder form (2*k + g - 0.5) * stride.
    // det a (cell 10,10), kpt0: (2*3.0 + 10 - 0.5)*8 = 124; kpt1: 108.
    CHECK_NEAR(results[0].kpts[a->kpt_offset + 0].x, 124.0f / 640, eps);
    CHECK_NEAR(results[0].kpts[a->kpt_offset + 0].y, 116.0f / 640, eps);
    CHECK_NEAR(results[0].kpts[a->kpt_offset + 0].conf,
               1.0f / (1.0f + std::exp(-0.9f)), 1e-4);
    CHECK_NEAR(results[0].kpts[a->kpt_offset + 1].x, 108.0f / 640, eps);
    CHECK_NEAR(results[0].kpts[a->kpt_offset + 1].y, 132.0f / 640, eps);
    CHECK_NEAR(results[0].kpts[a->kpt_offset + 1].conf, 0.5f, 1e-4);
    // det b (cell 20,5), kpt0: (2*3.0 + 20 - 0.5)*8 = 204; (2*2.5 + 5 - 0.5)*8 = 76.
    CHECK_NEAR(results[0].kpts[b->kpt_offset + 0].x, 204.0f / 640, eps);
    CHECK_NEAR(results[0].kpts[b->kpt_offset + 0].y, 76.0f / 640, eps);

    // Both frame slots decoded (the stub mirrors its tensors per slot).
    CHECK(find_det(results[1], 76.0f / 640, 92.0f / 640) != nullptr);

    // Batch-size and frame-format guards.
    CHECK(ctx->infer(frame_ptrs.data(), 9, 0.25f, 0.7f, results.data(), err) != 0);
    frames[0].h = 480;
    CHECK(ctx->infer(frame_ptrs.data(), 1, 0.25f, 0.7f, results.data(), err) != 0);
    frames[0].h = 640;

    hailo_stub::clear_outputs();

    // Empty stub output decodes to zero detections.
    std::vector<DetectionResult> one(1);
    CHECK(ctx->infer(frame_ptrs.data(), 1, 0.25f, 0.7f, one.data(), err) == 0);
    CHECK(one[0].dets.empty());
}

static void test_backend_batch_config() {
    std::string err;
    // auto + streams=4 -> batch 4 (chooseBatch).
    auto backend = create_backend(
        "hailo",
        R"({"model_path":"stub.hef","batch":{"mode":"auto","streams":4}})", err);
    CHECK(backend != nullptr);
    CHECK(backend->caps().max_batch == 4);

    // mode off -> batch 1.
    backend = create_backend(
        "hailo", R"({"model_path":"stub.hef","batch":{"mode":"off"}})", err);
    CHECK(backend != nullptr);
    CHECK(backend->caps().max_batch == 1);

    // Invalid mode fails the factory.
    err.clear();
    backend = create_backend(
        "hailo", R"({"model_path":"stub.hef","batch":{"mode":"3"}})", err);
    CHECK(backend == nullptr);
    CHECK(!err.empty());

    // Invalid rtsp codec fails the factory.
    err.clear();
    backend = create_backend(
        "hailo",
        R"({"model_path":"stub.hef","rtsp":{"codec":"mpeg4"}})", err);
    CHECK(backend == nullptr);
    CHECK(!err.empty());

    // model_sha256 mismatch refuses before any device access.
    err.clear();
    backend = create_backend(
        "hailo",
        R"({"model_path":"stub.hef","model_sha256":"zz"})", err);
    CHECK(backend == nullptr);
    CHECK(!err.empty());
}

int main() {
    test_batch_policy();
    test_frame_batcher();
    test_runner_lifecycle();
    test_runtime_config();
    test_stub_backend_end_to_end();
    test_backend_batch_config();
    std::printf("test_hailo_backend: all checks passed\n");
    return 0;
}
