// Synthetic backend unit test (spec BASE-1 §8 M1.7: deterministic detections
// by frame number, configurable max_batch).
#include <thread>

#include "check.h"
#include "vb/backend.h"
#include "vb/json.h"

using namespace vb;

namespace {

FrameBuf make_frame(uint32_t stream, uint64_t seq, int w, int h) {
    FrameBuf f;
    f.stream_index = stream;
    f.seq = seq;
    f.wall_ms = static_cast<double>(seq) * 1000.0;
    f.t_mono_s = static_cast<double>(seq) / 10.0;
    f.w = w;
    f.h = h;
    f.stride = w * 3;
    static uint8_t dummy[16] = {0};
    f.host = dummy;
    f.full_w = w;
    f.full_h = h;
    return f;
}

}  // namespace

int main() {
    std::string err;
    auto b = create_backend("synthetic", R"({"max_batch":3,"boxes":2,"model_w":416,"model_h":416,"infer_ms":0})", err);
    CHECK(b != nullptr);
    CHECK_STREQ(b->name(), "synthetic");
    CHECK(b->caps().max_batch == 3);
    auto hw = b->model_hw();
    CHECK(hw.first == 416 && hw.second == 416);
    CHECK(!b->model_sha256().empty());

    auto ctx = b->create_context(0, err);
    CHECK(ctx != nullptr);

    FrameBuf f0 = make_frame(0, 5, 640, 480);
    FrameBuf f1 = make_frame(1, 5, 640, 480);
    const FrameBuf* frames[2] = {&f0, &f1};
    DetectionResult out[2];
    CHECK(ctx->infer(frames, 2, 0.35f, 0.45f, out, err) == 0);
    CHECK(out[0].dets.size() == 2);
    CHECK(out[1].dets.size() == 2);
    // Determinism: same seq -> identical boxes.
    for (size_t i = 0; i < out[0].dets.size(); ++i) {
        CHECK_NEAR(out[0].dets[i].cx, out[1].dets[i].cx, 1e-9);
        CHECK_NEAR(out[0].dets[i].cy, out[1].dets[i].cy, 1e-9);
    }
    DetectionResult again[1];
    const FrameBuf* f0p[1] = {&f0};
    CHECK(ctx->infer(f0p, 1, 0.35f, 0.45f, again, err) == 0);
    for (size_t i = 0; i < out[0].dets.size(); ++i) {
        CHECK_NEAR(out[0].dets[i].cx, again[0].dets[i].cx, 1e-9);
        CHECK_NEAR(out[0].dets[i].cy, again[0].dets[i].cy, 1e-9);
        CHECK_NEAR(out[0].dets[i].score, again[0].dets[i].score, 1e-9);
    }
    // Different seq -> different cx.
    FrameBuf f2 = make_frame(0, 6, 640, 480);
    const FrameBuf* f2p[1] = {&f2};
    DetectionResult out2[1];
    CHECK(ctx->infer(f2p, 1, 0.35f, 0.45f, out2, err) == 0);
    CHECK(std::fabs(out2[0].dets[0].cx - out[0].dets[0].cx) > 1e-6);
    // Geom letterboxes the source frame.
    CHECK(out[0].geom.src_w == 640 && out[0].geom.src_h == 480);
    CHECK(out[0].geom.model_w == 416 && out[0].geom.model_h == 416);

    // Unknown backend.
    CHECK(create_backend("nope", "{}", err) == nullptr);
    CHECK(err.find("unknown backend") == 0);

    // Frame source.
    StreamSpec spec;
    spec.index = 0;
    spec.url = "synthetic://?w=320&h=240&fps=500&boxes=3&fail_at_seq=4";
    auto src = b->create_source(spec, err);
    CHECK(src != nullptr);
    CHECK_STREQ(src->decode_path(), "synthetic");
    CHECK(src->open(err));
    FrameBuf f;
    int r = src->read(f, 100);
    CHECK(r == 1);
    CHECK(f.seq == 0 && f.w == 320 && f.h == 240 && f.fmt == PixFmt::RGB888);
    CHECK(f.host != nullptr && f.hold != nullptr);
    // seq 1..3 fine, seq 4 -> -1 once, then continues from 5.
    CHECK(src->read(f, 100) == 1);
    CHECK(src->read(f, 100) == 1);
    CHECK(src->read(f, 100) == 1);
    CHECK(src->read(f, 100) == -1);
    CHECK(src->read(f, 100) == 1);
    CHECK(f.seq == 5);
    src->close();

    std::printf("synthetic_backend: OK\n");
    return 0;
}
