// CPU backend tests (spec BASE-1 §8 M1.9).
//
//   test_cpu_backend <fixtures_dir> <const.onnx>
//
// Part A: yolox_decode + nms against contracts/fixtures/vb/yolox_out_case1
// (hand-calculated, tolerance 1e-4, no ORT required).
// Part B: the full CPU backend over the constant-output ONNX graph generated
// by core-py/vision_base/tests/gen_const_onnx.py — detections must equal the
// same fixture.
// Part C (VB_WITH_GST): the GStreamer source on a videotestsrc pipeline.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "cpu_backend.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/post.h"

#if defined(VB_WITH_GST)
#include "sources/gst_source.h"
#endif

namespace {

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

vb::Json read_json(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return vb::json_parse(ss.str());
}

void check_same_as_fixture(const std::vector<vb::Detection>& dets, const vb::Json& fx) {
    const auto& want = fx.at("expected_dets");
    CHECK(dets.size() == want.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        const vb::Detection& d = dets[i];
        CHECK_NEAR(d.cx, want[i].at("cx").get<double>(), 1e-4);
        CHECK_NEAR(d.cy, want[i].at("cy").get<double>(), 1e-4);
        CHECK_NEAR(d.w, want[i].at("w").get<double>(), 1e-4);
        CHECK_NEAR(d.h, want[i].at("h").get<double>(), 1e-4);
        CHECK_NEAR(d.score, want[i].at("score").get<double>(), 1e-4);
        CHECK(d.class_id == want[i].at("class_id").get<int>());
    }
}

// Deterministic RGB frame (same pattern idea as the synthetic source).
vb::FrameBuf make_frame(int w, int h, uint64_t seq) {
    auto px = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < px->size(); ++i)
        (*px)[i] = static_cast<uint8_t>((seq * 31 + i) & 0xff);
    vb::FrameBuf f;
    f.seq = seq;
    f.wall_ms = 0;
    f.t_mono_s = static_cast<double>(seq) / 15.0;
    f.w = w;
    f.h = h;
    f.stride = w * 3;
    f.fmt = vb::PixFmt::RGB888;
    f.mem = vb::Mem::Host;
    f.host = px->data();
    f.full_w = w;
    f.full_h = h;
    f.hold = px;
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_cpu_backend <fixtures_dir> <const.onnx>\n");
        return 2;
    }
    const std::string fdir = argv[1];
    const std::string const_onnx = argv[2];

    vb::Json fx = read_json(fdir + "/yolox_out_case1.json");
    std::vector<uint8_t> raw = read_file_bytes(fdir + "/yolox_out_case1.bin");
    if (raw.empty()) {
        std::fprintf(stderr, "cannot read %s/yolox_out_case1.bin\n", fdir.c_str());
        return 1;
    }

    // ---- Part A: decode + NMS against the hand-calculated fixture ----
    {
        const float* out = reinterpret_cast<const float*>(raw.data());
        std::vector<vb::Detection> dets;
        vb::yolox_decode(out, fx.at("n_anchors").get<int>(), fx.at("n_cls").get<int>(),
                         fx.at("model_w").get<int>(), fx.at("model_h").get<int>(),
                         fx.at("score").get<float>(), dets);
        CHECK(dets.size() == 3);  // anchor 3 passes 0.25, removed by NMS below
        std::vector<vb::Keypoint> kpts;
        vb::nms(dets, kpts, fx.at("nms_iou").get<float>(), fx.at("class_aware").get<bool>());
        check_same_as_fixture(dets, fx);
        std::printf("A: yolox_decode + nms match fixture (%zu dets)\n", dets.size());
    }

    // ---- Part B: full CPU backend over the constant-output graph ----
    {
        std::ifstream f(const_onnx);
        if (!f) {
            // CMake generates const.onnx at build time when `uv` is on PATH
            // (backends/cpu/CMakeLists.txt); this is the fallback for a
            // uv-less build tree or a manually-invoked binary. ctest is
            // configured with SKIP_RETURN_CODE 77 for this test so a missing
            // file reports SKIP, not FAIL.
            std::fprintf(stderr, "SKIP: missing %s (run: uv run python "
                                 "core-py/vision_base/tests/gen_const_onnx.py %s)\n",
                         const_onnx.c_str(), const_onnx.c_str());
            return 77;
        }
        std::string err;
        vb::Json bj;
        bj["model_path"] = const_onnx;
        auto backend = vb::make_cpu_backend(vb::json_dump(bj), err);
        CHECK(backend != nullptr);
        if (!backend) {
            std::fprintf(stderr, "backend: %s\n", err.c_str());
            return 1;
        }
        CHECK(std::string(backend->name()) == "cpu");
        auto hw = backend->model_hw();
        CHECK(hw.first == fx.at("model_w").get<int>());
        CHECK(hw.second == fx.at("model_h").get<int>());
        CHECK(backend->model_sha256().size() == 64);
        CHECK(backend->caps().max_batch == 1);

        auto ctx = backend->create_context(0, err);
        CHECK(ctx != nullptr);
        vb::FrameBuf frame = make_frame(320, 240, 7);
        const vb::FrameBuf* frames[1] = {&frame};
        vb::DetectionResult res;
        int rc = ctx->infer(frames, 1, fx.at("score").get<float>(),
                             fx.at("nms_iou").get<float>(), &res, err);
        if (rc != 0 || !err.empty()) std::fprintf(stderr, "infer: rc=%d err=%s\n", rc, err.c_str());
        CHECK(rc == 0);
        check_same_as_fixture(res.dets, fx);
        CHECK(res.geom.model_w == hw.first && res.geom.model_h == hw.second);
        CHECK(res.preprocess_ms >= 0 && res.inference_ms >= 0 && res.postprocess_ms >= 0);
        std::printf("B: cpu backend on const graph matches fixture "
                    "(sha256 %.8s..., preprocess %.1f ms)\n",
                    backend->model_sha256().c_str(), res.preprocess_ms);

        // Batch > caps.max_batch is rejected.
        vb::FrameBuf f2 = make_frame(320, 240, 8);
        const vb::FrameBuf* two[2] = {&frame, &f2};
        CHECK(ctx->infer(two, 2, 0.25f, 0.5f, &res, err) != 0);

        // Bad model path / missing model_path surface as factory errors.
        CHECK(vb::make_cpu_backend("{\"model_path\":\"/nonexistent.onnx\"}", err) == nullptr);
        CHECK(!err.empty());
        CHECK(vb::make_cpu_backend("{}", err) == nullptr);
        CHECK(!err.empty());

        // Registry: the cpu backend self-registered (register.cpp).
        auto r = vb::create_backend("cpu", vb::json_dump(bj), err);
        if (!r) std::fprintf(stderr, "registry: %s\n", err.c_str());
        CHECK(r != nullptr);
    }

#if defined(VB_WITH_GST)
    // ---- Part C: GStreamer source on a bounded videotestsrc pipeline ----
    {
        std::string err;
        vb::StreamSpec spec;
        spec.url = "videotestsrc num-buffers=20 ! video/x-raw,width=64,height=48";
        auto src = vb::make_gst_source(spec, err);
        CHECK(src != nullptr);
        if (!src) {
            std::fprintf(stderr, "gst source: %s\n", err.c_str());
            return 1;
        }
        CHECK(src->open(err));
        int frames = 0, eos = 0;
        vb::FrameBuf f;
        uint64_t expect_seq = 0;
        for (int i = 0; i < 60; ++i) {
            int rc = src->read(f, 2000);
            if (rc == 1) {
                CHECK(f.w == 64 && f.h == 48 && f.stride == 64 * 3);
                CHECK(f.fmt == vb::PixFmt::RGB888 && f.mem == vb::Mem::Host);
                CHECK(f.host != nullptr);
                CHECK(f.seq == expect_seq++);
                ++frames;
            } else if (rc == -1) {
                eos = 1;
                // After EOS every read must keep reporting stream loss (-1).
                for (int k = 0; k < 3; ++k) CHECK(src->read(f, 100) == -1);
                break;
            }
        }
        CHECK(frames >= 3);
        CHECK(eos == 1);
        src->close();
        CHECK(std::string(src->decode_path()) == "sw");
        std::printf("C: gst source delivered %d frames + EOS\n", frames);
    }
#endif

    std::printf("cpu_backend: all OK\n");
    return 0;
}
