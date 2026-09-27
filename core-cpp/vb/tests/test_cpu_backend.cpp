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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "check.h"
#include "cpu_backend.h"
#include "dev_tensor.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/post.h"

#if defined(VB_WITH_GST)
#include "sources/gst_source.h"
#endif

namespace {

double mono_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch())
        .count();
}

#if defined(VB_WITH_GST)
// Writes a tiny Matroska clip with the GStreamer command line tool so the uri
// code path can be exercised end to end (the test target has no GStreamer
// headers of its own). False when the tool is unavailable or failed.
bool make_test_clip(const std::string& path) {
    ::unlink(path.c_str());
    const std::string cmd =
        "gst-launch-1.0 -q videotestsrc num-buffers=30 ! "
        "video/x-raw,format=I420,width=64,height=48 ! matroskamux ! filesink "
        "location=" +
        path + " >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0) return false;
    std::ifstream f(path, std::ios::binary);
    return f.good();
}
#endif

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

        // Resource bound: a model file larger than kCpuMaxModelBytes is
        // refused before it is read (sparse file, so this stays cheap).
        char big_path[] = "/tmp/vb_big_model_XXXXXX";
        int bfd = ::mkstemp(big_path);
        CHECK(bfd >= 0);
        CHECK(::ftruncate(bfd, static_cast<off_t>(vb::kCpuMaxModelBytes + 1)) == 0);
        ::close(bfd);
        vb::Json bigj;
        bigj["model_path"] = std::string(big_path);
        err.clear();
        CHECK(vb::make_cpu_backend(vb::json_dump(bigj), err) == nullptr);
        CHECK(err.find("too large") != std::string::npos);
        ::unlink(big_path);

        // §6.12 raw passthrough over a real ORT tensor: exactly one tensor,
        // copied at its true element width. The decoder path used to prepend
        // output_name_ to the complete output list, so output 0 was fetched
        // (and copied) twice.
        {
            vb::Json rj = bj;
            rj["decoder"] = vb::json_parse(R"({"type":"raw"})");
            err.clear();
            auto rawb = vb::make_cpu_backend(vb::json_dump(rj), err);
            if (!rawb) {
                std::fprintf(stderr, "raw backend: %s\n", err.c_str());
                return 1;
            }
            auto rctx = rawb->create_context(0, err);
            CHECK(rctx != nullptr);
            vb::DetectionResult rres;
            const vb::FrameBuf* one[1] = {&frame};
            CHECK(rctx->infer(one, 1, 0.25f, 0.5f, &rres, err) == 0);
            CHECK(rres.dets.empty());  // raw: no detections, §6.12
            auto* rsrc = dynamic_cast<vb::RawTensorSource*>(rctx.get());
            CHECK(rsrc != nullptr);
            std::vector<vb::DevTensor> ts;
            CHECK(rsrc->last_raw_tensors(ts));
            CHECK(ts.size() == 1);  // not the duplicated first output
            CHECK(ts[0].dtype == 0);  // VBT1 f32
            // [1, A, 5 + C] with the batch dimension dropped.
            CHECK(ts[0].dims.size() == 2);
            CHECK(ts[0].dims[0] == fx.at("n_anchors").get<int>());
            CHECK(ts[0].dims[1] == fx.at("n_cls").get<int>() + 5);
            const size_t expect_elems = static_cast<size_t>(fx.at("n_anchors").get<int>()) *
                                        (fx.at("n_cls").get<int>() + 5);
            CHECK(ts[0].data.size() == expect_elems * sizeof(float));
            std::printf("B: raw passthrough carries %zu tensor(s), %zu bytes\n",
                        ts.size(), ts[0].data.size());
        }

        // Digest: a configured model_sha256 must match the bytes ORT loads.
        vb::Json sj = bj;
        sj["model_sha256"] = std::string(64, '0');
        err.clear();
        CHECK(vb::make_cpu_backend(vb::json_dump(sj), err) == nullptr);
        CHECK(err.find("sha256 mismatch") != std::string::npos);
        sj["model_sha256"] = backend->model_sha256();
        err.clear();
        auto hashed = vb::make_cpu_backend(vb::json_dump(sj), err);
        if (!hashed) std::fprintf(stderr, "sha256 accept: %s\n", err.c_str());
        CHECK(hashed != nullptr);
        vb::Json bad = bj;
        bad["model_sha256"] = "not-a-digest";
        err.clear();
        CHECK(vb::make_cpu_backend(vb::json_dump(bad), err) == nullptr);
        std::printf("B: oversized model and wrong/malformed sha256 refused\n");
    }

#if defined(VB_WITH_GST)
    // ---- Part C1: dev pipeline (launch string) on a bounded videotestsrc ----
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
        int frames = 0, eos = 0, idle = 0;
        vb::FrameBuf f;
        uint64_t expect_seq = 0;
        for (int i = 0; i < 60; ++i) {
            int rc = src->read(f, 2000);
            if (rc == 0) ++idle;
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
        // Regression: try_pull_sample returns NULL *immediately* once the
        // appsink is EOS, so a read() that only consulted the bus could answer
        // "no frame yet" in a tight loop while the EOS message was still in
        // flight. EOS must be reported as soon as the sink says so, not after
        // the caller burns its budget: at most a couple of idle polls.
        CHECK(idle <= 3);
        src->close();
        CHECK(std::string(src->decode_path()) == "sw");
        std::printf("C1: gst source delivered %d frames + EOS\n", frames);
    }

    // ---- Part C2: C1 — a launch string is a dev-mode affordance ----
    {
        std::string err;
        vb::StreamSpec spec;
        spec.url = "videotestsrc num-buffers=5";
        ::setenv("VB_PRODUCTION", "1", 1);
        CHECK(vb::make_gst_source(spec, err) == nullptr);
        CHECK(!err.empty());
        ::unsetenv("VB_PRODUCTION");
        auto dev_ok = vb::make_gst_source(spec, err);
        CHECK(dev_ok != nullptr);

        // Unknown schemes are refused in every mode.
        vb::StreamSpec evil;
        evil.url = "evil://cam/s";
        err.clear();
        CHECK(vb::make_gst_source(evil, err) == nullptr);
        CHECK(!err.empty());
        // The url is applied as a property: a uri that cannot decode must fail
        // open() itself, not report success and fail later through reconnect.
        vb::StreamSpec missing;
        missing.url = "file:///nonexistent/vb_no_such_clip.mkv";
        err.clear();
        auto dead = vb::make_gst_source(missing, err);
        CHECK(dead != nullptr);
        if (dead) {
            const double t0 = mono_s();
            const bool opened = dead->open(err);
            const double dt = mono_s() - t0;
            if (opened) {
                std::fprintf(stderr, "undecodable uri reported open\n");
                return 1;
            }
            CHECK(!err.empty());
            CHECK(dt < vb::kGstOpenTimeoutS + 2.0);  // bounded, not a hang
            dead->close();
        }
        std::printf("C2: production refuses launch strings; open() fails on a dead uri\n");
    }

    // ---- Part C3: uri path end to end (uridecodebin + property url) ----
    {
        const std::string clip = "/tmp/vb_gst_source_test.mkv";
        if (!make_test_clip(clip)) {
            std::printf("C3: SKIP (gst-launch-1.0 unavailable, no clip to read)\n");
        } else {
            std::string err;
            vb::StreamSpec spec;
            spec.url = "file://" + clip;
            auto src = vb::make_gst_source(spec, err);
            CHECK(src != nullptr);
            if (!src) {
                std::fprintf(stderr, "uri source: %s\n", err.c_str());
                return 1;
            }
            CHECK(src->open(err));
            int frames = 0;
            vb::FrameBuf f;
            for (int i = 0; i < 200 && frames < 30; ++i) {
                int rc = src->read(f, 2000);
                if (rc == 1) {
                    CHECK(f.w == 64 && f.h == 48 && f.fmt == vb::PixFmt::RGB888);
                    ++frames;
                } else if (rc == -1) {
                    break;
                }
            }
            CHECK(frames >= 1);
            src->close();
            ::unlink(clip.c_str());
            std::printf("C3: uri source delivered %d frames\n", frames);
        }
    }
#endif

    std::printf("cpu_backend: all OK\n");
    return 0;
}
