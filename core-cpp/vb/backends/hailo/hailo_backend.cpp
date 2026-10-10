// Hailo (Pi 5 + Hailo-8) inference backend (spec BASE-1 §M2.3).
//
// One BatchedHailoRunner per process (one VDevice, exclusive) backs every
// InferenceContext; caps.max_batch comes from the copied batch policy. Frames
// arrive as tightly packed RGB888 model canvas from hailo_source.cpp; the
// context copies them into the runner's pre-allocated batch slots (fall
// newSample did the same copy), zero-pads a partial batch, runs once, and
// hands the dequantised outputs to the §6.11 yolo_pose decoder as TensorView.
//
// Output layouts (Hailo Model Zoo yolov8s_pose, one set per stride): box
// [H*W][64] UINT8, score [H*W][1] UINT8, kpt [H*W][51] UINT16 — cell-major,
// features innermost (the HEF reports them as FCR/NHWC(20x20x64) etc.). The
// shared split decoder consumes planar [C][H][W] with box 4*reg_max=64, cls
// nc=1 and kpt 3*keypoints=51 (channel j*3: x, y, conf), so the dequantiser
// below transposes as it converts width.
//
// The cell-major order is not inferred from the vstream names: both shipped
// decoders for this HEF index it that way —
// fall platforms/rpi-hailo/src/hailo_pose_decoder.cpp:29-45 reads
// `cell*64 + q*16 + n` and `cell*51 + j*3`, and the Python reference
// evaluation/reports/rpi-hailo8-python-vs-cpp-20260926/round1/bench_b.py:53-65
// reshapes the same buffers as (-1, 64) and (-1, 51). Feeding the raw buffer
// as planar made the score plane look right (one feature is layout-invariant)
// while the box and keypoint planes decoded to garbage, which is what the
// M2.3 parity run measured.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "batch_policy.h"
#include "batched_hailo_runner.h"
#include "hailo_backend.h"
#include "hailo_source.h"
#include "model_sha256.h"

#include "vb/decoder.h"
#include "vb/json.h"
#include "vb/post.h"

namespace vb {
namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

bool is_sha256_hex(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F')))
            return false;
    return true;
}

std::string lower_ascii(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

bool read_model_file(const std::string& path, std::vector<uint8_t>& bytes,
                     std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "hailo backend: cannot open model: " + path;
        return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(f),
                 std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        err = "hailo backend: empty model file: " + path;
        return false;
    }
    return true;
}

// Dequantised scratch for one frame's 9 outputs, in TensorView-ready form.
struct FrameViews {
    std::vector<float> data;
    std::vector<TensorView> views;
};

void dequantize(const std::vector<RawTensor>& tensors, FrameViews& dst) {
    size_t total = 0;
    for (const auto& t : tensors)
        total += static_cast<size_t>(t.info.features) * t.info.height * t.info.width;
    dst.data.assign(total, 0.0f);
    dst.views.clear();
    dst.views.reserve(tensors.size());
    size_t off = 0;
    for (const auto& t : tensors) {
        const size_t feats = static_cast<size_t>(t.info.features);
        const size_t cells = static_cast<size_t>(t.info.height) * t.info.width;
        const size_t elems = feats * cells;
        float* out = dst.data.data() + off;
        const float scale = t.info.qp_scale;
        const float zpf = static_cast<float>(t.info.qp_zp);
        // cell-major -> planar: out[c][cell] = in[cell][c]. A one-feature head
        // (the score plane) is unchanged by the transpose.
        if (t.info.format_type == kHailoFormatUint16) {
            const auto* in = reinterpret_cast<const uint16_t*>(t.data);
            for (size_t cell = 0; cell < cells; ++cell)
                for (size_t c = 0; c < feats; ++c)
                    out[c * cells + cell] =
                        (static_cast<float>(in[cell * feats + c]) - zpf) * scale;
        } else {
            for (size_t cell = 0; cell < cells; ++cell)
                for (size_t c = 0; c < feats; ++c)
                    out[c * cells + cell] =
                        (static_cast<float>(t.data[cell * feats + c]) - zpf) * scale;
        }
        TensorView v;
        v.data = out;
        v.count = elems;
        v.dims = {t.info.features, t.info.height, t.info.width};
        dst.views.push_back(std::move(v));
        off += elems;
    }
}

class HailoContext : public InferenceContext {
public:
    HailoContext(BatchedHailoRunner* runner, Decoder* decoder, int model_w,
                 int model_h)
        : runner_(runner), decoder_(decoder), model_w_(model_w), model_h_(model_h) {}

    int infer(const FrameBuf* const* frames, size_t n, float score, float nms_th,
              DetectionResult* out, std::string& err) override {
        if (n == 0 || static_cast<int>(n) > runner_->batch_size()) {
            err = "hailo backend: batch size " + std::to_string(n) +
                  " outside 1.." + std::to_string(runner_->batch_size());
            return -1;
        }
        const size_t input_bytes = static_cast<size_t>(model_w_) * model_h_ * 3;
        const double t0 = now_ms();
        std::vector<BatchFrame> batch(n);
        for (size_t i = 0; i < n; ++i) {
            const FrameBuf& f = *frames[i];
            if (f.mem != Mem::Host || f.fmt != PixFmt::RGB888 || f.host == nullptr ||
                f.w != model_w_ || f.h != model_h_) {
                err = "hailo backend requires RGB888 host frames on the model "
                      "canvas (got mem=" +
                      std::to_string(static_cast<int>(f.mem)) + " fmt=" +
                      std::to_string(static_cast<int>(f.fmt)) + " " +
                      std::to_string(f.w) + "x" + std::to_string(f.h) + ")";
                return -1;
            }
            if (static_cast<size_t>(f.stride) != static_cast<size_t>(f.w) * 3) {
                err = "hailo backend requires tightly packed RGB rows";
                return -1;
            }
            batch[i].stream = static_cast<int>(i);
            batch[i].seq = f.seq;
            batch[i].rgb.assign(f.host, f.host + input_bytes);
        }
        const double t1 = now_ms();
        std::vector<std::vector<RawTensor>> tensors;
        if (runner_->infer(batch, n, tensors, err) != 0) return -1;
        const double t2 = now_ms();
        for (size_t i = 0; i < n; ++i) {
            DetectionResult& res = out[i];
            res.dets.clear();
            res.kpts.clear();
            dequantize(tensors[i], views_);
            if (decoder_) {
                if (!decoder_->decode(views_.views.data(), views_.views.size(),
                                      model_w_, model_h_, score, nms_th, res, err))
                    return -1;
            } else {
                // No decoder configured: the raw pose head is not a YOLOX
                // output; refusing beats emitting garbage.
                err = "hailo backend requires backend.decoder (yolo_pose)";
                return -1;
            }
            res.geom = LetterboxGeom::fit(model_w_, model_h_, model_w_, model_h_,
                                          Align::Center);
            res.preprocess_ms = static_cast<float>(t1 - t0);
            res.inference_ms = static_cast<float>(t2 - t1);
            res.postprocess_ms = static_cast<float>(now_ms() - t2);
        }
        return 0;
    }

private:
    BatchedHailoRunner* runner_;
    Decoder* decoder_;
    int model_w_, model_h_;
    FrameViews views_;  // per-context scratch (one context per worker thread)
};

class HailoBackend : public Backend {
public:
    explicit HailoBackend(const std::string& json, std::string& err) {
        Json j;
        try {
            j = json.empty() ? Json::object() : json_parse(json);
        } catch (const std::exception& e) {
            err = std::string("hailo backend json: ") + e.what();
            return;
        }
        if (!j.contains("model_path")) {
            err = "hailo backend requires backend.model_path";
            return;
        }
        model_path_ = j.at("model_path").get<std::string>();
        const std::string want_sha = j.value("model_sha256", std::string());
        if (!want_sha.empty() && !is_sha256_hex(want_sha)) {
            err = "backend.model_sha256 must be 64 hex characters";
            return;
        }

        auto dj = j.find("decoder");
        if (dj != j.end() && dj->is_object() && !dj->empty()) {
            decoder_ = make_decoder(dj->dump(), err);
            if (!decoder_) return;
        } else {
            // The pose HEF family this backend validates is yolo_pose.
            decoder_ = make_decoder(R"({"type":"yolo_pose","keypoints":17})", err);
            if (!decoder_) return;
        }

        // Batch policy (copied from fall): mode + stream count decide the
        // configured HEF batch, which is also Caps.max_batch.
        BatchConfig batch{BatchMode::Auto, 1};
        int streams = 1;
        batch_wait_ms_ = 20;
        auto bj = j.find("batch");
        if (bj != j.end() && bj->is_object()) {
            const std::string mode = bj->value("mode", std::string("auto"));
            try {
                batch = parseBatchMode(mode);
            } catch (const std::exception& e) {
                err = std::string("backend.batch.mode: ") + e.what();
                return;
            }
            if (bj->contains("streams")) {
                const int64_t v = bj->at("streams").get<int64_t>();
                if (v < 1 || v > 64) {
                    err = "backend.batch.streams must be in 1..64";
                    return;
                }
                streams = static_cast<int>(v);
            }
            if (bj->contains("wait_ms")) {
                const int64_t v = bj->at("wait_ms").get<int64_t>();
                if (v < 0 || v > 1000) {
                    err = "backend.batch.wait_ms must be in 0..1000";
                    return;
                }
                batch_wait_ms_ = static_cast<int>(v);
            }
        }

        // rtsp settings for the frame source (fall env -> backend json).
        auto rj = j.find("rtsp");
        if (rj != j.end() && rj->is_object()) {
            if (rj->contains("latency_ms")) {
                const int64_t v = rj->at("latency_ms").get<int64_t>();
                if (v < 0) {
                    err = "backend.rtsp.latency_ms must be >= 0";
                    return;
                }
                rtsp_.latency_ms = static_cast<int>(v);
            }
            if (rj->contains("drop_on_latency"))
                rtsp_.drop_on_latency = rj->at("drop_on_latency").get<bool>();
            if (rj->contains("codec")) rtsp_.codec = rj->at("codec").get<std::string>();
        }
        try {
            rtsp_.validate();
        } catch (const std::exception& e) {
            err = std::string("backend.rtsp: ") + e.what();
            return;
        }

        int max_contexts = 1;
        if (j.contains("max_contexts")) {
            const int64_t v = j["max_contexts"].get<int64_t>();
            if (v < 1 || v > 64) {
                err = "backend.max_contexts must be in 1..64";
                return;
            }
            max_contexts = static_cast<int>(v);
        }

        // Model digest before the session: a mismatch must refuse without
        // touching the device.
        std::vector<uint8_t> bytes;
        if (read_model_file(model_path_, bytes, err)) {
            sha256_ = hailo_detail::sha256_hex(bytes);
            if (!want_sha.empty() && lower_ascii(want_sha) != sha256_) {
                err = "model sha256 mismatch: expected " + lower_ascii(want_sha) +
                      ", got " + sha256_;
                return;
            }
        } else if (!want_sha.empty()) {
            return;  // read_model_file already set err
        } else {
            err.clear();  // stub builds may name a path that is not a file
        }

        // Hailo session (exclusive VDevice): created here, shared by contexts.
        try {
            HefContextInfo info = BatchedHailoRunner::inspectHef(model_path_);
            BatchDecision decision = chooseBatch(batch, info.network_groups,
                                                 info.multi_context, streams);
            batch_size_ = decision.batch_size;
            batcher_ = std::make_unique<FrameBatcher>(1, batch_size_, batch_wait_ms_, 2);
            // The worker path stays available (start/stop); the backend uses
            // the synchronous infer() entry, so the batcher is never fed.
            runner_ = std::make_unique<BatchedHailoRunner>(
                model_path_, batch_size_, *batcher_, HailoResultHandler(),
                [](const std::string& e) {
                    std::fprintf(stderr, "hailo runner: %s\n", e.c_str());
                });
            model_w_ = 640;
            model_h_ = 640;  // the runner validates RGB UINT8 640x640 input
            std::fprintf(stderr,
                         "hailo backend: hef=%s groups=%d multi_context=%d "
                         "batch=%d streams=%d rtsp_codec=%s\n",
                         model_path_.c_str(), info.network_groups,
                         info.multi_context ? 1 : 0, batch_size_, streams,
                         rtsp_.codec.c_str());
        } catch (const std::exception& e) {
            err = std::string("hailo backend: ") + e.what();
            return;
        }

        max_contexts_ = max_contexts;
        ok_ = true;
    }

    bool ok() const { return ok_; }

    const char* name() const override { return "hailo"; }

    Caps caps() const override {
        Caps c;
        c.max_contexts = max_contexts_;
        c.max_batch = batch_size_;
        c.keypoints = decoder_ ? decoder_->keypoints() : 0;
        c.exclusive_device = true;
        return c;
    }

    std::pair<int, int> model_hw() const override { return {model_w_, model_h_}; }
    std::string model_sha256() const override { return sha256_; }

    std::unique_ptr<FrameSource> create_source(const StreamSpec& s,
                                               std::string& err) override {
#ifdef VB_WITH_GST
        return make_hailo_source(s, rtsp_, err);
#else
        (void)s;
        err = "hailo backend requires VB_WITH_GST=ON for frame sources";
        return nullptr;
#endif
    }

    std::unique_ptr<InferenceContext> create_context(int index,
                                                     std::string& err) override {
        if (index < 0 || index >= max_contexts_) {
            err = "hailo backend: context index " + std::to_string(index) +
                  " outside 0.." + std::to_string(max_contexts_ - 1);
            return nullptr;
        }
        // One exclusive runner serves every context; contexts are thin decode
        // adapters (the HEF batch was configured once, on the runner).
        return std::make_unique<HailoContext>(runner_.get(), decoder_.get(),
                                              model_w_, model_h_);
    }

private:
    std::string model_path_, sha256_;
    hailo_source::RtspSettings rtsp_{};
    std::unique_ptr<Decoder> decoder_;
    std::unique_ptr<FrameBatcher> batcher_;
    std::unique_ptr<BatchedHailoRunner> runner_;
    int batch_size_ = 1, batch_wait_ms_ = 20;
    int model_w_ = 0, model_h_ = 0, max_contexts_ = 1;
    bool ok_ = false;
};

}  // namespace

std::unique_ptr<Backend> make_hailo_backend(const std::string& backend_json,
                                            std::string& err) {
    err.clear();
    auto b = std::make_unique<HailoBackend>(backend_json, err);
    if (!err.empty() || !b->ok()) return nullptr;
    return b;
}

}  // namespace vb
