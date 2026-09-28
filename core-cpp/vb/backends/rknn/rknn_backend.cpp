// RK3576 / RK3588 inference backend (spec BASE-1 §M2.1).
//
// Direct librknnrt C API: every InferenceContext owns one rknn_context created
// with the core mask for its index, its input memory is an rknn_create_mem()
// buffer that RGA fills from the decoded NV12 DMA-BUF, and every output is an
// rknn_create_mem() buffer set with rknn_set_io_mem(). No Python binding and no
// ctypes anywhere on this path.
//
// Outputs are handed to the §6.11 decoder as TensorView. The runtime header
// this repository ships declares TensorView as float data, so the adapter asks
// librknnrt for dequantised outputs (rknn_output.want_float) instead of
// exposing scale/zero_point; see the M2.1 report for that deviation.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "hybrid_rga_rknn.h"
#include "mpp_source.h"
#include "model_sha256.h"
#include "rknn_backend.h"

#include "vb/decoder.h"
#include "vb/json.h"
#include "vb/post.h"

namespace vb {
namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// rknn_core_mask bits (rknn_api.h): core 0 = 1, core 1 = 2, core 2 = 4.
constexpr uint32_t kCore0 = 1u << 0;
constexpr uint32_t kCore1 = 1u << 1;
constexpr uint32_t kCore2 = 1u << 2;

std::string soc_compatible() {
    std::ifstream f("/proc/device-tree/compatible", std::ios::binary);
    if (!f) return {};
    std::string s((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
    return s;
}

// Default core masks, one entry per context (§M2.1): RK3588 has three NPU
// cores, RK3576 two. An unknown SoC gets core 0 only, which is valid on every
// Rockchip NPU; `backend.core_masks` overrides this table.
std::vector<uint32_t> default_core_masks(const std::string& compatible) {
    if (compatible.find("rk3588") != std::string::npos)
        return {kCore0, kCore1, kCore2};
    if (compatible.find("rk3576") != std::string::npos) return {kCore0, kCore1};
    return {kCore0};
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
        err = "rknn backend: cannot open model: " + path;
        return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(f),
                 std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        err = "rknn backend: empty model file: " + path;
        return false;
    }
    return true;
}

class RknnContext : public InferenceContext {
public:
    RknnContext(RknnHybrid* hybrid, Decoder* decoder, int model_w, int model_h)
        : hybrid_(hybrid), decoder_(decoder), model_w_(model_w), model_h_(model_h) {}

    int infer(const FrameBuf* const* frames, size_t n, float score, float nms_th,
              DetectionResult* out, std::string& err) override {
        if (n != 1) {
            err = "rknn backend max_batch is 1";
            return -1;
        }
        const FrameBuf& f = *frames[0];
        if (f.mem != Mem::DmaBuf || f.fmt != PixFmt::NV12 || f.dmabuf_fd < 0) {
            err = "rknn backend requires NV12 DMA-BUF frames (got mem=" +
                  std::to_string(static_cast<int>(f.mem)) + " fmt=" +
                  std::to_string(static_cast<int>(f.fmt)) + " fd=" +
                  std::to_string(f.dmabuf_fd) + ")";
            return -1;
        }
        double rga_ms = 0, rknn_ms = 0;
        const int rc = hybrid_->infer_nv12_fd(f.dmabuf_fd, f.w, f.h, f.stride,
                                              &rga_ms, &rknn_ms, err);
        if (rc != 0) return rc;

        const double t_dec0 = now_ms();
        DetectionResult& res = out[0];
        res.geom = hybrid_->last_geom();
        res.dets.clear();
        res.kpts.clear();

        // §6.11: one TensorView per output, batch dimension dropped when 1.
        std::vector<TensorView> views;
        const auto& descs = hybrid_->outputs();
        const float* base = hybrid_->flat();
        size_t offset = 0;
        views.reserve(descs.size());
        for (const auto& d : descs) {
            TensorView v;
            v.data = base + offset;
            v.count = d.n_elems;
            size_t first = 0;
            if (!d.dims.empty() && d.dims[0] == 1) first = 1;
            v.dims.reserve(d.dims.size() - first);
            for (size_t i = first; i < d.dims.size(); ++i)
                v.dims.push_back(static_cast<int64_t>(d.dims[i]));
            views.push_back(std::move(v));
            offset += d.n_elems;
        }

        if (decoder_) {
            if (!decoder_->decode(views.data(), views.size(), model_w_, model_h_,
                                  score, nms_th, res, err))
                return -1;
        } else {
            // Same default as the M1.9 CPU backend: the YOLOX head on output 0.
            const TensorView& v = views[0];
            if (v.dims.size() != 2) {
                err = "rknn backend: expected a [A, 5+nc] or [5+nc, A] output, got " +
                      std::to_string(v.dims.size()) + " dims";
                return -1;
            }
            int anchors = 0, n_cls = 0;
            bool transposed = false;
            const int64_t d0 = v.dims[0], d1 = v.dims[1];
            if (d1 >= 6 && d1 <= 200) {
                anchors = static_cast<int>(d0);
                n_cls = static_cast<int>(d1 - 5);
            } else if (d0 >= 6 && d0 <= 200) {
                anchors = static_cast<int>(d1);
                n_cls = static_cast<int>(d0 - 5);
                transposed = true;
            }
            if (anchors <= 0 || n_cls <= 0) {
                err = "rknn backend: unexpected output shape";
                return -1;
            }
            if (transposed) {
                tmp_.resize(static_cast<size_t>(anchors) * (n_cls + 5));
                const size_t row = static_cast<size_t>(n_cls) + 5;
                for (int i = 0; i < anchors; ++i)
                    for (size_t c = 0; c < row; ++c)
                        tmp_[i * row + c] = v.data[c * anchors + i];
                yolox_decode(tmp_.data(), anchors, n_cls, model_w_, model_h_, score,
                             res.dets);
            } else {
                yolox_decode(v.data, anchors, n_cls, model_w_, model_h_, score,
                             res.dets);
            }
            nms(res.dets, res.kpts, nms_th, true);
        }
        const double t_dec1 = now_ms();
        res.preprocess_ms = static_cast<float>(rga_ms);
        res.inference_ms = static_cast<float>(rknn_ms);
        res.postprocess_ms = static_cast<float>(t_dec1 - t_dec0);
        return 0;
    }

private:
    RknnHybrid* hybrid_;
    InputSpec input_{};  // backend.input, else decoder default
    Decoder* decoder_;
    int model_w_, model_h_;
    std::vector<float> tmp_;  // transposed-decode scratch (one context per thread)
};

class RknnBackend : public Backend {
public:
    explicit RknnBackend(const std::string& json, std::string& err) {
        Json j;
        try {
            j = json.empty() ? Json::object() : json_parse(json);
        } catch (const std::exception& e) {
            err = std::string("rknn backend json: ") + e.what();
            return;
        }
        if (!j.contains("model_path")) {
            err = "rknn backend requires backend.model_path";
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
            // §6.11 M1.15: backend.decoder object selects the shared decoder.
            decoder_ = make_decoder(dj->dump(), err);
            if (!decoder_) return;
        } else if (dj != j.end() && dj->is_string()) {
            const std::string decoder = dj->get<std::string>();
            if (decoder != "yolox") {
                err = "rknn backend: unsupported decoder '" + decoder + "'";
                return;
            }
        }

        // backend.input (§6.11): what the model's input tensor expects. Absent,
        // it follows the decoder family's own convention — the frame format the
        // pipeline negotiated says nothing about the model.
        {
            const std::string dec_type =
                (dj != j.end() && dj->is_object())
                    ? dj->value("type", std::string("yolox"))
                    : "yolox";
            input_ = InputSpec::default_for_decoder(dec_type);
            auto ij = j.find("input");
            if (ij != j.end() && ij->is_object()) {
                if (ij->contains("color_order")) {
                    const std::string co = ij->at("color_order").get<std::string>();
                    if (co == "bgr") input_.color_order = ColorOrder::BGR;
                    else if (co == "rgb") input_.color_order = ColorOrder::RGB;
                    else { err = "backend.input.color_order must be bgr or rgb"; return; }
                }
                if (ij->contains("divide")) {
                    const double dv = ij->at("divide").get<double>();
                    if (!(dv > 0.0)) { err = "backend.input.divide must be > 0"; return; }
                    input_.divide = static_cast<float>(dv);
                }
            }
        }

        if (j.contains("core_masks")) {
            if (!j["core_masks"].is_array() || j["core_masks"].empty()) {
                err = "backend.core_masks must be a non-empty array";
                return;
            }
            for (const auto& m : j["core_masks"]) {
                if (!m.is_number_unsigned() && !m.is_number_integer()) {
                    err = "backend.core_masks entries must be integers";
                    return;
                }
                const int64_t v = m.get<int64_t>();
                if (v <= 0 || v > 0xffff) {
                    err = "backend.core_masks entries must be rknn_core_mask bits";
                    return;
                }
                core_masks_.push_back(static_cast<uint32_t>(v));
            }
        } else {
            compatible_ = soc_compatible();
            core_masks_ = default_core_masks(compatible_);
        }

        std::vector<uint8_t> bytes;
        if (!read_model_file(model_path_, bytes, err)) return;
        sha256_ = rknn_detail::sha256_hex(bytes);
        if (!want_sha.empty() && lower_ascii(want_sha) != sha256_) {
            err = "model sha256 mismatch: expected " + lower_ascii(want_sha) +
                  ", got " + sha256_;
            return;
        }

        // One throwaway context for the model geometry and output layout: the
        // runtime asks Backend::model_hw() before any context exists.
        std::string probe_err;
        auto probe = RknnHybrid::create(model_path_, core_masks_[0], probe_err);
        if (!probe) {
            err = "rknn backend: " + probe_err;
            return;
        }
        model_w_ = probe->model_width();
        model_h_ = probe->model_height();
        rknn_sdk_ = probe->sdk_version();
        // One startup line per process: which runtime the board actually
        // loaded, the model geometry, and the core mask each context will take.
        std::string masks;
        for (size_t i = 0; i < core_masks_.size(); ++i)
            masks += (i ? "," : "") + std::to_string(core_masks_[i]);
        std::fprintf(stderr,
                     "rknn backend: librknnrt %s model %dx%d outputs=%zu "
                     "core_masks=[%s] soc=%s\n",
                     rknn_sdk_.empty() ? "?" : rknn_sdk_.c_str(), model_w_, model_h_,
                     probe->outputs().size(), masks.c_str(),
                     compatible_.empty() ? "?" : compatible_.c_str());

        int max_contexts = static_cast<int>(core_masks_.size());
        if (j.contains("max_contexts")) {
            const int64_t v = j["max_contexts"].get<int64_t>();
            if (v <= 0 || v > 64) {
                err = "backend.max_contexts must be in 1..64";
                return;
            }
            max_contexts = static_cast<int>(v);
        }
        max_contexts_ = max_contexts;
        ok_ = true;
    }

    bool ok() const { return ok_; }

    const char* name() const override { return "rknn"; }

    Caps caps() const override {
        Caps c;
        c.max_contexts = max_contexts_;
        c.max_batch = 1;
        c.keypoints = decoder_ ? decoder_->keypoints() : 0;
        c.exclusive_device = false;
        return c;
    }

    std::pair<int, int> model_hw() const override { return {model_w_, model_h_}; }
    std::string model_sha256() const override { return sha256_; }

    std::unique_ptr<FrameSource> create_source(const StreamSpec& s,
                                               std::string& err) override {
        return make_mpp_source(s, err);
    }

    std::unique_ptr<InferenceContext> create_context(int index,
                                                     std::string& err) override {
        if (index < 0 || index >= max_contexts_) {
            err = "rknn backend: context index " + std::to_string(index) +
                  " outside 0.." + std::to_string(max_contexts_ - 1);
            return nullptr;
        }
        const uint32_t mask = core_masks_[static_cast<size_t>(index) % core_masks_.size()];
        auto hybrid = RknnHybrid::create(model_path_, mask, err);
        if (hybrid) hybrid->set_input(input_);  // §6.11 backend.input
        if (!hybrid) return nullptr;
        if (hybrid->model_width() != model_w_ || hybrid->model_height() != model_h_) {
            err = "rknn backend: context model geometry differs from the backend probe";
            return nullptr;
        }
        contexts_.push_back(std::move(hybrid));
        return std::make_unique<RknnContext>(contexts_.back().get(), decoder_.get(),
                                             model_w_, model_h_);
    }

private:
    std::string model_path_, sha256_, compatible_, rknn_sdk_;
    std::vector<uint32_t> core_masks_;
    std::unique_ptr<Decoder> decoder_;
    std::vector<std::unique_ptr<RknnHybrid>> contexts_;  // outlives the contexts
    int model_w_ = 0, model_h_ = 0, max_contexts_ = 1;
    bool ok_ = false;
};

}  // namespace

std::unique_ptr<Backend> make_rknn_backend(const std::string& backend_json,
                                           std::string& err) {
    err.clear();
    auto b = std::make_unique<RknnBackend>(backend_json, err);
    if (!err.empty() || !b->ok()) return nullptr;
    return b;
}

}  // namespace vb
