// CPU inference backend (spec BASE-1 §8 M1.9).
//
// ONNX Runtime C API (1.20.1). No accelerator: letterbox, colour conversion
// and scaling are plain C++ loops; the YOLOX head decode + NMS are the
// shared src/post implementations. One OrtSession per backend; ORT sessions
// are safe for concurrent Run() calls, so every InferenceContext shares it.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

#include <onnxruntime_c_api.h>

#include "cpu_backend.h"
#include "dev_tensor.h"
#include "sha256.h"
#include "sources/gst_source.h"
#include "sources/synthetic.h"
#include "vb/decoder.h"
#include "vb/json.h"
#include "vb/letterbox.h"
#include "vb/post.h"

namespace vb {
namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

const OrtApi* ort() {
    static const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return api;
}

// Every ORT C API call that returns OrtStatus* is ORT_MUST_USE_RESULT: a
// non-null status must be inspected and released or ORT leaks it. Centralize
// that here so no call site can silently discard an error.
bool ort_check(OrtStatus* st, const char* what, std::string& err) {
    if (st == nullptr) return true;
    const char* msg = ort()->GetErrorMessage(st);
    err = std::string("ort: ") + what + ": " + (msg ? msg : "?");
    ort()->ReleaseStatus(st);
    return false;
}

OrtEnv* ort_env(std::string& err) {
    static OrtEnv* env = nullptr;
    static std::string init_err;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string e;
        if (!ort_check(ort()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vb-cpu", &env), "CreateEnv", e)) {
            init_err = e;
            env = nullptr;
        }
    });
    if (!env && err.empty()) err = init_err.empty() ? "ort: CreateEnv failed" : init_err;
    return env;
}

// ---- Letterbox + RGB(A/BGR) -> NCHW float preprocessing (plain loops) ----
void preprocess(const FrameBuf& f, const LetterboxGeom& g, bool bgr,
                std::vector<float>& nchw) {
    const int mw = g.model_w, mh = g.model_h;
    const int iw0 = static_cast<int>(g.pad_x);            // image region origin
    const int ih0 = static_cast<int>(g.pad_y);
    nchw.assign(static_cast<size_t>(mw) * mh * 3, 0.0f);
    float* plane_r = nchw.data();
    float* plane_g = plane_r + static_cast<size_t>(mw) * mh;
    float* plane_b = plane_g + static_cast<size_t>(mw) * mh;
    const int sw = f.w, sh = f.h, stride = f.stride;
    const uint8_t* src = f.host;
    for (int y = 0; y < mh; ++y) {
        float sy = (y - ih0 + 0.5f) / g.scale - 0.5f;
        int y0 = static_cast<int>(std::floor(sy));
        float fy = sy - y0;
        if (y0 < 0) { y0 = 0; fy = 0; }
        if (y0 > sh - 1) { y0 = sh - 1; fy = 0; }
        int y1 = std::min(y0 + 1, sh - 1);
        for (int x = 0; x < mw; ++x) {
            float sx = (x - iw0 + 0.5f) / g.scale - 0.5f;
            int x0 = static_cast<int>(std::floor(sx));
            float fx = sx - x0;
            if (x0 < 0) { x0 = 0; fx = 0; }
            if (x0 > sw - 1) { x0 = sw - 1; fx = 0; }
            int x1 = std::min(x0 + 1, sw - 1);
            const uint8_t* p00 = src + static_cast<size_t>(y0) * stride + x0 * 3;
            const uint8_t* p01 = src + static_cast<size_t>(y0) * stride + x1 * 3;
            const uint8_t* p10 = src + static_cast<size_t>(y1) * stride + x0 * 3;
            const uint8_t* p11 = src + static_cast<size_t>(y1) * stride + x1 * 3;
            float ch[3];
            for (int c = 0; c < 3; ++c) {
                float v = (p00[c] * (1 - fx) + p01[c] * fx) * (1 - fy) +
                          (p10[c] * (1 - fx) + p11[c] * fx) * fy;
                ch[c] = v / 255.0f;
            }
            size_t off = static_cast<size_t>(y) * mw + x;
            plane_r[off] = ch[0];
            plane_g[off] = ch[1];
            plane_b[off] = ch[2];
        }
    }
    if (bgr) std::swap_ranges(plane_r, plane_r + static_cast<size_t>(mw) * mh, plane_b);
}

class CpuBackend;

class CpuContext : public InferenceContext, public RawTensorSource {
public:
    CpuContext(class CpuBackend* b) : b_(b) {}

    int infer(const FrameBuf* const* frames, size_t n, float score, float nms_th,
              DetectionResult* out, std::string& err) override;

    // §6.12 dev mode: raw outputs of the most recent infer(), only populated
    // when backend.decoder.type == "raw". Defined after CpuBackend.
    bool last_raw_tensors(std::vector<DevTensor>& out) override;

private:
    class CpuBackend* b_;
    std::vector<DevTensor> raw_;  // this context's copy, taken under run_mu_
};

class CpuBackend : public Backend {
public:
    explicit CpuBackend(const std::string& json, std::string& err) {
        Json j;
        try {
            j = json.empty() ? Json::object() : json_parse(json);
        } catch (const std::exception& e) {
            err = std::string("cpu backend json: ") + e.what();
            return;
        }
        if (!j.contains("model_path")) {
            err = "cpu backend requires backend.model_path";
            return;
        }
        auto dj = j.find("decoder");
        if (dj != j.end() && dj->is_object()) {
            // §6.11 M1.15: backend.decoder object selects the shared decoder.
            decoder_ = make_decoder(dj->dump(), err);
            if (!decoder_) return;
            decoder_is_raw_ = dj->value("type", std::string()) == "raw";  // §6.12
        } else {
            std::string decoder =
                (dj != j.end() && dj->is_string()) ? dj->get<std::string>() : "yolox";
            if (decoder != "yolox") {
                err = "cpu backend: unsupported decoder '" + decoder + "'";
                return;
            }
        }
        model_path_ = j.at("model_path").get<std::string>();
        int threads = j.value("intra_threads", 2);

        FILE* f = std::fopen(model_path_.c_str(), "rb");
        if (!f) {
            err = "cannot open model: " + model_path_;
            return;
        }
        std::vector<uint8_t> bytes;
        char buf[65536];
        size_t r;
        while ((r = std::fread(buf, 1, sizeof buf, f)) > 0) bytes.insert(bytes.end(), buf, buf + r);
        std::fclose(f);
        if (bytes.empty()) {
            err = "empty model file: " + model_path_;
            return;
        }
        sha256_ = sha256_hex(bytes);

        OrtSessionOptions* so = nullptr;
        if (!ort_check(ort()->CreateSessionOptions(&so), "CreateSessionOptions", err)) return;
        if (!ort_check(ort()->SetIntraOpNumThreads(so, std::max(1, threads)),
                       "SetIntraOpNumThreads", err)) {
            ort()->ReleaseSessionOptions(so);
            return;
        }
        if (!ort_check(ort()->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL),
                       "SetSessionGraphOptimizationLevel", err)) {
            ort()->ReleaseSessionOptions(so);
            return;
        }
        OrtEnv* env = ort_env(err);
        if (!env) {
            ort()->ReleaseSessionOptions(so);
            return;
        }
        bool loaded = ort_check(ort()->CreateSession(env, model_path_.c_str(), so, &session_),
                                "load model", err);
        ort()->ReleaseSessionOptions(so);
        if (!loaded) {
            session_ = nullptr;
            return;
        }

        size_t n_in = 0, n_out = 0;
        if (!ort_check(ort()->SessionGetInputCount(session_, &n_in), "SessionGetInputCount", err))
            return;
        if (!ort_check(ort()->SessionGetOutputCount(session_, &n_out), "SessionGetOutputCount", err))
            return;
        if (n_in != 1 || n_out < 1 || (!decoder_ && n_out != 1)) {
            err = "cpu backend expects a single-input model with one output "
                  "(multi-output only with backend.decoder)";
            return;
        }
        OrtAllocator* alloc = nullptr;
        if (!ort_check(ort()->GetAllocatorWithDefaultOptions(&alloc),
                       "GetAllocatorWithDefaultOptions", err))
            return;
        char* iname = nullptr;
        if (!ort_check(ort()->SessionGetInputName(session_, 0, alloc, &iname),
                       "SessionGetInputName", err))
            return;
        input_name_ = iname ? iname : "";
        if (iname && !ort_check(ort()->AllocatorFree(alloc, iname), "AllocatorFree(input name)", err))
            return;
        char* oname = nullptr;
        if (!ort_check(ort()->SessionGetOutputName(session_, 0, alloc, &oname),
                       "SessionGetOutputName", err))
            return;
        output_name_ = oname ? oname : "";
        if (oname && !ort_check(ort()->AllocatorFree(alloc, oname), "AllocatorFree(output name)", err))
            return;
        if (decoder_) {  // all output names, §6.11 multi-output decoders
            for (size_t oi = 0; oi < n_out; ++oi) {
                char* nm = nullptr;
                if (!ort_check(ort()->SessionGetOutputName(session_, oi, alloc, &nm),
                               "SessionGetOutputName", err))
                    return;
                out_names_.push_back(nm ? nm : "");
                if (nm &&
                    !ort_check(ort()->AllocatorFree(alloc, nm), "AllocatorFree", err))
                    return;
            }
        }
        // Input shape [1, 3, H, W].
        OrtTypeInfo* ti = nullptr;
        if (!ort_check(ort()->SessionGetInputTypeInfo(session_, 0, &ti),
                       "SessionGetInputTypeInfo", err))
            return;
        const OrtTensorTypeAndShapeInfo* shape = nullptr;
        if (!ort_check(ort()->CastTypeInfoToTensorInfo(ti, &shape), "CastTypeInfoToTensorInfo", err)) {
            ort()->ReleaseTypeInfo(ti);
            return;
        }
        size_t ndim = 0;
        if (!ort_check(ort()->GetDimensionsCount(shape, &ndim), "GetDimensionsCount", err)) {
            ort()->ReleaseTypeInfo(ti);
            return;
        }
        std::vector<int64_t> dims(ndim);
        bool dims_ok = ort_check(ort()->GetDimensions(shape, dims.data(), ndim), "GetDimensions", err);
        ort()->ReleaseTypeInfo(ti);
        if (!dims_ok) return;
        if (ndim != 4 || dims[1] != 3) {
            err = "cpu backend expects an NCHW RGB input";
            return;
        }
        model_h_ = static_cast<int>(dims[2]);
        model_w_ = static_cast<int>(dims[3]);
        if (model_h_ <= 0 || model_w_ <= 0) {
            err = "model input must have static H/W";
            return;
        }
        ok_ = true;
    }

    ~CpuBackend() override {
        if (session_) ort()->ReleaseSession(session_);
    }

    bool ok() const { return ok_; }

    // Runs one frame through the session; returns false with err on failure.
    bool run(const FrameBuf& f, float score, float nms_th, DetectionResult& out,
             std::string& err) {
        double t0 = now_ms();
        out.geom = LetterboxGeom::fit(f.w, f.h, model_w_, model_h_, Align::Center);
        if (f.mem != Mem::Host || !f.host ||
            (f.fmt != PixFmt::RGB888 && f.fmt != PixFmt::BGR888)) {
            err = "cpu backend requires host RGB888/BGR888 frames";
            return false;
        }
        std::vector<float> nchw;
        preprocess(f, out.geom, f.fmt == PixFmt::BGR888, nchw);
        double t1 = now_ms();

        const int64_t in_shape[4] = {1, 3, model_h_, model_w_};
        OrtMemoryInfo* meminfo = nullptr;
        if (!ort_check(ort()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &meminfo),
                       "CreateCpuMemoryInfo", err))
            return false;
        OrtValue* in = nullptr;
        bool in_ok = ort_check(
            ort()->CreateTensorWithDataAsOrtValue(meminfo, nchw.data(), nchw.size() * sizeof(float),
                                                  in_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                  &in),
            "CreateTensorWithDataAsOrtValue", err);
        ort()->ReleaseMemoryInfo(meminfo);
        if (!in_ok || !in) {
            if (in_ok) err = "ort: input tensor creation returned null";
            return false;
        }
        const char* in_names[1] = {input_name_.c_str()};
        std::vector<const char*> onames{output_name_.c_str()};
        if (decoder_)
            for (const auto& nm : out_names_) onames.push_back(nm.c_str());
        std::vector<OrtValue*> outs(onames.size(), nullptr);
        bool run_ok = ort_check(
            ort()->Run(session_, nullptr, in_names, &in, 1, onames.data(), onames.size(),
                       outs.data()),
            "Run", err);
        ort()->ReleaseValue(in);
        if (!run_ok) return false;
        double t2 = now_ms();

        auto tensor_shape = [&](OrtValue* v, std::vector<int64_t>& dims) -> bool {
            OrtTypeInfo* ti = nullptr;
            if (!ort_check(ort()->GetTypeInfo(v, &ti), "GetTypeInfo", err)) return false;
            const OrtTensorTypeAndShapeInfo* shape = nullptr;
            if (!ort_check(ort()->CastTypeInfoToTensorInfo(ti, &shape),
                           "CastTypeInfoToTensorInfo", err)) {
                ort()->ReleaseTypeInfo(ti);
                return false;
            }
            size_t ndim = 0;
            if (!ort_check(ort()->GetDimensionsCount(shape, &ndim), "GetDimensionsCount", err)) {
                ort()->ReleaseTypeInfo(ti);
                return false;
            }
            dims.resize(ndim);
            bool ok = ort_check(ort()->GetDimensions(shape, dims.data(), ndim), "GetDimensions", err);
            ort()->ReleaseTypeInfo(ti);
            return ok;
        };

        if (decoder_) {
            // §6.11 M1.15: shared decoder over all outputs (batch assumed 1).
            std::vector<TensorView> views(outs.size());
            for (size_t oi = 0; oi < outs.size(); ++oi) {
                std::vector<int64_t> dims;
                if (!tensor_shape(outs[oi], dims)) {
                    for (OrtValue* v : outs) ort()->ReleaseValue(v);
                    return false;
                }
                float* data = nullptr;
                if (!ort_check(
                        ort()->GetTensorMutableData(outs[oi], reinterpret_cast<void**>(&data)),
                        "GetTensorMutableData", err)) {
                    for (OrtValue* v : outs) ort()->ReleaseValue(v);
                    return false;
                }
                size_t elem = 1;
                for (int64_t d : dims) elem *= static_cast<size_t>(d);
                views[oi].data = data;
                views[oi].count = elem;
                if (!dims.empty() && dims[0] == 1) dims.erase(dims.begin());  // drop batch
                views[oi].dims = std::move(dims);
                views[oi].name = onames[oi];
            }
            if (decoder_is_raw_) {
                // §6.12 dev mode: copy the raw outputs for VBT1 passthrough.
                std::vector<DevTensor> cap;
                for (const auto& v : views) {
                    if (v.dims.size() > 4) continue;  // VBT1 carries up to 4 dims
                    DevTensor t;
                    t.name = v.name;
                    for (int64_t d : v.dims) t.dims.push_back(static_cast<int32_t>(d));
                    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(v.data);
                    t.data.assign(bytes, bytes + v.count * sizeof(float));
                    cap.push_back(std::move(t));
                }
                std::lock_guard<std::mutex> lk(raw_mu_);
                last_raw_ = std::move(cap);
            }
            out.dets.clear();
            out.kpts.clear();
            bool ok = decoder_->decode(views.data(), views.size(), model_w_, model_h_,
                                       score, nms_th, out, err);
            for (OrtValue* v : outs) ort()->ReleaseValue(v);
            if (!ok) return false;
            double t3 = now_ms();
            out.preprocess_ms = static_cast<float>(t1 - t0);
            out.inference_ms = static_cast<float>(t2 - t1);
            out.postprocess_ms = static_cast<float>(t3 - t2);
            return true;
        }

        // Output assumed [1, A, 5 + n_cls] (batch 1).
        OrtTypeInfo* ti = nullptr;
        if (!ort_check(ort()->GetTypeInfo(outs[0], &ti), "GetTypeInfo", err)) {
            ort()->ReleaseValue(outs[0]);
            return false;
        }
        const OrtTensorTypeAndShapeInfo* shape = nullptr;
        if (!ort_check(ort()->CastTypeInfoToTensorInfo(ti, &shape), "CastTypeInfoToTensorInfo", err)) {
            ort()->ReleaseTypeInfo(ti);
            ort()->ReleaseValue(outs[0]);
            return false;
        }
        size_t ndim = 0;
        if (!ort_check(ort()->GetDimensionsCount(shape, &ndim), "GetDimensionsCount", err)) {
            ort()->ReleaseTypeInfo(ti);
            ort()->ReleaseValue(outs[0]);
            return false;
        }
        std::vector<int64_t> dims(ndim);
        bool dims_ok = ort_check(ort()->GetDimensions(shape, dims.data(), ndim), "GetDimensions", err);
        ort()->ReleaseTypeInfo(ti);
        if (!dims_ok) {
            ort()->ReleaseValue(outs[0]);
            return false;
        }
        size_t elem = 1;
        for (int64_t d : dims) elem *= static_cast<size_t>(d);
        float* data = nullptr;
        if (!ort_check(ort()->GetTensorMutableData(outs[0], reinterpret_cast<void**>(&data)),
                       "GetTensorMutableData", err)) {
            ort()->ReleaseValue(outs[0]);
            return false;
        }
        int anchors = 0, n_cls = 0;
        if (ndim == 3 && dims[0] == 1) {
            // YOLOX head: [1, A, 5+C] (standard export) or [1, 5+C, A]
            // (transposed). Channels dim is the one <= 200 (COCO-80+5 fits;
            // anchor counts are in the thousands).
            if (dims[2] >= 6 && dims[2] <= 200) {  // [1, A, 5+C]
                anchors = static_cast<int>(dims[1]);
                n_cls = static_cast<int>(dims[2] - 5);
            } else if (dims[1] >= 6 && dims[1] <= 200) {  // [1, 5+C, A]
                anchors = static_cast<int>(dims[2]);
                n_cls = static_cast<int>(dims[1] - 5);
                transposed_ = true;
            }
        }
        if (!data || anchors <= 0 || n_cls <= 0 ||
            elem < static_cast<size_t>(anchors) * (n_cls + 5)) {
            ort()->ReleaseValue(outs[0]);
            err = "cpu backend: unexpected output shape";
            return false;
        }
        out.dets.clear();
        out.kpts.clear();
        if (transposed_) {
            // Materialize [A, 5+C] rows from [5+C, A].
            tmp_.resize(static_cast<size_t>(anchors) * (n_cls + 5));
            const size_t row = static_cast<size_t>(n_cls) + 5;
            for (int i = 0; i < anchors; ++i)
                for (size_t c = 0; c < row; ++c)
                    tmp_[i * row + c] = data[c * anchors + i];
            yolox_decode(tmp_.data(), anchors, n_cls, model_w_, model_h_, score, out.dets);
        } else {
            yolox_decode(data, anchors, n_cls, model_w_, model_h_, score, out.dets);
        }
        ort()->ReleaseValue(outs[0]);
        nms(out.dets, out.kpts, nms_th, true);
        double t3 = now_ms();
        out.preprocess_ms = static_cast<float>(t1 - t0);
        out.inference_ms = static_cast<float>(t2 - t1);
        out.postprocess_ms = static_cast<float>(t3 - t2);
        return true;
    }

    const char* name() const override { return "cpu"; }
    Caps caps() const override {
        Caps c;
        unsigned n = std::thread::hardware_concurrency();
        c.max_contexts = static_cast<int>(std::min<unsigned>(n ? n : 4, 8));
        c.max_batch = 1;
        c.keypoints = decoder_ ? decoder_->keypoints() : 0;
        c.exclusive_device = false;
        return c;
    }
    std::pair<int, int> model_hw() const override { return {model_w_, model_h_}; }
    std::string model_sha256() const override { return sha256_; }

    std::unique_ptr<FrameSource> create_source(const StreamSpec& s,
                                               std::string& err) override {
        if (s.url.rfind("synthetic://", 0) == 0) {
            // Deterministic dev/test frames through the synthetic adapter.
            std::string e2;
            auto syn = make_synthetic_backend("{}", e2);
            return syn->create_source(s, err);
        }
#if defined(VB_WITH_GST)
        return make_gst_source(s, err);
#else
        (void)s;
        err = "cpu backend: no frame source for url (built without VB_WITH_GST)";
        return nullptr;
#endif
    }

    std::unique_ptr<InferenceContext> create_context(int index,
                                                     std::string& err) override {
        (void)index;
        (void)err;
        return std::make_unique<CpuContext>(this);
    }

private:
    friend class CpuContext;

    std::string model_path_, sha256_, input_name_, output_name_;
    std::unique_ptr<Decoder> decoder_;  // §6.11 backend.decoder (may be null)
    bool decoder_is_raw_ = false;       // §6.12 dev-mode raw passthrough
    std::mutex raw_mu_;                 // guards last_raw_
    std::vector<DevTensor> last_raw_;   // dev-mode raw outputs (§6.12)
    std::vector<std::string> out_names_;  // all outputs when decoder_ is set
    OrtSession* session_ = nullptr;
    int model_w_ = 0, model_h_ = 0;
    bool transposed_ = false;
    std::vector<float> tmp_;  // transposed-decode scratch (contexts share the backend)
    std::mutex run_mu_;       // guards tmp_
    bool ok_ = false;
};

bool CpuContext::last_raw_tensors(std::vector<DevTensor>& out) {
    out = raw_;
    return !out.empty();
}

int CpuContext::infer(const FrameBuf* const* frames, size_t n, float score, float nms_th,
                      DetectionResult* out, std::string& err) {
    if (n != 1) {
        err = "cpu backend max_batch is 1";
        return -1;
    }
    std::lock_guard<std::mutex> lk(b_->run_mu_);
    if (!b_->run(*frames[0], score, nms_th, out[0], err)) return -1;
    // The backend-wide buffer is overwritten by the next context's run(), so
    // copy it while run_mu_ still pins it to this frame (§6.12).
    if (b_->decoder_is_raw_) {
        std::lock_guard<std::mutex> rk(b_->raw_mu_);
        raw_ = b_->last_raw_;
    }
    return 0;
}

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const std::string& backend_json,
                                          std::string& err) {
    err.clear();
    auto b = std::make_unique<CpuBackend>(backend_json, err);
    if (!err.empty() || !b->ok()) return nullptr;
    return b;
}

}  // namespace vb
