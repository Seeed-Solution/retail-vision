#include "rknn_stage2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/dma-heap.h>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "rga/im2d.h"
#include "rga/rga.h"
#include "hybrid_rga_rknn.h"
#include "rknn_api.h"
#include "model_sha256.h"
#include "vb/decoder.h"
#include "vb/rate_crop.h"

namespace vb {
namespace {

std::mutex g_stage2_cache_mutex;

bool finite_positive(float x) { return std::isfinite(x) && x > 0.0f; }

bool read_model(const std::string& path, std::vector<uint8_t>& bytes,
                std::string& err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { err = "rknn stage2: cannot open model: " + path; return false; }
    const std::streamoff n = f.tellg();
    if (n <= 0 || static_cast<uint64_t>(n) > (512u << 20)) {
        err = "rknn stage2: model is empty or too large";
        return false;
    }
    bytes.resize(static_cast<size_t>(n));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(bytes.data()), n)) {
        err = "rknn stage2: model read failed";
        return false;
    }
    return true;
}

bool product(const std::vector<uint32_t>& dims, size_t& out) {
    out = 1;
    for (uint32_t d : dims) {
        if (!d || out > std::numeric_limits<size_t>::max() / d) return false;
        out *= d;
    }
    return true;
}

bool valid_crop(const CropReq& c) {
    return std::isfinite(c.x0) && std::isfinite(c.y0) &&
           std::isfinite(c.x1) && std::isfinite(c.y1) &&
           c.x0 >= 0.0f && c.y0 >= 0.0f && c.x1 <= 1.0f && c.y1 <= 1.0f &&
           c.x0 < c.x1 && c.y0 < c.y1;
}

int even_floor(int x) { return x & ~1; }
int even_ceil(int x) { return (x + 1) & ~1; }

}  // namespace

class RknnStage2Model {
public:
    ~RknnStage2Model() { if (base_) rknn_destroy(base_); }
    RknnStage2Model(const RknnStage2Model&) = delete;
    RknnStage2Model& operator=(const RknnStage2Model&) = delete;

    rknn_context duplicate(std::string& err) const {
        std::lock_guard<std::mutex> lock(dup_mutex_);
        rknn_context out = 0;
        const int rc = rknn_dup_context(const_cast<rknn_context*>(&base_), &out);
        if (rc != RKNN_SUCC || !out) {
            err = "rknn_dup_context=" + std::to_string(rc);
            return 0;
        }
        const int core_rc = rknn_set_core_mask(out, static_cast<rknn_core_mask>(core_mask_));
        if (core_rc != RKNN_SUCC) {
            rknn_destroy(out);
            err = "rknn_set_core_mask(stage2)=" + std::to_string(core_rc);
            return 0;
        }
        return out;
    }

    static std::shared_ptr<RknnStage2Model> load(const Stage2Spec& spec,
                                                 uint32_t core_mask,
                                                 std::string& err) {
        std::vector<uint8_t> bytes;
        if (!read_model(spec.model_path, bytes, err)) return {};
        // The model path is already guarded by the backend's model checksum
        // contract. Include file size and the requested input contract in the
        // cache key to avoid accidental cross-model aliasing.
        const std::string key = rknn_detail::sha256_hex(bytes) +
            ":" + std::to_string(spec.in_w) + "x" + std::to_string(spec.in_h) +
            ":" + (spec.bgr ? "bgr" : "rgb") + ":" + std::to_string(spec.scale) +
            ":" + std::to_string(spec.mean[0]) + ":" + std::to_string(spec.mean[1]) +
            ":" + std::to_string(spec.mean[2]) + ":core=" + std::to_string(core_mask);
        std::lock_guard<std::mutex> lock(g_stage2_cache_mutex);
        auto it = cache().find(key);
        if (it != cache().end()) if (auto p = it->second.lock()) return p;

        auto p = std::shared_ptr<RknnStage2Model>(new RknnStage2Model());
        p->core_mask_ = core_mask;
        // rknn_dup_context shares the loaded network from this base context.
        // RKNN 2.3.2 rejects RKNN_FLAG_SHARE_WEIGHT_MEM without an init
        // extension on the measured <rk3588-board> runtime, so leave init flags zero
        // and let the documented duplicate API perform the sharing.
        const int rc = rknn_init(&p->base_, bytes.data(), static_cast<uint32_t>(bytes.size()),
                                 0, nullptr);
        if (rc != RKNN_SUCC) { err = "rknn_init(stage2)=" + std::to_string(rc); return {}; }
        const int core_rc = rknn_set_core_mask(p->base_, static_cast<rknn_core_mask>(core_mask));
        if (core_rc != RKNN_SUCC) {
            err = "rknn_set_core_mask(stage2 base)=" + std::to_string(core_rc);
            return {};
        }
        rknn_sdk_version sdk{};
        std::string sdk_name = "?";
        if (rknn_query(p->base_, RKNN_QUERY_SDK_VERSION, &sdk, sizeof(sdk)) == RKNN_SUCC)
            sdk_name = std::string(sdk.api_version) + "/" + std::string(sdk.drv_version);
        std::fprintf(stderr, "rknn stage2: librknnrt %s model_sha256=%s input=%dx%d color=%s core=%u\n",
                     sdk_name.c_str(), rknn_detail::sha256_hex(bytes).c_str(),
                     spec.in_w, spec.in_h, spec.bgr ? "bgr" : "rgb", core_mask);
        cache()[key] = p;
        return p;
    }

private:
    RknnStage2Model() = default;
    static std::unordered_map<std::string, std::weak_ptr<RknnStage2Model>>& cache() {
        static std::unordered_map<std::string, std::weak_ptr<RknnStage2Model>> c;
        return c;
    }
    rknn_context base_ = 0;
    uint32_t core_mask_ = 1;
    mutable std::mutex dup_mutex_;
};

namespace {

class RknnStage2 final : public Stage2Context {
public:
    RknnStage2(std::shared_ptr<RknnStage2Model> model, const Stage2Spec& spec,
               std::string& err)
        : model_(std::move(model)), spec_(spec) {
        ctx_ = model_->duplicate(err);
        if (!ctx_) return;
        if (!setup(err)) {
            if (input_mem_) { rknn_destroy_mem(ctx_, input_mem_); input_mem_ = nullptr; }
            rknn_destroy(ctx_);
            ctx_ = 0;
        }
    }
    ~RknnStage2() override {
        if (input_mem_) rknn_destroy_mem(ctx_, input_mem_);
        if (ctx_) rknn_destroy(ctx_);
    }

    int infer_crops(const FrameBuf& frame, const CropReq* crops, size_t n,
                    TensorView* outs, std::string& err) override {
        if (outs && n <= 16) clear_outputs(outs, n);
        if (!ready_) { err = "rknn stage2 context is not ready"; return -1; }
        if (!crops || !outs || !n || n > 16) { err = "rknn stage2 invalid crop list"; return -1; }
        size_t source_bytes = 0;
        if (frame.mem != Mem::DmaBuf || frame.fmt != PixFmt::NV12 ||
            frame.dmabuf_fd < 0 || !nv12_storage_bytes(frame.w, frame.h, frame.stride,
                                                       frame.hstride, source_bytes)) {
            err = "rknn stage2 requires NV12 DMA-BUF frame";
            return -1;
        }
        std::lock_guard<std::mutex> lock(run_mutex_);
        output_stores_.clear();
        output_stores_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (!valid_crop(crops[i])) { err = "rknn stage2 crop bbox is invalid"; clear_outputs(outs, n); return -1; }
            int x0, y0, x1, y1;
            if (!crop_rect(frame, crops[i], x0, y0, x1, y1, err)) { clear_outputs(outs, n); return -1; }
            const int rc = rga_crop(frame, x0, y0, x1 - x0, y1 - y0, err);
            output_stores_.emplace_back(output_count_, 0.0f);
            if (rc != 0) {
                clear_outputs(outs, n); return -1;
            }
            // RGA writes the RKNN native input through its device mapping. Make
            // that write visible to the RKNN CPU view before rknn_run; the
            // default input cache handling must not obscure the device result.
            if (rknn_mem_sync(ctx_, input_mem_, RKNN_MEMORY_SYNC_FROM_DEVICE) != RKNN_SUCC) {
                err = "rknn_mem_sync(stage2 RGA input from device) failed";
                clear_outputs(outs, n); return -1;
            }
            if (run_output(outs[i], output_stores_.back(), err) != 0) {
                clear_outputs(outs, n); return -1;
            }
        }
        return 0;
    }

    int infer_rgb(const uint8_t* rgb, int w, int h, int stride,
                  TensorView* out, std::string& err) override {
        if (out) clear_outputs(out, 1);
        if (!ready_) { err = "rknn stage2 context is not ready"; return -1; }
        const size_t max_pixels = 16777216u;
        if (!rgb || !out || w <= 0 || h <= 0 ||
            static_cast<uint64_t>(w) * static_cast<uint64_t>(h) > max_pixels ||
            static_cast<uint64_t>(w) > (std::numeric_limits<size_t>::max() / 3u) ||
            static_cast<size_t>(stride < 0 ? 0 : stride) < static_cast<size_t>(w) * 3u) {
            err = "rknn stage2 infer_rgb frame invalid"; return -1;
        }
        std::lock_guard<std::mutex> lock(run_mutex_);
        output_stores_.assign(1, std::vector<float>(output_count_, 0.0f));
        // JPEG/image probes arrive as host RGB. The resize/format conversion is
        // deliberately CPU-side; the model run remains exclusively RKNN/NPU.
        uint8_t* dst = static_cast<uint8_t*>(input_mem_->virt_addr);
        const int ds = input_stride_ * 3;
        // Match the public host-image contract: half-pixel bilinear stretch,
        // followed by RGB/BGR packing.  Nearest-neighbour changes the pixels
        // for every non-native input size and therefore changes recognition.
        for (int y = 0; y < spec_.in_h; ++y) {
            const float fy = (static_cast<float>(y) + 0.5f) * static_cast<float>(h) /
                             static_cast<float>(spec_.in_h) - 0.5f;
            const int sy0 = std::clamp(static_cast<int>(std::floor(fy)), 0, h - 1);
            const int sy1 = std::min(sy0 + 1, h - 1);
            const float ay = std::clamp(fy - static_cast<float>(sy0), 0.0f, 1.0f);
            for (int x = 0; x < spec_.in_w; ++x) {
                const float fx = (static_cast<float>(x) + 0.5f) * static_cast<float>(w) /
                                 static_cast<float>(spec_.in_w) - 0.5f;
                const int sx0 = std::clamp(static_cast<int>(std::floor(fx)), 0, w - 1);
                const int sx1 = std::min(sx0 + 1, w - 1);
                const float ax = std::clamp(fx - static_cast<float>(sx0), 0.0f, 1.0f);
                const uint8_t* p00 = rgb + static_cast<size_t>(sy0) * stride + static_cast<size_t>(sx0) * 3;
                const uint8_t* p01 = rgb + static_cast<size_t>(sy0) * stride + static_cast<size_t>(sx1) * 3;
                const uint8_t* p10 = rgb + static_cast<size_t>(sy1) * stride + static_cast<size_t>(sx0) * 3;
                const uint8_t* p11 = rgb + static_cast<size_t>(sy1) * stride + static_cast<size_t>(sx1) * 3;
                uint8_t* d = dst + static_cast<size_t>(y) * ds + static_cast<size_t>(x) * 3;
                for (int c = 0; c < 3; ++c) {
                    const float top = static_cast<float>(p00[c]) +
                        (static_cast<float>(p01[c]) - static_cast<float>(p00[c])) * ax;
                    const float bot = static_cast<float>(p10[c]) +
                        (static_cast<float>(p11[c]) - static_cast<float>(p10[c])) * ax;
                    const uint8_t value = static_cast<uint8_t>(std::clamp(
                        static_cast<int>(std::lround(top + (bot - top) * ay)), 0, 255));
                    d[spec_.bgr ? (2 - c) : c] = value;
                }
            }
        }
        if (rknn_mem_sync(ctx_, input_mem_, RKNN_MEMORY_SYNC_TO_DEVICE) != RKNN_SUCC) {
            err = "rknn_mem_sync(stage2 host input) failed"; return -1;
        }
        return run_output(*out, output_stores_.back(), err);
    }

private:
    bool setup(std::string& err) {
        rknn_input_output_num io{};
        if (rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC ||
            io.n_input != 1 || io.n_output != 1) {
            err = "rknn stage2 expects one input and one output"; return false;
        }
        rknn_tensor_attr logical{};
        logical.index = 0;
        const int input_rc = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &logical, sizeof(logical));
        const bool nchw = logical.n_dims == 4 && logical.dims[0] == 1 &&
            logical.dims[1] == 3 && logical.dims[2] == static_cast<uint32_t>(spec_.in_h) &&
            logical.dims[3] == static_cast<uint32_t>(spec_.in_w);
        const bool nhwc = logical.n_dims == 4 && logical.dims[0] == 1 &&
            logical.dims[1] == static_cast<uint32_t>(spec_.in_h) &&
            logical.dims[2] == static_cast<uint32_t>(spec_.in_w) && logical.dims[3] == 3;
        if (input_rc != RKNN_SUCC || (!nchw && !nhwc)) {
            err = "rknn stage2 input shape unsupported dims=" +
                std::to_string(logical.dims[0]) + "," + std::to_string(logical.dims[1]) + "," +
                std::to_string(logical.dims[2]) + "," + std::to_string(logical.dims[3]) +
                " fmt=" + std::to_string(static_cast<int>(logical.fmt)); return false;
        }
        rknn_tensor_attr native{};
        native.index = 0;
        if (rknn_query(ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &native, sizeof(native)) != RKNN_SUCC) {
            err = "RKNN_QUERY_NATIVE_INPUT_ATTR(stage2) failed"; return false;
        }
        native.type = RKNN_TENSOR_UINT8;
        input_stride_ = native.w_stride ? static_cast<int>(native.w_stride) : spec_.in_w;
        input_mem_ = rknn_create_mem(ctx_, native.size_with_stride);
        if (!input_mem_ || !input_mem_->virt_addr || input_mem_->size <
            static_cast<uint32_t>(input_stride_ * spec_.in_h * 3)) {
            err = "rknn stage2 input memory unavailable"; return false;
        }
        std::memset(input_mem_->virt_addr, 0, input_mem_->size);
        if (rknn_mem_sync(ctx_, input_mem_, RKNN_MEMORY_SYNC_TO_DEVICE) != RKNN_SUCC ||
            rknn_set_io_mem(ctx_, input_mem_, &native) != RKNN_SUCC) {
            err = "rknn stage2 set input memory failed"; return false;
        }
        rknn_tensor_attr oa{};
        oa.index = 0;
        if (rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &oa, sizeof(oa)) != RKNN_SUCC ||
            oa.type != RKNN_TENSOR_FLOAT16 && oa.type != RKNN_TENSOR_FLOAT32) {
            err = "rknn stage2 output must be floating point"; return false;
        }
        output_dims_.clear();
        for (uint32_t i = 0; i < oa.n_dims; ++i)
            output_dims_.push_back(static_cast<int64_t>(oa.dims[i]));
        if (output_dims_.size() == 3 && output_dims_[0] == 1) {
            output_dims_.erase(output_dims_.begin());
        } else if (output_dims_.size() == 4 && output_dims_[0] == 1 &&
                   output_dims_[2] == 1) {
            output_dims_.erase(output_dims_.begin());
            output_dims_.erase(output_dims_.begin() + 1);
        }
        output_count_ = 1;
        for (int64_t d : output_dims_) {
            if (d <= 0 || output_count_ > std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(d)) { output_count_ = 0; break; }
            output_count_ *= static_cast<size_t>(d);
        }
        if (output_dims_.size() != 2 || !output_count_) {
            err = "rknn stage2 output shape unsupported dims=";
            for (size_t i = 0; i < output_dims_.size(); ++i)
                err += (i ? "," : "") + std::to_string(output_dims_[i]);
            return false;
        }
        ready_ = true;
        return true;
    }

    bool crop_rect(const FrameBuf& f, const CropReq& c, int& x0, int& y0,
                   int& x1, int& y1, std::string& err) const {
        const int fw = f.full_w > 0 ? f.full_w : f.w;
        const int fh = f.full_h > 0 ? f.full_h : f.h;
        const int ox = f.full_w > 0 ? f.crop_x0 : 0;
        const int oy = f.full_h > 0 ? f.crop_y0 : 0;
        if (fw <= 0 || fh <= 0 || fw > 8192 || fh > 8192 ||
            ox < 0 || oy < 0 || ox > fw || oy > fh) {
            err = "rknn stage2 source geometry is invalid"; return false;
        }
        const double dx0 = std::floor(static_cast<double>(c.x0) * fw) - ox;
        const double dy0 = std::floor(static_cast<double>(c.y0) * fh) - oy;
        const double dx1 = std::ceil(static_cast<double>(c.x1) * fw) - ox;
        const double dy1 = std::ceil(static_cast<double>(c.y1) * fh) - oy;
        if (!std::isfinite(dx0) || !std::isfinite(dy0) || !std::isfinite(dx1) ||
            !std::isfinite(dy1)) { err = "rknn stage2 crop geometry is non-finite"; return false; }
        x0 = static_cast<int>(std::clamp(dx0, 0.0, static_cast<double>(f.w)));
        y0 = static_cast<int>(std::clamp(dy0, 0.0, static_cast<double>(f.h)));
        x1 = static_cast<int>(std::clamp(dx1, 0.0, static_cast<double>(f.w)));
        y1 = static_cast<int>(std::clamp(dy1, 0.0, static_cast<double>(f.h)));
        x0 = even_floor(x0); y0 = even_floor(y0);
        x1 = even_ceil(x1); y1 = even_ceil(y1);
        x1 = std::min(x1, even_floor(f.w)); y1 = std::min(y1, even_floor(f.h));
        if (x1 <= x0 || y1 <= y0) { err = "rknn stage2 crop is empty after NV12 alignment"; return false; }
        return true;
    }

    int rga_crop(const FrameBuf& f, int x, int y, int w, int h, std::string& err) {
        rga_buffer_t src = wrapbuffer_fd(f.dmabuf_fd, f.w, f.h, RK_FORMAT_YCbCr_420_SP,
                                         f.stride, f.hstride == 0 ? f.h : f.hstride);
        const auto fmt = spec_.bgr ? RK_FORMAT_BGR_888 : RK_FORMAT_RGB_888;
        // RGA rejects a single resize when either axis shrinks by more than 16x.
        // Keep the source crop on hardware by inserting one DMA intermediate and
        // split the resize into two bounded RGA passes.
        const int pass_w = std::max(spec_.in_w, (w + 15) / 16);
        const int pass_h = std::max(spec_.in_h, (h + 15) / 16);
        int intermediate_fd = -1;
        int pass_pitch = pass_w;
        if (pass_w != spec_.in_w || pass_h != spec_.in_h) {
            pass_pitch = (pass_w + 15) & ~15;
            if (pass_pitch < pass_w || static_cast<size_t>(pass_pitch) >
                                           kMaxHostRgbBytes / 3u ||
                static_cast<size_t>(pass_h) >
                    kMaxHostRgbBytes / (static_cast<size_t>(pass_pitch) * 3u)) {
                err = "rknn stage2 intermediate RGB layout exceeds 64 MiB";
                return -1;
            }
            const size_t image_bytes = static_cast<size_t>(pass_pitch) *
                                        static_cast<size_t>(pass_h) * 3u;
            if (image_bytes > std::numeric_limits<size_t>::max() - 4095u) {
                err = "rknn stage2 intermediate RGB size overflow";
                return -1;
            }
            const size_t bytes = (image_bytes + 4095u) & ~size_t(4095u);
            const int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
            if (heap_fd < 0) { err = "rknn stage2 dma heap unavailable"; return -1; }
            dma_heap_allocation_data alloc{};
            alloc.len = bytes;
            alloc.fd_flags = O_RDWR | O_CLOEXEC;
            if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) != 0) {
                close(heap_fd); err = "rknn stage2 intermediate DMA allocation failed"; return -1;
            }
            close(heap_fd);
            intermediate_fd = static_cast<int>(alloc.fd);
        }
        auto run_rga = [&](const rga_buffer_t& from, int sx, int sy, int sw, int sh,
                           const rga_buffer_t& to, int dw, int dh) -> IM_STATUS {
            im_rect srect{sx, sy, sw, sh};
            im_rect drect{0, 0, dw, dh};
            im_rect prect{0, 0, 0, 0};
            std::lock_guard<std::mutex> lock(rknn_rga_mutex());
            return improcess(from, to, {}, srect, drect, prect, -1, nullptr, nullptr, IM_SYNC);
        };
        IM_STATUS st = IM_STATUS_SUCCESS;
        if (intermediate_fd >= 0) {
            rga_buffer_t mid = wrapbuffer_fd(intermediate_fd, pass_w, pass_h, fmt, pass_pitch, pass_h);
            st = run_rga(src, x, y, w, h, mid, pass_w, pass_h);
            if (st == IM_STATUS_SUCCESS) {
                rga_buffer_t dst = wrapbuffer_fd(input_mem_->fd, spec_.in_w, spec_.in_h, fmt,
                                                 input_stride_, spec_.in_h);
                    st = run_rga(mid, 0, 0, pass_w, pass_h, dst, spec_.in_w, spec_.in_h);
            }
            close(intermediate_fd);
        } else {
            rga_buffer_t dst = wrapbuffer_fd(input_mem_->fd, spec_.in_w, spec_.in_h, fmt,
                                             input_stride_, spec_.in_h);
            st = run_rga(src, x, y, w, h, dst, spec_.in_w, spec_.in_h);
        }
        if (st != IM_STATUS_SUCCESS) { err = "rknn stage2 RGA crop=" + std::to_string(st) + " " + imStrError(st); return -1; }
        return 0;
    }

    int run_output(TensorView& out, std::vector<float>& storage, std::string& err) {
        if (rknn_run(ctx_, nullptr) != RKNN_SUCC) { err = "rknn stage2 run failed"; return -1; }
        rknn_output ro{}; ro.index = 0; ro.want_float = 1; ro.is_prealloc = 1;
        ro.buf = storage.data(); ro.size = storage.size() * sizeof(float);
        if (rknn_outputs_get(ctx_, 1, &ro, nullptr) != RKNN_SUCC) { err = "rknn stage2 outputs_get failed"; return -1; }
        const int rc = rknn_outputs_release(ctx_, 1, &ro);
        if (rc != RKNN_SUCC) { err = "rknn stage2 outputs_release failed"; return -1; }
        for (float v : storage) {
            if (!std::isfinite(v)) { out = TensorView{}; err = "rknn stage2 output contains NaN or Inf"; return -1; }
        }
        out.data = storage.data(); out.count = storage.size(); out.dims = output_dims_; out.dtype = 0;
        out.raw_data = nullptr;
        return 0;
    }

    static void clear_outputs(TensorView* out, size_t n) {
        for (size_t i = 0; i < n; ++i) { out[i] = TensorView{}; }
    }
    std::shared_ptr<RknnStage2Model> model_;
    Stage2Spec spec_;
    rknn_context ctx_ = 0;
    rknn_tensor_mem* input_mem_ = nullptr;
    int input_stride_ = 0;
    std::vector<int64_t> output_dims_;
    size_t output_count_ = 0;
    std::vector<std::vector<float>> output_stores_;
    std::mutex run_mutex_;
    bool ready_ = false;
};

}  // namespace

std::shared_ptr<RknnStage2Model> load_rknn_stage2_model(
    const Stage2Spec& spec, uint32_t core_mask, std::string& err) {
    if (spec.in_h <= 0 || spec.in_w <= 0 || spec.in_h > 4096 || spec.in_w > 4096 ||
        !finite_positive(spec.scale) ||
        std::fabs(spec.scale - (1.0f / 255.0f)) > 1e-7f ||
        !std::isfinite(spec.mean[0]) || !std::isfinite(spec.mean[1]) ||
        !std::isfinite(spec.mean[2]) || spec.mean[0] != 0.0f ||
        spec.mean[1] != 0.0f || spec.mean[2] != 0.0f) {
        err = "rknn stage2 requires folded uint8 normalization: scale=1/255 and mean=[0,0,0]";
        return {};
    }
    return RknnStage2Model::load(spec, core_mask, err);
}

std::unique_ptr<Stage2Context> make_rknn_stage2_context(
    const std::shared_ptr<RknnStage2Model>& model, const Stage2Spec& spec,
    std::string& err) {
    if (!model) { err = "rknn stage2 model is unavailable"; return nullptr; }
    auto p = std::make_unique<RknnStage2>(model, spec, err);
    return err.empty() ? std::move(p) : nullptr;
}

}  // namespace vb
