// Source: fall-detection platforms/rknn/native/src/hybrid_rga_rknn.cpp
//   upstream commit  5af128d
//   upstream sha256  b13a4fd71e25fdf24281d1a740b03589a09df1ffe53415fa7ba561abd8c6a47b
//                    (recorded in fall platforms/rknn/native/SHA256SUMS)
//   upstream license Apache-2.0
//
// Changes made for BASE-1 M2.1 (Apache-2.0 §4(b) notice):
//   1. the extern "C" export layer (hybrid_create / hybrid_* accessors /
//      hybrid_infer_*_fd / hybrid_destroy) is replaced by the vb::RknnHybrid
//      class; last-error state moved from a thread_local std::string to an
//      `std::string& err` out-parameter;
//   2. the "source dimensions must equal model input" restriction
//      (upstream hybrid_rga_rknn.cpp:214-215 and :260-263) is removed: the NPU
//      input canvas is pre-filled with the 114 grey letterbox value once, and
//      RGA writes the scaled source into the [pad_x,pad_y,scaled_w,scaled_h]
//      rectangle of that canvas instead of covering it;
//   3. the checksum-only path (hybrid_infer_nv12_fd) is dropped; the surviving
//      path returns dequantised float outputs, as the TensorView decoders need.
//
// The zero-copy structure is unchanged: RGA reads the decoded NV12 DMA-BUF and
// writes into the NPU input tensor memory (rknn_create_mem + rknn_set_io_mem),
// so no frame pixel is copied through the CPU.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "hybrid_rga_rknn.h"
#include "vb/backend.h"
#include "vb/rate_crop.h"

#include "rga/im2d.h"
#include "rga/rga.h"
#include "rknn_api.h"

namespace vb {
namespace {

using Clock = std::chrono::steady_clock;

// Concurrent BLITs from independent MPP pipelines fail on the measured RK3588
// driver. A process-wide queue preserves hardware preprocessing while avoiding
// that driver race (upstream comment kept).
std::mutex g_rga_mutex;

// Letterbox padding value; the same 114 grey the Python and C++ preprocessing
// paths fill with (core-py/vision_base/letterbox.py, src/letterbox.cpp callers).
constexpr uint8_t kLetterboxFill = 114;

// Byte-for-byte the same rounding as LetterboxGeom::fit() (src/letterbox.cpp):
// Python round() is banker's rounding, and the rectangle RGA writes has to
// agree with the geometry the decoder is told about.
int py_round(float v) {
    float f = std::floor(v);
    float r = v - f;
    if (r > 0.5f) return static_cast<int>(f) + 1;
    if (r < 0.5f) return static_cast<int>(f);
    return (static_cast<int>(f) % 2 == 0) ? static_cast<int>(f)
                                          : static_cast<int>(f) + 1;
}

std::vector<uint8_t> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto size = file.tellg();
    if (size <= 0) return {};
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), size);
    return file ? data : std::vector<uint8_t>{};
}

bool dimensions(const rknn_tensor_attr& attr, int* width, int* height) {
    if (attr.n_dims != 4) return false;
    if (attr.fmt == RKNN_TENSOR_NCHW) {
        *height = static_cast<int>(attr.dims[2]);
        *width = static_cast<int>(attr.dims[3]);
    } else {
        *height = static_cast<int>(attr.dims[1]);
        *width = static_cast<int>(attr.dims[2]);
    }
    return *width > 0 && *height > 0;
}

}  // namespace

std::mutex& rknn_rga_mutex() { return g_rga_mutex; }

struct RknnHybrid::Impl {
    rknn_context ctx{};
    rknn_tensor_attr input_native{};
    rknn_tensor_attr input_logical{};
    rknn_tensor_mem* input_mem = nullptr;
    std::vector<rknn_tensor_attr> output_native;
    std::vector<rknn_tensor_attr> output_logical;
    std::vector<rknn_tensor_mem*> output_mems;
    rknn_tensor_mem* snapshot_mem = nullptr;
    size_t snapshot_bytes = 0;
    int snapshot_pitch = 0;
    std::mutex run_mutex;

    ~Impl() {
        for (auto* mem : output_mems)
            if (mem) rknn_destroy_mem(ctx, mem);
        if (input_mem) rknn_destroy_mem(ctx, input_mem);
        if (snapshot_mem) rknn_destroy_mem(ctx, snapshot_mem);
        if (ctx) rknn_destroy(ctx);
    }
};

RknnHybrid::RknnHybrid() : impl_(new Impl()) {}

RknnHybrid::~RknnHybrid() = default;

bool RknnHybrid::copy_snapshot_rgb(const FrameBuf& frame,
                                   std::vector<uint8_t>& pixels, int& w, int& h,
                                   std::string& err) {
    pixels.clear();
    w = 0;
    h = 0;
    err.clear();
    const int source_w = frame.full_w > 0 ? frame.full_w : frame.w;
    const int source_h = frame.full_h > 0 ? frame.full_h : frame.h;
    if (frame.mem != Mem::DmaBuf || frame.fmt != PixFmt::NV12 ||
        frame.dmabuf_fd < 0 || frame.w <= 0 || frame.h <= 0 ||
        (frame.full_w > 0 && frame.full_w != frame.w) ||
        (frame.full_h > 0 && frame.full_h != frame.h) || frame.crop_x0 != 0 ||
        frame.crop_y0 != 0 || source_w != frame.w || source_h != frame.h) {
        err = "rknn snapshot requires a complete NV12 DMA-BUF source";
        return false;
    }
    size_t source_bytes = 0;
    if (!nv12_storage_bytes(source_w, source_h, frame.stride, frame.hstride,
                            source_bytes)) {
        err = "rknn snapshot source NV12 layout is invalid";
        return false;
    }
    const size_t tight_row = static_cast<size_t>(source_w) * 3u;
    if (tight_row > kMaxHostRgbBytes ||
        static_cast<size_t>(source_h) > kMaxHostRgbBytes / tight_row) {
        err = "rknn snapshot RGB layout exceeds 64 MiB";
        return false;
    }
    const int pitch = (source_w + 15) & ~15;
    if (pitch < source_w || static_cast<size_t>(pitch) > kMaxHostRgbBytes / 3u ||
        static_cast<size_t>(source_h) >
            kMaxHostRgbBytes / (static_cast<size_t>(pitch) * 3u)) {
        err = "rknn snapshot padded RGB layout exceeds 64 MiB";
        return false;
    }
    const size_t padded_bytes = static_cast<size_t>(pitch) *
                                static_cast<size_t>(source_h) * 3u;
    const size_t tight_bytes = tight_row * static_cast<size_t>(source_h);
    Impl& s = *impl_;
    std::lock_guard<std::mutex> run_guard(s.run_mutex);
    if (!s.snapshot_mem || s.snapshot_pitch != pitch ||
        s.snapshot_bytes < padded_bytes) {
        rknn_tensor_mem* next = rknn_create_mem(s.ctx, padded_bytes);
        if (!next || next->fd < 0 || !next->virt_addr || next->size < padded_bytes) {
            if (next) rknn_destroy_mem(s.ctx, next);
            err = "rknn snapshot scratch allocation/mapping failed";
            return false;
        }
        if (s.snapshot_mem) rknn_destroy_mem(s.ctx, s.snapshot_mem);
        s.snapshot_mem = next;
        s.snapshot_bytes = next->size;
        s.snapshot_pitch = pitch;
    }
    const int source_hstride = frame.hstride == 0 ? source_h : frame.hstride;
    rga_buffer_t src = wrapbuffer_fd(frame.dmabuf_fd, source_w, source_h,
                                     RK_FORMAT_YCbCr_420_SP, frame.stride,
                                     source_hstride);
    rga_buffer_t dst = wrapbuffer_fd(s.snapshot_mem->fd, source_w, source_h,
                                     RK_FORMAT_RGB_888, pitch, source_h);
    im_rect rect{0, 0, source_w, source_h};
    IM_STATUS status;
    int sync_rc = RKNN_SUCC;
    {
        std::lock_guard<std::mutex> rga_guard(rknn_rga_mutex());
        status = improcess(src, dst, {}, rect, rect, {}, -1, nullptr, nullptr,
                           IM_SYNC);
        if (status == IM_STATUS_SUCCESS) {
            sync_rc = rknn_mem_sync(s.ctx, s.snapshot_mem,
                                    RKNN_MEMORY_SYNC_FROM_DEVICE);
            if (sync_rc != RKNN_SUCC)
                err = "rknn snapshot mem sync from device=" +
                      std::to_string(sync_rc);
        }
    }
    if (status != IM_STATUS_SUCCESS || sync_rc != RKNN_SUCC) {
        if (err.empty())
            err = std::string("rknn snapshot RGA=") + std::to_string(status) +
                  " " + imStrError(status);
        return false;
    }
    std::vector<uint8_t> next_pixels(tight_bytes);
    const auto* base = static_cast<const uint8_t*>(s.snapshot_mem->virt_addr);
    for (int y = 0; y < source_h; ++y)
        std::memcpy(next_pixels.data() + static_cast<size_t>(y) * tight_row,
                    base + static_cast<size_t>(y) * pitch * 3u, tight_row);
    pixels.swap(next_pixels);
    w = source_w;
    h = source_h;
    return true;
}

std::unique_ptr<RknnHybrid> RknnHybrid::create(const std::string& model_path,
                                               uint32_t core_mask,
                                               std::string& err) {
    err.clear();
    auto model = read_file(model_path.c_str());
    if (model.empty()) {
        err = "cannot read model: " + model_path;
        return nullptr;
    }
    std::unique_ptr<RknnHybrid> self(new RknnHybrid());
    Impl& s = *self->impl_;

    int ret = rknn_init(&s.ctx, model.data(),
                        static_cast<uint32_t>(model.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
        err = "rknn_init=" + std::to_string(ret);
        return nullptr;
    }
    ret = rknn_set_core_mask(s.ctx, static_cast<rknn_core_mask>(core_mask));
    if (ret != RKNN_SUCC) {
        err = "rknn_set_core_mask(" + std::to_string(core_mask) +
              ")=" + std::to_string(ret);
        return nullptr;
    }

    rknn_sdk_version version{};
    if (rknn_query(s.ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) ==
        RKNN_SUCC) {
        self->sdk_version_ = std::string(version.api_version) + "/" +
                             std::string(version.drv_version);
    }

    rknn_input_output_num io{};
    ret = rknn_query(s.ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret != RKNN_SUCC || io.n_input != 1 || io.n_output == 0) {
        err = "unexpected model IO: ret=" + std::to_string(ret) +
              " inputs=" + std::to_string(io.n_input) +
              " outputs=" + std::to_string(io.n_output);
        return nullptr;
    }
    s.input_logical.index = 0;
    ret = rknn_query(s.ctx, RKNN_QUERY_INPUT_ATTR, &s.input_logical,
                     sizeof(s.input_logical));
    if (ret != RKNN_SUCC ||
        !dimensions(s.input_logical, &self->width_, &self->height_)) {
        err = "RKNN_QUERY_INPUT_ATTR=" + std::to_string(ret);
        return nullptr;
    }
    s.input_native.index = 0;
    ret = rknn_query(s.ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &s.input_native,
                     sizeof(s.input_native));
    if (ret != RKNN_SUCC) {
        err = "RKNN_QUERY_NATIVE_INPUT_ATTR=" + std::to_string(ret);
        return nullptr;
    }
    // Rockchip's official zero-copy example keeps the native layout but changes
    // the external element type to UINT8, allowing normalization/quantization to
    // remain fused in the NPU graph.
    s.input_native.type = RKNN_TENSOR_UINT8;
    // Bring-up aid, same switch as the canvas dump: the RGA destination buffer is
    // wrapped with input_native.w_stride, so the unit this runtime reports
    // (bytes vs pixels) has to be visible rather than inferred.
    if (const char* dbg = std::getenv("VB_RK_DUMP_CANVAS"); dbg && *dbg) {
        std::fprintf(stderr,
                     "rknn input logical: dims=[%u,%u,%u,%u] fmt=%d type=%d "
                     "n_elems=%u size=%u w_stride=%u h_stride=%u size_with_stride=%u\n",
                     s.input_logical.dims[0], s.input_logical.dims[1],
                     s.input_logical.dims[2], s.input_logical.dims[3],
                     static_cast<int>(s.input_logical.fmt),
                     static_cast<int>(s.input_logical.type), s.input_logical.n_elems,
                     s.input_logical.size, s.input_logical.w_stride,
                     s.input_logical.h_stride, s.input_logical.size_with_stride);
        std::fprintf(stderr,
                     "rknn input native : dims=[%u,%u,%u,%u] fmt=%d n_elems=%u "
                     "size=%u w_stride=%u h_stride=%u size_with_stride=%u "
                     "model=%dx%d\n",
                     s.input_native.dims[0], s.input_native.dims[1],
                     s.input_native.dims[2], s.input_native.dims[3],
                     static_cast<int>(s.input_native.fmt), s.input_native.n_elems,
                     s.input_native.size, s.input_native.w_stride,
                     s.input_native.h_stride, s.input_native.size_with_stride,
                     self->width_, self->height_);
    }
    s.input_mem = rknn_create_mem(s.ctx, s.input_native.size_with_stride);
    if (!s.input_mem) {
        err = "rknn_create_mem(input) returned null";
        return nullptr;
    }
    if (!s.input_mem->virt_addr || s.input_mem->size == 0) {
        err = "rknn input memory is not CPU-mapped; cannot prime the letterbox";
        return nullptr;
    }
    // Pre-fill the whole canvas with the letterbox grey. RGA only ever writes
    // the [pad_x,pad_y,scaled_w,scaled_h] rectangle, so the padding survives
    // every frame and does not have to be re-filled per frame.
    std::memset(s.input_mem->virt_addr, kLetterboxFill, s.input_mem->size);
    ret = rknn_mem_sync(s.ctx, s.input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN_SUCC) {
        err = "rknn_mem_sync(input fill)=" + std::to_string(ret);
        return nullptr;
    }
    ret = rknn_set_io_mem(s.ctx, s.input_mem, &s.input_native);
    if (ret != RKNN_SUCC) {
        err = "rknn_set_io_mem(input)=" + std::to_string(ret);
        return nullptr;
    }

    s.output_native.resize(io.n_output);
    s.output_logical.resize(io.n_output);
    s.output_mems.resize(io.n_output, nullptr);
    self->outputs_.resize(io.n_output);
    size_t total_elems = 0;
    for (uint32_t i = 0; i < io.n_output; ++i) {
        auto& logical = s.output_logical[i];
        logical.index = i;
        ret = rknn_query(s.ctx, RKNN_QUERY_OUTPUT_ATTR, &logical, sizeof(logical));
        if (ret != RKNN_SUCC) {
            err = "RKNN_QUERY_OUTPUT_ATTR[" + std::to_string(i) +
                  "]=" + std::to_string(ret);
            return nullptr;
        }
        auto& attr = s.output_native[i];
        attr.index = i;
        ret = rknn_query(s.ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) {
            err = "RKNN_QUERY_NATIVE_OUTPUT_ATTR[" + std::to_string(i) +
                  "]=" + std::to_string(ret);
            return nullptr;
        }
        s.output_mems[i] = rknn_create_mem(s.ctx, attr.size_with_stride);
        if (!s.output_mems[i]) {
            err = "rknn_create_mem(output) returned null";
            return nullptr;
        }
        ret = rknn_set_io_mem(s.ctx, s.output_mems[i], &attr);
        if (ret != RKNN_SUCC) {
            err = "rknn_set_io_mem(output)=" + std::to_string(ret);
            return nullptr;
        }
        RknnOutputDesc& desc = self->outputs_[i];
        desc.n_elems = logical.n_elems;
        desc.dims.reserve(logical.n_dims);
        for (uint32_t d = 0; d < logical.n_dims; ++d) desc.dims.push_back(logical.dims[d]);
        total_elems += logical.n_elems;
    }
    if (total_elems == 0) {
        err = "model reports no output elements";
        return nullptr;
    }
    self->flat_.assign(total_elems, 0.0f);
    return self;
}

int RknnHybrid::infer_nv12_fd(int src_fd, int src_w, int src_h, int y_stride,
                              double* rga_ms, double* rknn_ms, std::string& err,
                              int src_hstride) {
    err.clear();
    size_t source_bytes = 0;
    if (src_fd < 0 || !nv12_storage_bytes(src_w, src_h, y_stride, src_hstride, source_bytes)) {
        err = "invalid NV12 source: fd=" + std::to_string(src_fd) +
              " size=" + std::to_string(src_w) + "x" + std::to_string(src_h) +
              " y_stride=" + std::to_string(y_stride) + " hstride=" + std::to_string(src_hstride);
        return -1;
    }
    Impl& s = *impl_;
    std::lock_guard<std::mutex> run_guard(s.run_mutex);

    const int dst_stride = s.input_native.w_stride > 0
                               ? static_cast<int>(s.input_native.w_stride)
                               : width_;
    const LetterboxGeom geom =
        LetterboxGeom::fit(src_w, src_h, width_, height_, input_.align);
    int scaled_w = py_round(static_cast<float>(src_w) * geom.scale);
    int scaled_h = py_round(static_cast<float>(src_h) * geom.scale);
    // LetterboxGeom is the single source of truth for alignment and integer
    // padding. In particular, top_left keeps both offsets at zero.
    int dst_x = static_cast<int>(geom.pad_x);
    int dst_y = static_cast<int>(geom.pad_y);
    scaled_w = std::min(std::max(scaled_w, 1), width_ - dst_x);
    scaled_h = std::min(std::max(scaled_h, 1), height_ - dst_y);

    rga_buffer_t src = wrapbuffer_fd(src_fd, src_w, src_h,
                                     RK_FORMAT_YCbCr_420_SP, y_stride,
                                     src_hstride == 0 ? src_h : src_hstride);
    // The canvas is what the model sees, so its channel order comes from
    // backend.input (defaulted by decoder family), not from the frame format.
    const auto rga_fmt = (input_.color_order == ColorOrder::BGR)
                             ? RK_FORMAT_BGR_888
                             : RK_FORMAT_RGB_888;
    rga_buffer_t dst = wrapbuffer_fd(s.input_mem->fd, width_, height_,
                                     rga_fmt, dst_stride, height_);
    im_rect srect{0, 0, src_w, src_h};
    im_rect drect{dst_x, dst_y, scaled_w, scaled_h};
    im_rect prect{0, 0, 0, 0};

    const auto before_rga = Clock::now();
    IM_STATUS rga_status;
    {
        std::lock_guard<std::mutex> rga_guard(g_rga_mutex);
        // One RGA pass does NV12 -> RGB/BGR, the aspect-preserving scale and the
        // placement inside the canvas; the grey fill stays outside drect.
        rga_status = improcess(src, dst, {}, srect, drect, prect, -1, nullptr,
                               nullptr, IM_SYNC);
    }
    const auto after_rga = Clock::now();
    if (rga_status != IM_STATUS_SUCCESS) {
        err = std::string("rga improcess=") + std::to_string(rga_status) + " " +
              imStrError(rga_status);
        return -3;
    }

    const int run_status = rknn_run(s.ctx, nullptr);
    const auto after_rknn = Clock::now();
    if (run_status != RKNN_SUCC) {
        err = "rknn_run=" + std::to_string(run_status);
        return -4;
    }

    // Bring-up aid: VB_RK_DUMP_CANVAS=<path> writes the first frame's NPU input
    // canvas (tightly packed RGB888) so it can be fed to rknnlite for
    // comparison. Off by default; never enabled by the runtime itself.
    if (!dump_done_) {
        dump_done_ = true;
        // Verbatim copy of the whole NPU input buffer (size_with_stride bytes):
        // the tight repack below assumes a byte-per-row stride, which is exactly
        // what the wrap has to be checked against.
        const char* raw_path = std::getenv("VB_RK_DUMP_CANVAS_RAW");
        if (raw_path && *raw_path && s.input_mem->virt_addr && s.input_mem->size) {
            std::FILE* rf = std::fopen(raw_path, "wb");
            if (rf) {
                std::fwrite(s.input_mem->virt_addr, 1, s.input_mem->size, rf);
                std::fclose(rf);
                std::fprintf(stderr, "rknn: raw canvas %zu bytes -> %s\n",
                             s.input_mem->size, raw_path);
            }
        }
        const char* dump_path = std::getenv("VB_RK_DUMP_CANVAS");
        if (dump_path && *dump_path) {
            const uint8_t* base = static_cast<const uint8_t*>(s.input_mem->virt_addr);
            std::vector<uint8_t> tight(static_cast<size_t>(width_) * height_ * 3);
            for (int y = 0; y < height_; ++y)
                // dst_stride is the RGA wrap stride, in pixels; the canvas is
                // three bytes per pixel, so the row offset is stride * 3.
                std::memcpy(tight.data() + static_cast<size_t>(y) * width_ * 3,
                            base + static_cast<size_t>(y) * dst_stride * 3,
                            static_cast<size_t>(width_) * 3);
            std::FILE* f = std::fopen(dump_path, "wb");
            if (f) {
                std::fwrite(tight.data(), 1, tight.size(), f);
                std::fclose(f);
            }
        }
    }

    std::vector<rknn_output> outs(outputs_.size());
    uint64_t offset = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        outs[i].index = static_cast<uint32_t>(i);
        outs[i].want_float = 1;
        outs[i].is_prealloc = 1;
        outs[i].buf = flat_.data() + offset;
        outs[i].size = outputs_[i].n_elems * sizeof(float);
        offset += outputs_[i].n_elems;
    }
    const int get_status =
        rknn_outputs_get(s.ctx, static_cast<uint32_t>(outs.size()), outs.data(),
                         nullptr);
    if (get_status != RKNN_SUCC) {
        err = "rknn_outputs_get=" + std::to_string(get_status);
        return -5;
    }
    const int release_status = rknn_outputs_release(
        s.ctx, static_cast<uint32_t>(outs.size()), outs.data());
    if (release_status != RKNN_SUCC) {
        err = "rknn_outputs_release=" + std::to_string(release_status);
        return -6;
    }
    if (rga_ms)
        *rga_ms = std::chrono::duration<double, std::milli>(after_rga - before_rga)
                      .count();
    if (rknn_ms)
        *rknn_ms =
            std::chrono::duration<double, std::milli>(after_rknn - after_rga)
                .count();
    geom_ = geom;
    return 0;
}

}  // namespace vb
