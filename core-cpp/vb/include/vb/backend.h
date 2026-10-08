// Platform adaptation interface: FrameBuf / FrameSource / InferenceContext /
// Backend + the name-keyed backend registry (spec BASE-1 §6.1).
//
// Core code never branches on backend names; platform adapters register
// themselves via register_backend() (§5.2.3).
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vb/types.h"
#include "vb/analyzer.h"

namespace vb {

struct TensorView;

enum class PixFmt : uint8_t { RGB888 = 0, BGR888 = 1, NV12 = 2 };
enum class Mem : uint8_t { Host = 0, DmaBuf = 1, Cuda = 2, Opaque = 3 };

struct FrameBuf {  // frame-source output; pixels are not copied
    uint32_t stream_index = 0;
    uint64_t seq = 0;
    double wall_ms = 0, t_mono_s = 0;
    int32_t w = 0, h = 0, stride = 0;
    // NV12 luma storage rows, distinct from visible h; 0 preserves h-stride
    // behavior for callers with tightly packed planes. Ignored for host RGB.
    int32_t hstride = 0;
    PixFmt fmt = PixFmt::RGB888;
    Mem mem = Mem::Host;
    const uint8_t* host = nullptr;
    // Original host backing store. For a host ROI, host points into this
    // allocation while full_host/full_stride preserve the source image.
    const uint8_t* full_host = nullptr;
    int32_t full_stride = 0;
    int dmabuf_fd = -1;
    void* opaque = nullptr;
    bool letterboxed = false;
    LetterboxGeom geom{};
    int32_t full_w = 0, full_h = 0, crop_x0 = 0, crop_y0 = 0;
    std::shared_ptr<void> hold;  // destruction returns the frame to its source

    FrameBuf() = default;
    // Movable like a plain struct (shared_ptr member is movable).
    FrameBuf(FrameBuf&&) = default;
    FrameBuf& operator=(FrameBuf&&) = default;
    FrameBuf(const FrameBuf&) = delete;
    FrameBuf& operator=(const FrameBuf&) = delete;
};

// The single-fd RGA NV12 layout uses equal Y/UV row strides and places UV
// immediately after stride * storage_h bytes. No arbitrary plane offsets.
inline bool nv12_storage_bytes(int w, int h, int stride, int hstride,
                               size_t& bytes) {
    bytes = 0;
    const int storage_h = hstride == 0 ? h : hstride;
    if (w <= 0 || h <= 0 || w > 8192 || h > 8192 || (w % 2) || (h % 2) ||
        stride < w || (stride % 2) || storage_h < h || storage_h > 8192 ||
        (storage_h % 2)) return false;
    const size_t rows = static_cast<size_t>(storage_h) + storage_h / 2;
    if (static_cast<size_t>(stride) > std::numeric_limits<size_t>::max() / rows)
        return false;
    bytes = static_cast<size_t>(stride) * rows;
    return true;
}

inline bool nv12_dmabuf_layout(int w, int h, int y_stride, int uv_stride,
                               int planes, size_t y_offset, size_t uv_offset,
                               size_t memory_offset, size_t memory_size,
                               int& hstride, size_t& bytes) {
    hstride = 0;
    bytes = 0;
    if (planes != 2 || y_stride <= 0 || uv_stride != y_stride || y_offset != 0 ||
        memory_offset != 0 || uv_offset % static_cast<size_t>(y_stride) != 0)
        return false;
    const size_t rows = uv_offset / static_cast<size_t>(y_stride);
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) || rows == 0)
        return false;
    const int storage_h = static_cast<int>(rows);
    size_t needed = 0;
    if (!nv12_storage_bytes(w, h, y_stride, storage_h, needed) || needed > memory_size)
        return false;
    hstride = storage_h;
    bytes = needed;
    return true;
}

struct DetectionResult {
    std::vector<Detection> dets;
    std::vector<Keypoint> kpts;
    LetterboxGeom geom{};
    float preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0;
};

// A second-stage request is expressed in source-normalized coordinates.  The
// stage2 adapter owns the crop/preprocess scratch; callers retain the frame
// until infer_crops() returns.
struct CropReq {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0; // source-normalized
    uint32_t track_id = 0;                 // identity of the selected candidate
};

struct Stage2Spec {
    std::string model_path;
    int32_t in_h = 32;
    int32_t in_w = 96;
    bool bgr = false;
    float mean[3] = {0, 0, 0};
    float scale = 1.0f / 255.0f;
};

class Stage2Context {
public:
    virtual ~Stage2Context() = default;
    virtual int infer_crops(const FrameBuf& frame, const CropReq* crops, size_t n,
                            TensorView* outs, std::string& err) = 0;
    virtual int infer_rgb(const uint8_t* rgb, int w, int h, int stride,
                          TensorView* out, std::string& err) = 0;
};

struct StreamSpec {
    uint32_t index = 0;
    std::string id, url, name, transport;
    float score_threshold = 0.35f;
    std::string options_json = "{}";
    float max_fps = 0.0f;                         // B8; 0 = unlimited
    bool has_roi_crop = false;
    float roi_crop[4] = {0, 0, 1, 1};             // B8 [x0,y0,x1,y1] source-norm
};

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual bool open(std::string& err) = 0;
    virtual int read(FrameBuf& out, int timeout_ms) = 0;  // 1 frame, 0 timeout, -1 lost
    virtual void close() = 0;
    virtual const char* decode_path() const = 0;
    virtual bool set_crop(int x0_px, int y0_px, int w_px, int h_px) {
        (void)x0_px; (void)y0_px; (void)w_px; (void)h_px;
        return false;
    }
};

struct Caps {
    int max_contexts = 1;
    int max_batch = 1;
    int keypoints = 0;
    bool exclusive_device = false;
};

class InferenceContext {
public:
    virtual ~InferenceContext() = default;
    // frames[0..n) come from distinct streams, n <= caps.max_batch.
    virtual int infer(const FrameBuf* const* frames, size_t n, float score, float nms,
                      DetectionResult* out, std::string& err) = 0;
    // Optional snapshot conversion for non-host frames. Implementations must
    // return a tightly packed RGB copy of the original full source image,
    // using its full_w/full_h (or w/h when unset). The pool validates that
    // RGB dimensions fit its host snapshot budget before calling this method.
    virtual bool copy_snapshot_rgb(const FrameBuf& frame, std::vector<uint8_t>& pixels,
                                   int& w, int& h, std::string& err) {
        (void)frame; (void)pixels; (void)w; (void)h;
        err = "snapshot RGB conversion unsupported";
        return false;
    }
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual Caps caps() const = 0;
    virtual std::pair<int, int> model_hw() const = 0;
    virtual std::string model_sha256() const = 0;
    virtual std::unique_ptr<FrameSource> create_source(const StreamSpec& s, std::string& err) = 0;
    virtual std::unique_ptr<InferenceContext> create_context(int index, std::string& err) = 0;
    // Optional second-stage recognizer. Backends that cannot run stage2 must
    // return nullptr and a descriptive error; no backend is silently treated
    // as CPU-compatible.
    virtual std::unique_ptr<Stage2Context> create_stage2(const Stage2Spec& spec,
                                                         std::string& err) {
        (void)spec;
        err = "backend does not provide stage2";
        return nullptr;
    }
};

using BackendFactory = std::unique_ptr<Backend> (*)(const std::string& backend_json,
                                                    std::string& err);

// Platform adapters (backends/<p>/register.cpp) register via a static object.
bool register_backend(const char* name, BackendFactory f);

// Returns nullptr with err set for unknown names or factory failure.
std::unique_ptr<Backend> create_backend(const std::string& name,
                                        const std::string& backend_json,
                                        std::string& err);

}  // namespace vb
