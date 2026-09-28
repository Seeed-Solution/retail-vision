// Platform adaptation interface: FrameBuf / FrameSource / InferenceContext /
// Backend + the name-keyed backend registry (spec BASE-1 §6.1).
//
// Core code never branches on backend names; platform adapters register
// themselves via register_backend() (§5.2.3).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vb/types.h"

namespace vb {

enum class PixFmt : uint8_t { RGB888 = 0, BGR888 = 1, NV12 = 2 };
enum class Mem : uint8_t { Host = 0, DmaBuf = 1, Cuda = 2, Opaque = 3 };

struct FrameBuf {  // frame-source output; pixels are not copied
    uint32_t stream_index = 0;
    uint64_t seq = 0;
    double wall_ms = 0, t_mono_s = 0;
    int32_t w = 0, h = 0, stride = 0;
    PixFmt fmt = PixFmt::RGB888;
    Mem mem = Mem::Host;
    const uint8_t* host = nullptr;
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

struct DetectionResult {
    std::vector<Detection> dets;
    std::vector<Keypoint> kpts;
    LetterboxGeom geom{};
    float preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0;
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
