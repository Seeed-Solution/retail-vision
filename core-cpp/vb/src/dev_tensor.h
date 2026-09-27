// Dev-mode raw tensor passthrough VBT1 (spec BASE-1 §6.12, M1.23a).
//
// Only active when the config carries dev.raw_tensors=true, the process was
// started with --dev (allow_dev), and backend.decoder.type == "raw". Limited
// to dev.max_fps per stream, dev.max_streams == 1, and a 16 MiB record cap.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vb/json.h"

namespace vb {

// dtype codes for DevTensor::dtype (VBT1 u8 dtype).
constexpr uint8_t kDevDtypeF32 = 0;  // float32, little-endian
// 1..255 reserved.

// Whole-record cap (magic + body_len + body), §6.12.
constexpr size_t kDevTensorMaxRecord = 16u * 1024 * 1024;

// One raw model output tensor (batch dimension removed).
struct DevTensor {
    uint8_t dtype = kDevDtypeF32;
    bool nhwc = false;                 // false: row-major as exported (NCHW etc.)
    std::vector<int32_t> dims;         // up to 4 entries
    float scale = 1.0f;                // dequant scale (1.0 for f32)
    int32_t zero_point = 0;
    std::string name;                  // utf-8, may be empty
    std::vector<uint8_t> data;         // raw little-endian bytes
};

// VBT1 body content (one frame's raw outputs).
struct DevTensorFrame {
    uint32_t stream_index = 0;
    uint64_t seq = 0;
    double wall_ms = 0;
    int32_t src_w = 0, src_h = 0, model_w = 0, model_h = 0;
    float scale = 0, pad_x = 0, pad_y = 0;
    uint8_t align = 0;  // 0 center, 1 top-left
    std::vector<DevTensor> tensors;
};

// Appends the full VBT1 record (8-byte header + body) to out. Returns false
// with err set and leaves out untouched when the frame cannot be encoded
// faithfully: more than 65535 tensors, a tensor with more than 4 dims, a name
// longer than 65535 bytes, data larger than u32, or a record above
// kDevTensorMaxRecord (the caller drops it and counts dev_tensor_oversize).
bool wire_encode_vbt1(const DevTensorFrame& f, std::vector<uint8_t>& out,
                      std::string& err);

// Per-stream VBT1 rate limiter (dev.max_fps <= 2). First frame passes.
class DevRateLimiter {
public:
    explicit DevRateLimiter(double max_fps = 1.0) { set_max_fps(max_fps); }
    void set_max_fps(double max_fps) {
        min_interval_s_ = max_fps > 0.0 ? 1.0 / max_fps : 0.0;
    }
    bool try_send(double now_s) {
        if (last_sent_s_ >= 0.0 && now_s - last_sent_s_ < min_interval_s_)
            return false;
        last_sent_s_ = now_s;
        return true;
    }

private:
    double min_interval_s_ = 1.0;
    double last_sent_s_ = -1.0;
};

// Top-level `dev` config object (contracts/vb-config.schema.json).
struct DevConfig {
    bool raw_tensors = false;
    double max_fps = 1.0;
    int max_streams = 1;
};

// Parses the "dev" member of a config (absent or null -> defaults, no error).
// Errors: raw_tensors without allow_dev ("dev.raw_tensors is not allowed in
// production configs"), max_fps outside (0, 2], max_streams != 1.
bool dev_config_from_json(const Json& config, bool allow_dev, DevConfig& out,
                          std::string& err);

// Optional interface an InferenceContext can implement to expose the raw
// outputs of its most recent infer() call (frames[0]). Retrieved via
// dynamic_cast; backends without raw support simply do not implement it.
class RawTensorSource {
public:
    virtual ~RawTensorSource() = default;
    // Copies the last raw outputs into out; false when none available.
    virtual bool last_raw_tensors(std::vector<DevTensor>& out) = 0;
};

}  // namespace vb
