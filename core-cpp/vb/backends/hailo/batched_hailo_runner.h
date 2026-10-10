// HailoRT batched inference session (spec BASE-1 §M2.3).
//
// Copied from fall-detection platforms/rpi-hailo/src/batched_hailo_runner.{h,cpp}
// (origin/main@eb72e1e, Apache-2.0; see NOTICE) and aligned to §6.1: the fall
// worker pulled frames from a FrameBatcher and pushed raw tensors to a
// callback; the vb ContextPool already batches across streams, so this class
// keeps the fall session (one VDevice, one configured HEF, pre-allocated
// DMA-mapped binding slots, zero-padded partial batches) behind a synchronous
// infer() that an InferenceContext calls. Namespaced to vb::.
//
// VB_HAILO_STUB=ON builds a fake runner with the same interface and no
// HailoRT: it synthesises the 9 pose outputs for every input frame, which is
// what the offline tests decode.
//
// RawTensor mirrors the consumed fields of hailo_vstream_info_t (shape,
// format type, quantisation) so the stub build needs no HailoRT headers. The
// real runner fills it from the HEF's own vstream infos; the dequantised
// TensorView handed to the §6.11 decoder is built by hailo_backend.cpp.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "frame_batcher.h"

namespace vb {

// hailo_format_type_t values this backend consumes (hailort.h enum values,
// mirrored so the stub build compiles without HailoRT headers).
constexpr uint8_t kHailoFormatUint8 = 1;
constexpr uint8_t kHailoFormatUint16 = 2;

struct HailoVStreamInfo {
    // HEF output shape, batch removed: [features, height, width].
    int32_t features = 0, height = 0, width = 0;
    uint8_t format_type = kHailoFormatUint8;  // kHailoFormatUint8 / Uint16
    float qp_scale = 1.0f;                    // dequant: (raw - qp_zp) * qp_scale
    int32_t qp_zp = 0;
};

struct RawTensor {
    const uint8_t* data = nullptr;
    HailoVStreamInfo info;
};

struct HefContextInfo {
    int network_groups = 0;
    bool multi_context = false;
};

// The fall ResultHandler signature (frame + raw tensors) kept for the
// batcher-shaped path; the synchronous entry used by the backend is infer().
using HailoResultHandler = std::function<void(BatchFrame&&, const std::vector<RawTensor>&)>;
using HailoErrorHandler = std::function<void(const std::string&)>;

class BatchedHailoRunner {
public:
    static HefContextInfo inspectHef(const std::string& hef_path);

    BatchedHailoRunner(const std::string& hef_path, int batch_size,
                       FrameBatcher& batcher, HailoResultHandler on_result,
                       HailoErrorHandler on_error);
    ~BatchedHailoRunner();
    BatchedHailoRunner(const BatchedHailoRunner&) = delete;
    BatchedHailoRunner& operator=(const BatchedHailoRunner&) = delete;

    void start();
    void stop();

    // §6.1 alignment: run one batch synchronously. frames[0..n) are consumed
    // (n <= batch size, zero-padded exactly like fall's infer()); returns the
    // raw output tensors for each frame slot 0..n-1, valid until the next
    // infer() call on this runner.
    int infer(const std::vector<BatchFrame>& frames, size_t n,
              std::vector<std::vector<RawTensor>>& out, std::string& err);

    int batch_size() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    int batch_size_ = 0;
};

// ---- VB_HAILO_STUB test hooks ----
// The stub runner returns these raw outputs for every frame slot on every
// infer() call (bytes are copied into stub-owned storage and stay alive until
// the next set_outputs/clear). With no configuration the stub emits
// zero-filled buffers with the 9 standard pose-output infos, which decode to
// zero detections.
#ifdef VB_HAILO_STUB
namespace hailo_stub {
void set_outputs(std::vector<HailoVStreamInfo> infos,
                 std::vector<std::vector<uint8_t>> bytes);
void clear_outputs();
}  // namespace hailo_stub
#endif

}  // namespace vb
