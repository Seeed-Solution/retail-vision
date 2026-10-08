// DMA-BUF -> RGA -> RKNN zero-copy inference context (spec BASE-1 §M2.1).
//
// Derived from fall-detection platforms/rknn/native/src/hybrid_rga_rknn.cpp
// @ 5af128d (Apache-2.0); see hybrid_rga_rknn.cpp for the source line, the
// upstream sha256 and the list of changes.
//
// The class owns one rknn_context whose input and output tensors are backed by
// rknn_create_mem() allocations, so a frame never leaves the device: RGA reads
// the decoded NV12 DMA-BUF and writes directly into the NPU input memory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vb/types.h"

namespace vb {

struct FrameBuf;
std::mutex& rknn_rga_mutex();

// One RKNN output tensor, batch dimension kept as reported by
// RKNN_QUERY_OUTPUT_ATTR (the adapter drops it when it is 1).
struct RknnOutputDesc {
    uint32_t n_elems = 0;
    std::vector<uint32_t> dims;
};

// Not thread-safe: one instance per InferenceContext, and infer_nv12_fd()
// serialises on an internal run mutex.
class RknnHybrid {
public:
    // Loads `model_path` and applies `core_mask` (a rknn_core_mask bit set).
    // Returns nullptr with err set on any failure.
    static std::unique_ptr<RknnHybrid> create(const std::string& model_path,
                                              uint32_t core_mask, std::string& err);
    ~RknnHybrid();
    RknnHybrid(const RknnHybrid&) = delete;
    RknnHybrid& operator=(const RknnHybrid&) = delete;

    int model_width() const { return width_; }
    int model_height() const { return height_; }
    const std::vector<RknnOutputDesc>& outputs() const { return outputs_; }
    const std::string& sdk_version() const { return sdk_version_; }

    // RGA converts+scales the NV12 DMA-BUF `src_fd` into the letterbox rectangle
    // of the NPU input memory (the rest of the canvas keeps the 114 grey fill),
    // then runs the model. The dequantised float outputs of this call are
    // available through flat() until the next call.
    //
    // Returns 0 on success, negative on failure with err set.
    int infer_nv12_fd(int src_fd, int src_w, int src_h, int y_stride,
                      double* rga_ms, double* rknn_ms, std::string& err,
                      int src_hstride = 0);

    bool copy_snapshot_rgb(const FrameBuf& frame, std::vector<uint8_t>& pixels,
                           int& w, int& h, std::string& err);

    // Flat, output-major float outputs of the most recent infer_nv12_fd().
    const float* flat() const { return flat_.data(); }
    size_t flat_count() const { return flat_.size(); }

    // Letterbox geometry used by the most recent infer_nv12_fd(). Valid after
    // the first call (default-constructed before that).
    const LetterboxGeom& last_geom() const { return geom_; }

    // What the model's input tensor expects (spec §6.11 backend.input).
    // RGA writes the canvas directly, so the channel order is applied here;
    // RKNN models take the uint8 canvas, so InputSpec::divide does not apply.
    void set_input(const InputSpec& s) { input_ = s; }

private:
    RknnHybrid();

    struct Impl;
    std::unique_ptr<Impl> impl_;

    int width_ = 0, height_ = 0;
    std::vector<RknnOutputDesc> outputs_;
    std::vector<float> flat_;
    LetterboxGeom geom_{};
    InputSpec input_{};
    bool dump_done_ = false;  // VB_RK_DUMP_CANVAS writes at most one canvas
    std::string sdk_version_;
};

}  // namespace vb
