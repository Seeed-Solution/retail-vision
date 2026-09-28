// Backend-agnostic decoder selection (spec BASE-1 §6.11, M1.15).
//
// Platform adapters call make_decoder() with the `backend.decoder` config
// object; an empty string yields nullptr (the adapter keeps its own
// default decode path, e.g. the CPU backend's YOLOX head).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vb/types.h"
#include "vb/backend.h"  // DetectionResult

namespace vb {

// Parsed decoder parameters (§6.11 table).
struct DecodeSpec {
    std::string type;
    int num_classes = 80;
    int reg_max = 16;
    std::vector<int> strides{8, 16, 32};
    int keypoints = 17;
    int top_k = 1;
    bool softmax = true;
};

// One model output tensor with the batch dimension already removed,
// row-major float data. Adapters without named outputs leave name empty.
struct TensorView {
    const float* data = nullptr;
    size_t count = 0;
    std::vector<int64_t> dims;  // batch removed, e.g. [4+nc, N]
    std::string name;
};

class Decoder {
public:
    virtual ~Decoder() = default;

    // Goes into Caps.keypoints.
    virtual int keypoints() const = 0;

    // Decodes outputs into model-canvas-normalized detections (internal NMS
    // included). Returns false with err on shape mismatch.
    virtual bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                        float score, float nms, DetectionResult& out,
                        std::string& err) = 0;
};

// Empty (or missing) decoder_json -> nullptr, err empty (adapter default).
// Errors: unknown type, "ctc" (stage2-only), bad parameters.
std::unique_ptr<Decoder> make_decoder(const std::string& decoder_json,
                                      std::string& err);

}  // namespace vb
