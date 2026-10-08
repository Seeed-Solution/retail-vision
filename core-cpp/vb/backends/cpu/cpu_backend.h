// CPU inference backend (spec BASE-1 §8 M1.9): ONNX Runtime via the C API.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <onnxruntime_c_api.h>

#include "vb/backend.h"

namespace vb {

const OrtApi* cpu_ort_api();
OrtEnv* cpu_ort_env(std::string& err);

// Resource bounds, all checked before anything is allocated. §7 keeps models
// external, so a model path is untrusted input like any other config value.
inline constexpr std::size_t kCpuMaxModelBytes = 256u * 1024 * 1024;
inline constexpr int64_t kCpuMaxModelDim = 4096;               // NCHW side
inline constexpr int64_t kCpuMaxInputBytes = 128ll * 1024 * 1024;  // float buffer

// B5: maps an ONNX Runtime tensor element type to the VBT1 (spec §6.12) dtype
// code and the element width in bytes used to copy the tensor. VBT1 defines
// only dtype 0 = float32 and reserves 1-255 (contracts/vb-wire.md), so every
// other element type is refused: copying an int8/f16 tensor as if it were f32
// reads past the tensor and mislabels the record. Inline so the mapping can be
// exercised by test targets that link vb_core only.
inline bool cpu_vbt1_dtype(int32_t ort_element_type, uint8_t& vbt1_dtype,
                           std::size_t& elem_bytes, std::string& err) {
    // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == 1 in onnxruntime_c_api.h; the ORT
    // headers are not visible outside the backend target, hence the literal.
    if (ort_element_type == 1) {
        vbt1_dtype = 0;  // kDevDtypeF32
        elem_bytes = 4;
        return true;
    }
    err = "raw tensor element type " + std::to_string(ort_element_type) +
          " is not float32; VBT1 carries f32 only";
    return false;
}

// backend_json keys:
//   model_path (required)  path to the ONNX model (YOLOX-style head,
//                          output [1, A, 5 + n_cls], M1.9 decoder = yolox)
//   intra_threads (opt)    ORT intra-op threads (default: 2)
//   model_sha256 (opt)     expected SHA-256 of the model file; the file is
//                          read once, verified, and loaded from those bytes
std::unique_ptr<Backend> make_cpu_backend(const std::string& backend_json, std::string& err);
std::unique_ptr<Stage2Context> make_cpu_stage2(const Stage2Spec& spec, std::string& err);

}  // namespace vb
