// CPU inference backend (spec BASE-1 §8 M1.9): ONNX Runtime via the C API.
#pragma once

#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// backend_json keys:
//   model_path (required)  path to the ONNX model (YOLOX-style head,
//                          output [1, A, 5 + n_cls], M1.9 decoder = yolox)
//   intra_threads (opt)    ORT intra-op threads (default: 2)
std::unique_ptr<Backend> make_cpu_backend(const std::string& backend_json, std::string& err);

}  // namespace vb
