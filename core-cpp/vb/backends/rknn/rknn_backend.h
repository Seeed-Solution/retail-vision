// RK3576 / RK3588 backend factory (spec BASE-1 §M2.1).
#pragma once

#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// backend JSON keys:
//   model_path     (required) .rknn file
//   model_sha256   (optional) 64 hex digits; mismatch refuses to load
//   decoder        (optional) §6.11 decoder object; default yolox
//   core_masks     (optional) array of rknn_core_mask bit sets, one per context
//   max_contexts   (optional) Caps.max_contexts, default = core_masks length
std::unique_ptr<Backend> make_rknn_backend(const std::string& backend_json,
                                           std::string& err);

}  // namespace vb
