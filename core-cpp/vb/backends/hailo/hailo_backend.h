// Hailo (Pi 5 + Hailo-8) backend factory (spec BASE-1 §M2.3).
#pragma once

#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// backend JSON keys:
//   model_path     (required) .hef file; stub builds accept any path
//   model_sha256   (optional) 64 hex digits; mismatch refuses to load
//   decoder        (optional §6.11 object; default yolo_pose, 17 keypoints)
//   batch          (optional) {"mode":"auto|off|1|4|8","streams":1,"wait_ms":20}
//                  - mode: fall HAILO_BATCH_MODE; auto = chooseBatch()
//                  - streams: the stream count the deployment runs (decides
//                    the auto policy's batch size; default 1 per §11 D1)
//   rtsp           (optional) {"latency_ms":100,"drop_on_latency":false,
//                             "codec":"h264|h265"} for hailo_source
//   max_contexts   (optional; default 1, this HEF is single network group)
//
// Caps: {max_contexts:1, max_batch:<batch policy>, keypoints:<decoder>,
// exclusive_device:true} — one VDevice owned by the backend, shared by all
// contexts.
std::unique_ptr<Backend> make_hailo_backend(const std::string& backend_json,
                                            std::string& err);

}  // namespace vb
