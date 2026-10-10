// Shared post-processing (spec BASE-1 §6.1, M1.9). Platform-agnostic;
// decoders emit model-canvas normalized Detections.
#pragma once

#include <vector>

#include "vb/types.h"

namespace vb {

// Greedy NMS by descending score; class_aware = per-class suppression.
// Detection kpt slices (kpt_offset/kpt_count) are compacted along with the
// kept detections.
void nms(std::vector<Detection>& dets, std::vector<Keypoint>& kpts, float iou,
         bool class_aware);

// YOLOX head decode. `out` is the [1, n_anchors, 5 + n_cls] output tensor
// (row-major, batch 1): rows are x1, y1, x2, y2, obj, cls[0..n_cls) in model
// canvas pixels; obj/cls are already sigmoid scores. Detections are emitted
// in model-canvas normalized units, one per anchor (best class only), with
// score = obj * max(cls) > `score`.
void yolox_decode(const float* out, int n_anchors, int n_cls, int model_w,
                  int model_h, float score, std::vector<Detection>& dets);

}  // namespace vb
