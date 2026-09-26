// Scalar letterbox coordinate mapping (spec BASE-1 §6.1, mirrors core-py/vision_base/letterbox.py).
#pragma once

#include "vb/types.h"

namespace vb {

// Fit (src_w, src_h) into the (model_w, model_h) canvas; throws std::invalid_argument
// on non-positive sizes.
LetterboxGeom letterbox_fit(int src_w, int src_h, int model_w, int model_h, Align a);

}  // namespace vb
