// Internal factory for the built-in pose_angle analyzer
// (spec BASE-1 §6.2.5, M1.17).
#pragma once

#include <memory>

#include "vb/analyzer.h"

namespace vb {
std::unique_ptr<Analyzer> make_pose_angle_analyzer();
}  // namespace vb
