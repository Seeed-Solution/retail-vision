// Internal factory for the built-in count_threshold analyzer
// (spec BASE-1 §6.2.5, M1.17).
#pragma once

#include <memory>

#include "vb/analyzer.h"

namespace vb {
std::unique_ptr<Analyzer> make_count_threshold_analyzer();
}  // namespace vb
