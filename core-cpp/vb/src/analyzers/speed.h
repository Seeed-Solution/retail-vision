// Internal factory for the built-in speed analyzer (spec BASE-1 §6.2.5, M1.16).
#pragma once

#include <memory>

#include "vb/analyzer.h"

namespace vb {
std::unique_ptr<Analyzer> make_speed_analyzer();
}  // namespace vb
