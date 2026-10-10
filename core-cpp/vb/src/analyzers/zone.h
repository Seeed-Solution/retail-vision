// Internal factory for the built-in zone analyzer (spec BASE-1 §6.2, M1.6).
#pragma once

#include <memory>

#include "vb/analyzer.h"

namespace vb {
std::unique_ptr<Analyzer> make_zone_analyzer();
}  // namespace vb
