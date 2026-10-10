// Shared 2D geometry helpers (spec BASE-1 §6.2.5, M1.16; extended in M4.1).
#pragma once

#include <array>
#include <vector>

namespace vb {

using Polygon = std::vector<std::array<float, 2>>;

// Ray-casting point-in-polygon; boundary behaviour follows the zone analyzer
// (§6.2). Points on an edge are treated consistently by the even-odd rule.
bool point_in_polygon(float x, float y, const Polygon& poly);

}  // namespace vb
