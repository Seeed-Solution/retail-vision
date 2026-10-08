// Shared 2D geometry helpers (spec BASE-1 §6.2.5, M1.16; extended in M4.1).
#pragma once

#include <array>
#include <string>
#include <vector>

namespace vb {

using Polygon = std::vector<std::array<float, 2>>;

// Ray-casting point-in-polygon; boundary behaviour follows the zone analyzer
// (§6.2). Points on an edge are treated consistently by the even-odd rule.
bool point_in_polygon(float x, float y, const Polygon& poly);

// Geometry primitives used by convex ROI analyzers.  Coordinates are in the
// same normalized coordinate space as Polygon.
float polygon_signed_area(const Polygon& poly);
float polygon_area(const Polygon& poly);
bool valid_convex_polygon(const Polygon& poly, std::string* err = nullptr);
Polygon clip_polygon_rect(const Polygon& poly, float x0, float y0, float x1, float y1);
float polygon_intersection_area_rect(const Polygon& poly, float x0, float y0,
                                     float x1, float y1);

}  // namespace vb
