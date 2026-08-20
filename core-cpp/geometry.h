#ifndef _RETAIL_GEOMETRY_H_
#define _RETAIL_GEOMETRY_H_

// Pure 2D geometry helpers for zone/line configuration.
// No project dependencies so the logic is unit-testable on the host.
// All coordinates are normalized [0,1] (resolution independent).

#include <array>
#include <cstddef>
#include <vector>

namespace retail_vision {
namespace geom {

using Point = std::array<float, 2>; // {x, y}

// Ray-casting point-in-polygon test (crossing number, handles non-convex
// polygons). Points exactly on an edge may fall on either side; that is
// acceptable for occupancy counting.
inline bool point_in_polygon(float px, float py, const std::vector<Point>& poly) {
    if (poly.size() < 3) return false;
    bool inside = false;
    size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        float xi = poly[i][0], yi = poly[i][1];
        float xj = poly[j][0], yj = poly[j][1];
        // Does the horizontal ray from (px,py) cross edge (j -> i)?
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

// Signed side of point p relative to the directed line a -> b.
// > 0 : p is on the LEFT of a->b, < 0 : RIGHT, == 0 : collinear.
inline float line_side(float ax, float ay, float bx, float by, float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// Did the movement segment p0 -> p1 cross the finite line segment a -> b?
// Returns:
//    0 : no crossing
//   +1 : crossed from the LEFT side of a->b to the RIGHT side
//   -1 : crossed from the RIGHT side of a->b to the LEFT side
// Both "strictly opposite sides" tests are required so that only genuine
// segment-segment intersections count (touching an endpoint or moving
// parallel past the line does not).
inline int segment_crossing(float ax, float ay, float bx, float by,
                            float p0x, float p0y, float p1x, float p1y) {
    float side0 = line_side(ax, ay, bx, by, p0x, p0y);
    float side1 = line_side(ax, ay, bx, by, p1x, p1y);
    if (side0 == 0.0f || side1 == 0.0f || (side0 > 0) == (side1 > 0)) {
        return 0; // both ends on the same side (or touching) -> no crossing
    }
    // The movement segment straddles the infinite line; now require that the
    // line segment a->b also straddles the movement segment p0->p1.
    float sa = line_side(p0x, p0y, p1x, p1y, ax, ay);
    float sb = line_side(p0x, p0y, p1x, p1y, bx, by);
    if (sa == 0.0f || sb == 0.0f || (sa > 0) == (sb > 0)) {
        return 0;
    }
    return (side0 > 0) ? +1 : -1; // left -> right : +1
}

} // namespace geom
} // namespace retail_vision

#endif // _RETAIL_GEOMETRY_H_
