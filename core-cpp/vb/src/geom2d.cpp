// Shared 2D geometry helpers (spec BASE-1 §6.2.5, M1.16).
#include <cstddef>
#include "vb/geom2d.h"

namespace vb {

bool point_in_polygon(float x, float y, const Polygon& poly) {
    bool inside = false;
    size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        float xi = poly[i][0], yi = poly[i][1];
        float xj = poly[j][0], yj = poly[j][1];
        if ((yi > y) != (yj > y) &&
            x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside = !inside;
    }
    return inside;
}

}  // namespace vb
