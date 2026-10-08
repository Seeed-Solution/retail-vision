// Shared 2D geometry helpers (spec BASE-1 §6.2.5, M1.16).
#include <cstddef>
#include <cmath>
#include <algorithm>
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

float polygon_signed_area(const Polygon& poly) {
    if (poly.size() < 3) return 0.0f;
    double s = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];
        s += static_cast<double>(a[0]) * b[1] - static_cast<double>(b[0]) * a[1];
    }
    return static_cast<float>(s * 0.5);
}

float polygon_area(const Polygon& poly) { return std::fabs(polygon_signed_area(poly)); }

bool valid_convex_polygon(const Polygon& poly, std::string* err) {
    constexpr double kCrossEps = 1e-12;
    auto fail = [&](const char* why) {
        if (err) *err = why;
        return false;
    };
    if (poly.size() < 3 || poly.size() > 16) return fail("polygon must have 3..16 vertices");
    for (const auto& p : poly)
        if (!std::isfinite(p[0]) || !std::isfinite(p[1])) return fail("polygon coordinates must be finite");
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];
        const double dx = b[0] - a[0], dy = b[1] - a[1];
        if (dx * dx + dy * dy <= kCrossEps * kCrossEps) return fail("polygon has duplicate vertices");
    }
    if (polygon_area(poly) < 1e-6) return fail("polygon area is too small");
    int sign = 0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % poly.size()];
        const auto& c = poly[(i + 2) % poly.size()];
        const double cross = (b[0] - a[0]) * (c[1] - b[1]) -
                             (b[1] - a[1]) * (c[0] - b[0]);
        if (std::fabs(cross) <= kCrossEps) continue;
        const int cs = cross > 0 ? 1 : -1;
        if (!sign) sign = cs;
        else if (sign != cs) return fail("polygon must be convex and simple");
    }
    // A constant-turn polygon with all turns in one direction has exterior
    // angle sum 2*pi; this also documents the intended rejection of star paths.
    constexpr double kPi = 3.14159265358979323846;
    double turn_sum = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& a = poly[(i + poly.size() - 1) % poly.size()];
        const auto& b = poly[i];
        const auto& c = poly[(i + 1) % poly.size()];
        const double u = std::atan2(b[1] - a[1], b[0] - a[0]);
        const double v = std::atan2(c[1] - b[1], c[0] - b[0]);
        double d = v - u;
        while (d <= -kPi) d += 2 * kPi;
        while (d > kPi) d -= 2 * kPi;
        turn_sum += d;
    }
    if (std::fabs(std::fabs(turn_sum) - 2 * kPi) > 1e-6) return fail("polygon exterior angle sum is invalid");
    return true;
}

namespace {
template <typename Inside, typename Intersect>
Polygon clip_edge(const Polygon& in, Inside inside, Intersect intersect) {
    Polygon out;
    if (in.empty()) return out;
    auto prev = in.back();
    bool prev_in = inside(prev);
    for (const auto& cur : in) {
        const bool cur_in = inside(cur);
        if (cur_in != prev_in) out.push_back(intersect(prev, cur));
        if (cur_in) out.push_back(cur);
        prev = cur;
        prev_in = cur_in;
    }
    return out;
}
}

Polygon clip_polygon_rect(const Polygon& poly, float x0, float y0, float x1, float y1) {
    if (x1 <= x0 || y1 <= y0) return {};
    Polygon p = poly;
    p = clip_edge(p, [x0](auto q) { return q[0] >= x0; }, [x0](auto a, auto b) {
        float t = (x0 - a[0]) / (b[0] - a[0]); return std::array<float,2>{x0, a[1] + t * (b[1] - a[1])}; });
    p = clip_edge(p, [x1](auto q) { return q[0] <= x1; }, [x1](auto a, auto b) {
        float t = (x1 - a[0]) / (b[0] - a[0]); return std::array<float,2>{x1, a[1] + t * (b[1] - a[1])}; });
    p = clip_edge(p, [y0](auto q) { return q[1] >= y0; }, [y0](auto a, auto b) {
        float t = (y0 - a[1]) / (b[1] - a[1]); return std::array<float,2>{a[0] + t * (b[0] - a[0]), y0}; });
    p = clip_edge(p, [y1](auto q) { return q[1] <= y1; }, [y1](auto a, auto b) {
        float t = (y1 - a[1]) / (b[1] - a[1]); return std::array<float,2>{a[0] + t * (b[0] - a[0]), y1}; });
    return p;
}

float polygon_intersection_area_rect(const Polygon& poly, float x0, float y0, float x1, float y1) {
    return polygon_area(clip_polygon_rect(poly, x0, y0, x1, y1));
}

}  // namespace vb
