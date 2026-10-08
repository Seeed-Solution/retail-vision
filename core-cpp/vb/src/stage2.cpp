#include "stage2.h"

#include <algorithm>
#include <cmath>

namespace vb {

static bool in_polygon(float x, float y, const std::vector<float>& p) {
    if (p.size() < 6 || (p.size() & 1)) return false;
    bool inside = false;
    for (size_t i = 0, j = p.size() - 2; i < p.size(); j = i, i += 2) {
        float xi = p[i], yi = p[i + 1], xj = p[j], yj = p[j + 1];
        bool cross = ((yi > y) != (yj > y));
        if (cross && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
}

std::vector<CropReq> stage2_select_crops(const std::vector<Detection>& dets,
                                         double t, const Stage2FilterConfig& c,
                                         std::map<uint32_t, Stage2TrackState>& st) {
    struct Candidate { size_t i; float score; };
    std::vector<Candidate> v;
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        if (!std::isfinite(d.score) || !std::isfinite(d.cx) || !std::isfinite(d.cy) ||
            !std::isfinite(d.w) || !std::isfinite(d.h) || d.score < c.min_score ||
            d.w <= 0 || d.h <= 0) continue;
        if (!c.classes.empty() && std::find(c.classes.begin(), c.classes.end(), d.class_id) == c.classes.end()) continue;
        v.push_back({i, d.score});
    }
    std::stable_sort(v.begin(), v.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    std::vector<CropReq> out;
    for (const Candidate& x : v) {
        const Detection& d = dets[x.i];
        float x0 = d.cx - d.w * (0.5f + c.expand), y0 = d.cy - d.h * (0.5f + c.expand);
        float x1 = d.cx + d.w * (0.5f + c.expand), y1 = d.cy + d.h * (0.5f + c.expand);
        x0 = std::max(c.roi[0], std::max(0.0f, x0)); y0 = std::max(c.roi[1], std::max(0.0f, y0));
        x1 = std::min(c.roi[2], std::min(1.0f, x1)); y1 = std::min(c.roi[3], std::min(1.0f, y1));
        if (!(x0 < x1 && y0 < y1)) continue;
        float bx = std::clamp(d.cx, 0.0f, 1.0f), by = std::clamp(d.cy + d.h * 0.5f, 0.0f, 1.0f);
        if (!c.roi_polygon.empty() && !in_polygon(bx, by, c.roi_polygon)) continue;
        Stage2TrackState& q = st[d.track_id];
        if (c.max_per_track > 0 && q.attempts >= static_cast<uint64_t>(c.max_per_track)) continue;
        if (c.min_gap_s > 0 && q.last_attempt_s > -1e20 && t - q.last_attempt_s < c.min_gap_s) continue;
        if (c.max_crops_per_frame > 0 && out.size() >= static_cast<size_t>(c.max_crops_per_frame)) break;
        CropReq r; r.x0 = x0; r.y0 = y0; r.x1 = x1; r.y1 = y1; r.track_id = d.track_id;
        out.push_back(r); ++q.attempts; q.last_attempt_s = t;
    }
    return out;
}

} // namespace vb
