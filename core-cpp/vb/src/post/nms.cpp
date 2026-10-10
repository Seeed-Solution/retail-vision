// Greedy NMS (spec BASE-1 §6.1, M1.9).
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "vb/post.h"

namespace vb {
namespace {

float iou(const Detection& a, const Detection& b) {
    float ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
    float ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
    float bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
    float bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

}  // namespace

void nms(std::vector<Detection>& dets, std::vector<Keypoint>& kpts, float iou_th,
         bool class_aware) {
    std::vector<size_t> order(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return dets[a].score > dets[b].score; });

    std::vector<char> suppressed(dets.size(), 0);
    std::vector<Detection> kept_dets;
    std::vector<Keypoint> kept_kpts;
    kept_dets.reserve(dets.size());
    for (size_t oi : order) {
        if (suppressed[oi]) continue;
        const Detection& keep = dets[oi];
        for (size_t oj : order) {
            if (oj == oi || suppressed[oj]) continue;
            const Detection& other = dets[oj];
            if (class_aware && keep.class_id != other.class_id) continue;
            if (iou(keep, other) > iou_th) suppressed[oj] = 1;
        }
        Detection d = keep;
        uint32_t cnt = d.kpt_count;
        if (cnt > 0 && keep.kpt_offset + cnt <= kpts.size()) {
            // kpt_offset in the input vector points at the original slice.
            d.kpt_offset = static_cast<uint32_t>(kept_kpts.size());
            kept_kpts.insert(kept_kpts.end(), kpts.begin() + keep.kpt_offset,
                             kpts.begin() + keep.kpt_offset + cnt);
        } else {
            d.kpt_count = 0;
            d.kpt_offset = 0;
        }
        kept_dets.push_back(d);
    }
    dets.swap(kept_dets);
    kpts.swap(kept_kpts);
}

}  // namespace vb
