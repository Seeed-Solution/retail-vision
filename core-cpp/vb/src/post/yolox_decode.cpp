// YOLOX head decode (spec BASE-1 §6.1/§6.11, M1.9).
#include "vb/post.h"

namespace vb {

void yolox_decode(const float* out, int n_anchors, int n_cls, int model_w, int model_h,
                  float score, std::vector<Detection>& dets) {
    const size_t row = static_cast<size_t>(n_cls) + 5;
    for (int i = 0; i < n_anchors; ++i) {
        const float* p = out + i * row;
        float obj = p[4];
        int best = 0;
        float best_cls = p[5];
        for (int c = 1; c < n_cls; ++c) {
            if (p[5 + c] > best_cls) {
                best_cls = p[5 + c];
                best = c;
            }
        }
        float sc = obj * best_cls;
        if (sc <= score) continue;
        Detection d;
        d.cx = (p[0] + p[2]) / 2.0f / static_cast<float>(model_w);
        d.cy = (p[1] + p[3]) / 2.0f / static_cast<float>(model_h);
        d.w = (p[2] - p[0]) / static_cast<float>(model_w);
        d.h = (p[3] - p[1]) / static_cast<float>(model_h);
        d.score = sc;
        d.class_id = best;
        dets.push_back(d);
    }
}

}  // namespace vb
