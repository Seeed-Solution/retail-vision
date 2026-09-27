// Classification decoder (spec BASE-1 §6.11, M1.15): one [nc] (or [1,nc])
// logit vector -> top_k whole-frame detections (cx=cy=0.5, w=h=1) sorted by
// descending score. NMS is intentionally skipped: identical full-frame boxes
// would collapse to one. Python config validation additionally requires
// tracker.enabled=false for this decoder (M1.15b).
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "post/tensor_check.h"
#include "vb/decoder.h"

namespace vb {
namespace {

class ClassifyDecoder : public Decoder {
public:
    explicit ClassifyDecoder(DecodeSpec s) : spec_(std::move(s)) {}

    int keypoints() const override { return 0; }

    bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                float score, float nms_th, DetectionResult& out,
                std::string& err) override {
        (void)nms_th;  // full-frame boxes: NMS would collapse them
        if (outs == nullptr || n != 1) {
            err = "classify decoder expects one [nc] output vector";
            return false;
        }
        if (!post_detail::check_view(outs[0], "classify decoder", err)) return false;
        if (outs[0].dims.size() != 1) {
            err = "classify decoder expects one [nc] output vector";
            return false;
        }
        const int nc = spec_.num_classes;
        if (outs[0].dims[0] != nc) {
            err = "classify decoder: output size does not match num_classes";
            return false;
        }
        (void)model_w;
        (void)model_h;
        std::vector<float> p(outs[0].data, outs[0].data + nc);
        if (spec_.softmax) {
            float mx = *std::max_element(p.begin(), p.end());
            float sum = 0;
            for (float& v : p) {
                v = std::exp(v - mx);
                sum += v;
            }
            for (float& v : p) v /= sum;
        }
        // Non-finite scores are dropped (they cannot be compared against the
        // threshold) and demoted to -inf so the sort keeps a strict weak
        // ordering. Their count is reported through err.
        size_t dropped = 0;
        for (float& v : p) {
            if (!std::isfinite(v)) {
                v = -std::numeric_limits<float>::infinity();
                ++dropped;
            }
        }
        std::vector<int> order(nc);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return p[a] > p[b]; });
        out.dets.clear();
        out.kpts.clear();
        for (int k = 0; k < spec_.top_k && k < nc; ++k) {
            int cid = order[k];
            if (!std::isfinite(p[cid]) || p[cid] <= score) break;
            Detection d;
            d.cx = 0.5f;
            d.cy = 0.5f;
            d.w = 1.0f;
            d.h = 1.0f;
            d.score = p[cid];
            d.class_id = cid;
            d.track_id = 0;
            out.dets.push_back(d);
        }
        post_detail::note_dropped(err, "classify decoder", dropped);
        return true;
    }

private:
    DecodeSpec spec_;
};

}  // namespace

std::unique_ptr<Decoder> make_classify_decoder(DecodeSpec s) {
    return std::make_unique<ClassifyDecoder>(std::move(s));
}

}  // namespace vb
