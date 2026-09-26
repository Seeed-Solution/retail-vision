// YOLOv8 and YOLOv8-DFL decoders (spec BASE-1 §6.11, M1.15).
//
// yolov8: one output with already-decoded cx,cy,w,h plus nc class scores,
// laid out either [4+nc, N] (standard transposed export) or [N, 4+nc]
// (detected by dimension). Box values are model-canvas pixels; class scores
// are already sigmoid probabilities.
//
// yolov8_dfl: per stride either one [4*reg_max+nc, H, W] tensor or a split
// pair box [4*reg_max,H,W] + cls [nc,H,W] (RKNN/Hailo style export). Each
// direction's reg_max bins get softmax'd and expectation-reduced to a
// distance in strides; x0 = (gx+0.5-l)*stride etc.; cls is sigmoid.
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "vb/decoder.h"
#include "vb/post.h"

namespace vb {
namespace {

void best_class(const float* cls_scores, int nc, float& best, int& best_idx) {
    best = cls_scores[0];
    best_idx = 0;
    for (int c = 1; c < nc; ++c)
        if (cls_scores[c] > best) {
            best = cls_scores[c];
            best_idx = c;
        }
}

class YoloV8Decoder : public Decoder {
public:
    explicit YoloV8Decoder(DecodeSpec s) : spec_(std::move(s)) {}

    int keypoints() const override { return 0; }

    bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                float score, float nms_th, DetectionResult& out,
                std::string& err) override {
        if (n != 1 || outs[0].dims.size() != 2) {
            err = "yolov8 decoder expects one 2-D output tensor";
            return false;
        }
        const int nc = spec_.num_classes;
        const int64_t d0 = outs[0].dims[0], d1 = outs[0].dims[1];
        const float* data = outs[0].data;
        int64_t count = 0;
        bool transposed = false;
        if (d0 == 4 + nc) {
            count = d1;
            transposed = true;
        } else if (d1 == 4 + nc) {
            count = d0;
        } else {
            err = "yolov8 decoder: output shape does not match num_classes";
            return false;
        }
        out.dets.clear();
        out.kpts.clear();
        for (int64_t i = 0; i < count; ++i) {
            float cx, cy, w, h;
            std::vector<float> cls(nc);
            if (transposed) {  // [4+nc, N]
                cx = data[0 * count + i];
                cy = data[1 * count + i];
                w = data[2 * count + i];
                h = data[3 * count + i];
                for (int c = 0; c < nc; ++c) cls[c] = data[(4 + c) * count + i];
            } else {  // [N, 4+nc]
                const float* row = data + i * (4 + nc);
                cx = row[0];
                cy = row[1];
                w = row[2];
                h = row[3];
                for (int c = 0; c < nc; ++c) cls[c] = row[4 + c];
            }
            float s;
            int cid;
            best_class(cls.data(), nc, s, cid);
            if (s <= score) continue;
            Detection d;
            d.cx = cx / model_w;
            d.cy = cy / model_h;
            d.w = w / model_w;
            d.h = h / model_h;
            d.score = s;
            d.class_id = cid;
            d.track_id = 0;
            out.dets.push_back(d);
        }
        nms(out.dets, out.kpts, nms_th, true);
        return true;
    }

private:
    DecodeSpec spec_;
};

class YoloV8DflDecoder : public Decoder {
public:
    explicit YoloV8DflDecoder(DecodeSpec s) : spec_(std::move(s)) {}

    int keypoints() const override { return 0; }

    bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                float score, float nms_th, DetectionResult& out,
                std::string& err) override {
        const int nc = spec_.num_classes, rm = spec_.reg_max;
        // stride -> {box, cls, H, W}; combined tensors fill both slots.
        struct Level {
            const float* box = nullptr;
            const float* cls = nullptr;
            int64_t H = 0, W = 0;
        };
        std::map<int, Level> levels;
        for (size_t i = 0; i < n; ++i) {
            const TensorView& t = outs[i];
            if (t.dims.size() != 3) {
                err = "yolov8_dfl decoder expects [C, H, W] tensors";
                return false;
            }
            const int64_t C = t.dims[0], H = t.dims[1], W = t.dims[2];
            int stride = 0;
            for (int s : spec_.strides) {
                if (model_h % s == 0 && model_h / s == H && model_w % s == 0 &&
                    model_w / s == W) {
                    stride = s;
                    break;
                }
            }
            if (stride == 0) {
                err = "yolov8_dfl decoder: output HxW does not match a stride";
                return false;
            }
            Level& lv = levels[stride];
            if (C == 4 * rm + nc) {
                lv.box = t.data;
                lv.cls = t.data + 4 * static_cast<size_t>(rm) * H * W;
                lv.H = H;
                lv.W = W;
            } else if (C == 4 * rm) {
                lv.box = t.data;
                lv.H = H;
                lv.W = W;
            } else if (C == nc) {
                lv.cls = t.data;
                lv.H = H;
                lv.W = W;
            } else {
                err = "yolov8_dfl decoder: channel count does not match "
                      "reg_max/num_classes";
                return false;
            }
        }
        out.dets.clear();
        out.kpts.clear();
        for (const auto& [stride, lv] : levels) {
            if (!lv.box || !lv.cls || lv.H == 0) {
                err = "yolov8_dfl decoder: incomplete box/cls tensors for a stride";
                return false;
            }
            const int64_t H = lv.H, W = lv.W;
            const size_t plane = static_cast<size_t>(H) * W;
            for (int64_t gy = 0; gy < H; ++gy) {
                for (int64_t gx = 0; gx < W; ++gx) {
                    float dist[4];
                    for (int d = 0; d < 4; ++d) {
                        // softmax over reg_max bins, expectation -> distance
                        float mx = -1e30f;
                        const float* bins =
                            lv.box + (static_cast<size_t>(d) * rm) * plane +
                            gy * W + gx;
                        for (int k = 0; k < rm; ++k) mx = std::max(mx, bins[k * plane]);
                        float sum = 0, expct = 0;
                        for (int k = 0; k < rm; ++k) {
                            float e = std::exp(bins[k * plane] - mx);
                            sum += e;
                            expct += e * k;
                        }
                        dist[d] = expct / sum;
                    }
                    float x0 = (gx + 0.5f - dist[0]) * stride;
                    float x1 = (gx + 0.5f + dist[2]) * stride;
                    float y0 = (gy + 0.5f - dist[1]) * stride;
                    float y1 = (gy + 0.5f + dist[3]) * stride;
                    std::vector<float> cls(nc);
                    for (int c = 0; c < nc; ++c)
                        cls[c] = 1.0f / (1.0f + std::exp(-lv.cls[static_cast<size_t>(c) * plane + gy * W + gx]));
                    float s;
                    int cid;
                    best_class(cls.data(), nc, s, cid);
                    if (s <= score) continue;
                    Detection det;
                    det.cx = (x0 + x1) * 0.5f / model_w;
                    det.cy = (y0 + y1) * 0.5f / model_h;
                    det.w = (x1 - x0) / model_w;
                    det.h = (y1 - y0) / model_h;
                    det.score = s;
                    det.class_id = cid;
                    det.track_id = 0;
                    out.dets.push_back(det);
                }
            }
        }
        nms(out.dets, out.kpts, nms_th, true);
        return true;
    }

private:
    DecodeSpec spec_;
};

}  // namespace

std::unique_ptr<Decoder> make_yolov8_decoder(DecodeSpec s) {
    return std::make_unique<YoloV8Decoder>(std::move(s));
}

std::unique_ptr<Decoder> make_yolov8_dfl_decoder(DecodeSpec s) {
    return std::make_unique<YoloV8DflDecoder>(std::move(s));
}

}  // namespace vb
