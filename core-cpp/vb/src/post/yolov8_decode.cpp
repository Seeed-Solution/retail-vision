// YOLOv8, YOLOv8-DFL and YOLO-pose decoders (spec BASE-1 §6.11, M1.15).
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
//
// yolo_pose: the layouts of the pose decoders that §6.11 consolidates here.
//   A (split DFL pose head, one tensor set per stride; skeleton per the M3
//     migration table "以 RK rknn_postprocess.cpp 的布局判断（:19-29，兼容
//     NCHW/NHWC 与 51/64 通道）为骨架"):
//       box   [4*reg_max, H, W]   DFL logits, reg_max bins per side
//       cls   [nc, H, W]          objectness/class plane
//       kpt   [3*keypoints, H, W] x, y in grid units + confidence
//     Keypoints follow the Ultralytics raw-head layout
//     x = (2*px + gx) * stride (Ultralytics; fall-detection
//     platforms/rknn/cpp/rknn_postprocess.cpp:124-128, and
//     platforms/rpi-hailo/src/hailo_pose_decoder.cpp:38-42 for the same head).
//   B (single end-to-end export [5+3K, N] or [N, 5+3K]; fall-detection
//     platforms/jetson/main/yolo_pose.cpp:38-73): cx, cy, w, h in model-input
//     pixels (or normalized when |v| <= 2), score, then K × (x, y, conf).
//     The boxes here are already DFL-decoded and multiplied by the strides,
//     and the class plane is already sigmoid'd (the Ultralytics export fuses
//     dfl -> dist2bbox -> *strides and the sigmoid into the graph), so this
//     path needs no grid or DFL step: it normalizes by the canvas and is done.
//     Score/conf values are classified as probability-or-logit with kProbTol
//     of slack, because the fused sigmoid can land a few ULP outside [0,1].
//
// Non-finite inputs are dropped rather than rescaled: a NaN score escapes the
// `score <= threshold` test and reaches the output layer as JSON null, so the
// affected candidate is discarded and the count reported through err.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "post/tensor_check.h"
#include "vb/decoder.h"
#include "vb/post.h"

namespace vb {
namespace {

using post_detail::check_canvas;
using post_detail::check_view;
using post_detail::note_dropped;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void best_class(const float* cls_scores, int nc, float& best, int& best_idx) {
    best = cls_scores[0];
    best_idx = 0;
    for (int c = 1; c < nc; ++c)
        if (cls_scores[c] > best) {
            best = cls_scores[c];
            best_idx = c;
        }
}

// DFL expectation over `rm` bins strided by `plane` floats. False when a bin
// or the resulting distance is not finite: NaN bins poison the softmax, and
// all-(-inf) / all-(+inf) bins make mx non-finite (exp(inf-inf) = NaN).
bool dfl_distance(const float* bins, int rm, size_t plane, float& dist) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < rm; ++k) {
        const float v = bins[static_cast<size_t>(k) * plane];
        if (!std::isfinite(v)) return false;
        mx = std::max(mx, v);
    }
    if (!std::isfinite(mx)) return false;
    float sum = 0.0f, expct = 0.0f;
    for (int k = 0; k < rm; ++k) {
        const float e = std::exp(bins[static_cast<size_t>(k) * plane] - mx);
        sum += e;
        expct += e * static_cast<float>(k);
    }
    if (!std::isfinite(sum) || sum <= 0.0f || !std::isfinite(expct)) return false;
    dist = expct / sum;
    return std::isfinite(dist);
}

// Stride whose grid matches this output's HxW, or 0 when none does.
int match_stride(const DecodeSpec& spec, int model_w, int model_h, int64_t H,
                 int64_t W) {
    for (int s : spec.strides) {
        if (model_h % s == 0 && model_h / s == H && model_w % s == 0 &&
            model_w / s == W)
            return s;
    }
    return 0;
}

// D3: with reg_max=16 and num_classes=64 a box tensor and a cls tensor have
// the same channel count, so the shape alone does not say which is which.
// Output names are matched first (+1 box, -1 cls, 0 no hint); names that
// match both roles or neither yield no hint and the caller falls back to the
// documented output order (box first).
int role_hint(const std::string& name) {
    std::string n;
    n.reserve(name.size());
    for (char c : name) n.push_back(static_cast<char>(std::tolower(c)));
    auto has = [&](const char* needle) { return n.find(needle) != std::string::npos; };
    const bool box = has("box") || has("bbox") || has("reg") || has("dfl");
    const bool cls = has("cls") || has("class") || has("score") || has("prob") ||
                     has("conf");
    if (box == cls) return 0;
    return box ? 1 : -1;
}

class YoloV8Decoder : public Decoder {
public:
    explicit YoloV8Decoder(DecodeSpec s) : spec_(std::move(s)) {}

    int keypoints() const override { return 0; }

    bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                float score, float nms_th, DetectionResult& out,
                std::string& err) override {
        if (outs == nullptr || n != 1) {
            err = "yolov8 decoder expects one 2-D output tensor";
            return false;
        }
        if (!check_canvas(model_w, model_h, "yolov8 decoder", err)) return false;
        if (!check_view(outs[0], "yolov8 decoder", err)) return false;
        if (outs[0].dims.size() != 2) {
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
        std::vector<float> cls(nc);
        out.dets.clear();
        out.kpts.clear();
        size_t dropped = 0;
        for (int64_t i = 0; i < count; ++i) {
            float cx, cy, w, h;
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
            // D4: the finiteness test runs before the threshold comparison.
            if (!std::isfinite(s) || !std::isfinite(cx) || !std::isfinite(cy) ||
                !std::isfinite(w) || !std::isfinite(h)) {
                ++dropped;
                continue;
            }
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
        note_dropped(err, "yolov8 decoder", dropped);
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
        if (outs == nullptr || n == 0) {
            err = "yolov8_dfl decoder expects [C, H, W] tensors";
            return false;
        }
        if (!check_canvas(model_w, model_h, "yolov8_dfl decoder", err)) return false;
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
            if (!check_view(t, "yolov8_dfl decoder", err)) return false;
            if (t.dims.size() != 3) {
                err = "yolov8_dfl decoder expects [C, H, W] tensors";
                return false;
            }
            const int64_t C = t.dims[0], H = t.dims[1], W = t.dims[2];
            const int stride = match_stride(spec_, model_w, model_h, H, W);
            if (stride == 0) {
                err = "yolov8_dfl decoder: output HxW does not match a stride";
                return false;
            }
            Level& lv = levels[stride];
            if (lv.H == 0) {
                lv.H = H;
                lv.W = W;
            } else if (lv.H != H || lv.W != W) {
                err = "yolov8_dfl decoder: one stride got two spatial sizes";
                return false;
            }
            // D3: roles are resolved explicitly instead of by "first match
            // wins", which silently let a cls tensor overwrite the box slot.
            const bool combined = (C == 4LL * rm + nc);
            const bool fits_box = (C == 4LL * rm);
            const bool fits_cls = (C == nc);
            if (!combined && !fits_box && !fits_cls) {
                err = "yolov8_dfl decoder: channel count does not match "
                      "reg_max/num_classes";
                return false;
            }
            bool want_box = combined, want_cls = combined;
            if (!combined) {
                if (fits_box && fits_cls) {  // 4*reg_max == num_classes
                    int hint = role_hint(t.name);
                    if (hint == 0) hint = lv.box == nullptr ? 1 : -1;
                    want_box = hint > 0;
                    want_cls = hint < 0;
                } else {
                    want_box = fits_box;
                    want_cls = fits_cls;
                }
            }
            if (want_box) {
                if (lv.box != nullptr) {
                    err = "yolov8_dfl decoder: two box tensors for one stride";
                    return false;
                }
                lv.box = t.data;
            }
            if (want_cls) {
                if (lv.cls != nullptr) {
                    err = "yolov8_dfl decoder: two cls tensors for one stride";
                    return false;
                }
                // A combined [4*reg_max+nc, H, W] tensor carries the class
                // plane after the box planes; a separate cls tensor starts at
                // its own channel 0.
                lv.cls = combined
                             ? t.data + 4 * static_cast<size_t>(rm) * H * W
                             : t.data;
            }
        }
        out.dets.clear();
        out.kpts.clear();
        size_t dropped = 0;
        for (const auto& [stride, lv] : levels) {
            if (!lv.box || !lv.cls || lv.H == 0) {
                err = "yolov8_dfl decoder: incomplete box/cls tensors for a stride";
                return false;
            }
            const int64_t H = lv.H, W = lv.W;
            const size_t plane = static_cast<size_t>(H) * W;
            std::vector<float> cls(nc);
            for (int64_t gy = 0; gy < H; ++gy) {
                for (int64_t gx = 0; gx < W; ++gx) {
                    const size_t cell = static_cast<size_t>(gy) * W +
                                        static_cast<size_t>(gx);
                    float dist[4];
                    bool finite = true;
                    for (int d = 0; d < 4 && finite; ++d) {
                        const float* bins =
                            lv.box + (static_cast<size_t>(d) * rm) * plane + cell;
                        finite = dfl_distance(bins, rm, plane, dist[d]);
                    }
                    float s = 0.0f;
                    int cid = 0;
                    if (finite) {
                        for (int c = 0; c < nc; ++c) {
                            const float logit =
                                lv.cls[static_cast<size_t>(c) * plane + cell];
                            cls[c] = sigmoid(logit);
                            if (!std::isfinite(cls[c])) {
                                finite = false;
                                break;
                            }
                        }
                    }
                    if (finite) {
                        best_class(cls.data(), nc, s, cid);
                        finite = std::isfinite(s);
                    }
                    // D4: drop the candidate, never emit non-finite numbers.
                    if (!finite) {
                        ++dropped;
                        continue;
                    }
                    if (s <= score) continue;
                    const float x0 = (gx + 0.5f - dist[0]) * stride;
                    const float x1 = (gx + 0.5f + dist[2]) * stride;
                    const float y0 = (gy + 0.5f - dist[1]) * stride;
                    const float y1 = (gy + 0.5f + dist[3]) * stride;
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
        note_dropped(err, "yolov8_dfl decoder", dropped);
        nms(out.dets, out.kpts, nms_th, true);
        return true;
    }

private:
    DecodeSpec spec_;
};

// ---- yolo_pose -----------------------------------------------------------

// How far outside [0,1] a value may sit and still be a probability: a graph
// that already applied the sigmoid can emit a value a few ULP below zero.
// Measured (M2.3): on the same model and the same input tensor, the
// Ultralytics pose export's class lane holds -2^-24 (-5.96e-08) at 35 of 8400
// anchors under the device's aarch64 ONNX Runtime 1.21, while macOS arm64
// 1.29 rounds the same computation to +0 (no other lane disagrees in sign).
// Testing `v < 0` per value and running sigmoid() on that value returns
// exactly 0.5, which clears any threshold <= 0.5 and turns every such anchor
// into a detection: 97.6% of the float reference's boxes carried score 0.5,
// so no parity comparison against it could be evaluated. 1e-6 is six orders
// of magnitude below any real logit's distance from the interval.
constexpr float kProbTol = 1e-6f;

// Split-head exports either fuse the sigmoid into the class plane or leave
// raw logits; the RK decoder separates them by value range
// (platforms/rknn/cpp/rknn_postprocess.cpp:95-105). Non-finite entries do not
// take part in the range test and are dropped per candidate later.
bool plane_is_logits(const float* p, size_t n) {
    float lo = p[0], hi = p[0];
    for (size_t i = 1; i < n; ++i) {
        lo = std::min(lo, p[i]);
        hi = std::max(hi, p[i]);
    }
    return lo < -kProbTol || hi > 1.0f + kProbTol;
}

// Layout B box/keypoint coordinates: values in [-2,2] are treated as
// normalized, anything else as model-input pixels
// (platforms/jetson/main/yolo_pose.cpp:27-30).
float coord_to_input(float v, float extent) {
    return std::fabs(v) <= 2.0f ? v * extent : v;
}

// A value already classified as a probability, with a rounded tail pinned
// back into [0,1] instead of being re-sigmoided into 0.5.
float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// Layout B probabilities are emitted either as probabilities or as logits
// (platforms/jetson/main/yolo_pose.cpp:19-23); the classification tolerates a
// few ULP of rounding outside [0,1] (kProbTol).
float probability(float v) {
    return (v >= -kProbTol && v <= 1.0f + kProbTol) ? clamp01(v) : sigmoid(v);
}

class YoloPoseDecoder : public Decoder {
public:
    explicit YoloPoseDecoder(DecodeSpec s) : spec_(std::move(s)) {}

    int keypoints() const override { return spec_.keypoints; }

    bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                float score, float nms_th, DetectionResult& out,
                std::string& err) override {
        if (outs == nullptr || n == 0) {
            err = "yolo_pose decoder expects output tensors";
            return false;
        }
        if (!check_canvas(model_w, model_h, "yolo_pose decoder", err)) return false;
        const int K = spec_.keypoints;
        if (K < 1) {
            err = "yolo_pose decoder requires keypoints >= 1";
            return false;
        }
        // Layout B: one end-to-end tensor with 4 box + 1 score + 3K features.
        if (n == 1 && outs[0].dims.size() == 2)
            return decode_end2end(outs[0], model_w, model_h, score, nms_th, out, err);
        return decode_split(outs, n, model_w, model_h, score, nms_th, out, err);
    }

private:
    // [5+3K, N] or [N, 5+3K].
    bool decode_end2end(const TensorView& t, int model_w, int model_h, float score,
                        float nms_th, DetectionResult& out, std::string& err) {
        if (!check_view(t, "yolo_pose decoder", err)) return false;
        const int K = spec_.keypoints;
        const int64_t feats = 5 + 3LL * K;
        const int64_t d0 = t.dims[0], d1 = t.dims[1];
        int64_t anchors = 0;
        bool feature_major = false;
        if (d0 == feats) {
            anchors = d1;
            feature_major = true;
        } else if (d1 == feats) {
            anchors = d0;
        } else {
            err = "yolo_pose decoder: single 2-D output does not match "
                  "5 + 3*keypoints";
            return false;
        }
        const float* data = t.data;
        auto at = [&](int64_t anchor, int64_t feature) {
            return feature_major ? data[feature * anchors + anchor]
                                 : data[anchor * feats + feature];
        };
        const float mw = static_cast<float>(model_w), mh = static_cast<float>(model_h);
        out.dets.clear();
        out.kpts.clear();
        size_t dropped = 0;
        for (int64_t a = 0; a < anchors; ++a) {
            const float s = probability(at(a, 4));
            if (!std::isfinite(s)) {
                ++dropped;
                continue;
            }
            if (s <= score) continue;
            const float cx = coord_to_input(at(a, 0), mw);
            const float cy = coord_to_input(at(a, 1), mh);
            const float w = std::fabs(coord_to_input(at(a, 2), mw));
            const float h = std::fabs(coord_to_input(at(a, 3), mh));
            if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
                !std::isfinite(h)) {
                ++dropped;
                continue;
            }
            if (!(w > 1e-3f && h > 1e-3f)) continue;  // degenerate box
            Detection det;
            det.cx = cx / mw;
            det.cy = cy / mh;
            det.w = w / mw;
            det.h = h / mh;
            det.score = s;
            det.class_id = 0;
            det.track_id = 0;
            det.kpt_offset = static_cast<uint32_t>(out.kpts.size());
            det.kpt_count = static_cast<uint32_t>(K);
            bool finite = true;
            for (int k = 0; k < K && finite; ++k) {
                const int64_t o = 5 + 3LL * k;
                const float kx = coord_to_input(at(a, o), mw);
                const float ky = coord_to_input(at(a, o + 1), mh);
                const float kc = probability(at(a, o + 2));
                if (!std::isfinite(kx) || !std::isfinite(ky) || !std::isfinite(kc))
                    finite = false;
                else
                    out.kpts.push_back(Keypoint{kx / mw, ky / mh, kc});
            }
            if (!finite) {  // drop the whole candidate, keypoints included
                out.kpts.resize(det.kpt_offset);
                ++dropped;
                continue;
            }
            out.dets.push_back(det);
        }
        note_dropped(err, "yolo_pose decoder", dropped);
        nms(out.dets, out.kpts, nms_th, true);
        return true;
    }

    // Per stride: box [4*reg_max, H, W], cls [nc, H, W], kpt [3K, H, W].
    bool decode_split(const TensorView* outs, size_t n, int model_w, int model_h,
                      float score, float nms_th, DetectionResult& out,
                      std::string& err) {
        const int K = spec_.keypoints, rm = spec_.reg_max;
        struct Level {
            const float* box = nullptr;
            const float* cls = nullptr;
            const float* kpt = nullptr;
            int64_t H = 0, W = 0;
            int nc = 0;
        };
        enum Role { ROLE_BOX, ROLE_CLS, ROLE_KPT };
        std::map<int, Level> levels;
        for (size_t i = 0; i < n; ++i) {
            const TensorView& t = outs[i];
            if (!check_view(t, "yolo_pose decoder", err)) return false;
            if (t.dims.size() != 3) {
                err = "yolo_pose decoder: split pose head expects [C, H, W] tensors";
                return false;
            }
            const int64_t C = t.dims[0], H = t.dims[1], W = t.dims[2];
            const int stride = match_stride(spec_, model_w, model_h, H, W);
            if (stride == 0) {
                err = "yolo_pose decoder: output HxW does not match a stride";
                return false;
            }
            Level& lv = levels[stride];
            if (lv.H == 0) {
                lv.H = H;
                lv.W = W;
            } else if (lv.H != H || lv.W != W) {
                err = "yolo_pose decoder: one stride got two spatial sizes";
                return false;
            }
            // Roles by channel count: 4*reg_max is the DFL box head, 3*K the
            // keypoint head, anything else the class plane. A count that fits
            // both heads (>0 only when 4*reg_max == 3*keypoints) is resolved
            // by name, then by the documented order (box before keypoints);
            // a second claim on an already-filled role fails loudly.
            const bool fits_box = (C == 4LL * rm);
            const bool fits_kpt = (C == 3LL * K);
            Role role = ROLE_CLS;
            if (fits_box && fits_kpt) {
                const int hint = role_hint(t.name);
                role = hint > 0   ? ROLE_BOX
                       : hint < 0 ? ROLE_KPT
                                  : (lv.box == nullptr ? ROLE_BOX : ROLE_KPT);
            } else if (fits_box) {
                role = ROLE_BOX;
            } else if (fits_kpt) {
                role = ROLE_KPT;
            }
            if (role == ROLE_BOX) {
                if (lv.box != nullptr) {
                    err = "yolo_pose decoder: two box tensors for one stride";
                    return false;
                }
                lv.box = t.data;
            } else if (role == ROLE_KPT) {
                if (lv.kpt != nullptr) {
                    err = "yolo_pose decoder: two keypoint tensors for one stride";
                    return false;
                }
                lv.kpt = t.data;
            } else {
                if (lv.cls != nullptr) {
                    err = "yolo_pose decoder: two cls tensors for one stride";
                    return false;
                }
                lv.cls = t.data;
                lv.nc = static_cast<int>(C);
            }
        }
        out.dets.clear();
        out.kpts.clear();
        size_t dropped = 0;
        for (const auto& [stride, lv] : levels) {
            if (!lv.box || !lv.cls || !lv.kpt || lv.H == 0 || lv.nc < 1) {
                err = "yolo_pose decoder: incomplete box/cls/kpt tensors for a "
                      "stride (split pose head needs box [4*reg_max,H,W], cls "
                      "[nc,H,W] and kpt [3*keypoints,H,W])";
                return false;
            }
            const int64_t H = lv.H, W = lv.W;
            const size_t plane = static_cast<size_t>(H) * W;
            const bool logits = plane_is_logits(lv.cls, plane * lv.nc);
            for (int64_t gy = 0; gy < H; ++gy) {
                for (int64_t gx = 0; gx < W; ++gx) {
                    const size_t cell = static_cast<size_t>(gy) * W +
                                        static_cast<size_t>(gx);
                    float dist[4];
                    bool finite = true;
                    for (int d = 0; d < 4 && finite; ++d) {
                        const float* bins =
                            lv.box + (static_cast<size_t>(d) * rm) * plane + cell;
                        finite = dfl_distance(bins, rm, plane, dist[d]);
                    }
                    float s = 0.0f;
                    int cid = 0;
                    if (finite) {
                        float best = 0.0f;
                        for (int c = 0; c < lv.nc; ++c) {
                            float v = lv.cls[static_cast<size_t>(c) * plane + cell];
                            v = logits ? sigmoid(v) : clamp01(v);
                            if (!std::isfinite(v)) {
                                finite = false;
                                break;
                            }
                            if (c == 0 || v > best) {
                                best = v;
                                cid = c;
                            }
                        }
                        s = best;
                    }
                    if (!finite) {  // D4
                        ++dropped;
                        continue;
                    }
                    if (s <= score) continue;
                    const float x0 = (gx + 0.5f - dist[0]) * stride;
                    const float x1 = (gx + 0.5f + dist[2]) * stride;
                    const float y0 = (gy + 0.5f - dist[1]) * stride;
                    const float y1 = (gy + 0.5f + dist[3]) * stride;
                    Detection det;
                    det.cx = (x0 + x1) * 0.5f / model_w;
                    det.cy = (y0 + y1) * 0.5f / model_h;
                    det.w = (x1 - x0) / model_w;
                    det.h = (y1 - y0) / model_h;
                    det.score = s;
                    det.class_id = cid;
                    det.track_id = 0;
                    det.kpt_offset = static_cast<uint32_t>(out.kpts.size());
                    det.kpt_count = static_cast<uint32_t>(K);
                    for (int k = 0; k < K && finite; ++k) {
                        const size_t o = static_cast<size_t>(k) * 3 * plane + cell;
                        const float kx = lv.kpt[o];
                        const float ky = lv.kpt[o + plane];
                        const float kc = lv.kpt[o + 2 * plane];
                        if (!std::isfinite(kx) || !std::isfinite(ky) ||
                            !std::isfinite(kc))
                            finite = false;
                        else
                            // Ultralytics decodes keypoints as
                            //   (2*k + (anchor - 0.5)) * stride  with anchor = g + 0.5
                            // i.e. (2*k + g) * stride — the same cell origin the box
                            // path above uses (gx + 0.5). Written with g - 0.5 the
                            // keypoints land half a cell short (4/8/16 px at stride
                            // 8/16/32); confirmed against the ONNX graph constants:
                            // the box anchor is [0.5,1.5,...] and the kpt addend is
                            // [0,1,2,...], always 0.5 apart.
                            out.kpts.push_back(Keypoint{
                                (2.0f * kx + gx) * stride / model_w,
                                (2.0f * ky + gy) * stride / model_h,
                                sigmoid(kc)});
                    }
                    if (!finite) {  // drop the whole candidate
                        out.kpts.resize(det.kpt_offset);
                        ++dropped;
                        continue;
                    }
                    out.dets.push_back(det);
                }
            }
        }
        note_dropped(err, "yolo_pose decoder", dropped);
        nms(out.dets, out.kpts, nms_th, true);
        return true;
    }

    DecodeSpec spec_;
};

}  // namespace

std::unique_ptr<Decoder> make_yolov8_decoder(DecodeSpec s) {
    return std::make_unique<YoloV8Decoder>(std::move(s));
}

std::unique_ptr<Decoder> make_yolov8_dfl_decoder(DecodeSpec s) {
    return std::make_unique<YoloV8DflDecoder>(std::move(s));
}

std::unique_ptr<Decoder> make_yolo_pose_decoder(DecodeSpec s, std::string& err) {
    if (s.keypoints < 1) {
        err = "yolo_pose decoder requires keypoints >= 1";
        return nullptr;
    }
    return std::make_unique<YoloPoseDecoder>(std::move(s));
}

}  // namespace vb
