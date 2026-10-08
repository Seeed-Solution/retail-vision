// Decoder selection factory (spec BASE-1 §6.11, M1.15).
#include <cmath>
#include <string>

#include "post/tensor_check.h"
#include "vb/decoder.h"
#include "vb/json.h"
#include "vb/post.h"

namespace vb {
namespace {

// ---- yolox: [N, 5+nc] (obj*cls), delegates to the shared yolox_decode ----
//
// Two output layouts exist in the wild and the decoder picks by geometry:
//   * raw head (rknn_model_zoo-style export, the M2.1 parity model): rows hold
//     pre-decode values [tx,ty,tw,th,obj,c...]; the box becomes a pixel corner
//     only after the per-stride grid decode. When `strides` tiles the model
//     canvas to exactly the anchor count, this decoder applies that decode
//     here (levels in the declared order, row-major inside a level), then
//     hands pixel corners to yolox_decode. Measured on the M2.1 model: with
//     strides [8,16,32] the top anchor decodes to the bus at cx=203 cy=189
//     w=172 h=115 of the 416 canvas; corner-math on the same rows produces
//     sub-pixel corner boxes instead (the M2.1 "CPU detects nothing" bug).
//   * decoded export (legacy M1.9 assumption, fixture yolox_out_case1.json):
//     rows are already [x1,y1,x2,y2,...] pixel corners. Any geometry the
//     strides cannot tile falls through to this path unchanged.
class YoloXDecoder : public Decoder {
public:
    explicit YoloXDecoder(DecodeSpec s) : spec_(std::move(s)) {}

    int keypoints() const override { return 0; }

    bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                float score, float nms_th, DetectionResult& out,
                std::string& err) override {
        if (outs == nullptr || n != 1) {
            err = "yolox decoder expects one [N, 5+nc] output tensor";
            return false;
        }
        if (!post_detail::check_canvas(model_w, model_h, "yolox decoder", err))
            return false;
        if (!post_detail::check_view(outs[0], "yolox decoder", err)) return false;
        if (outs[0].dims.size() != 2) {
            err = "yolox decoder expects one [N, 5+nc] output tensor";
            return false;
        }
        const int64_t d0 = outs[0].dims[0], d1 = outs[0].dims[1];
        const int nc = spec_.num_classes;
        int anchors = 0, ch = 0;
        bool transposed = false;
        if (d1 == 5 + nc) {
            anchors = static_cast<int>(d0);
            ch = static_cast<int>(d1);
        } else if (d0 == 5 + nc) {  // [5+nc, N]
            anchors = static_cast<int>(d1);
            ch = static_cast<int>(d0);
            transposed = true;
        } else {
            err = "yolox decoder: output shape does not match num_classes";
            return false;
        }
        const float* data = outs[0].data;
        std::vector<float> tmp;
        if (transposed) {
            tmp.resize(static_cast<size_t>(anchors) * ch);
            for (int i = 0; i < anchors; ++i)
                for (int c = 0; c < ch; ++c)
                    tmp[i * ch + c] = data[c * anchors + i];
            data = tmp.data();
        }
        // Raw-head grid decode: only when the declared strides tile the model
        // canvas to exactly this anchor count.
        std::vector<float> corners;
        if (raw_head_geometry(anchors, model_w, model_h)) {
            corners.resize(static_cast<size_t>(anchors) * ch);
            grid_decode(data, corners.data(), anchors, ch, model_w, model_h);
            data = corners.data();
        }
        out.dets.clear();
        out.kpts.clear();
        yolox_decode(data, anchors, nc, model_w, model_h, score, out.dets);
        nms(out.dets, out.kpts, nms_th, true);
        return true;
    }

private:
    // True when every stride divides the canvas and the tiled grid sums to
    // exactly `anchors` levels-in-declared-order, row-major per level.
    bool raw_head_geometry(int anchors, int model_w, int model_h) const {
        if (spec_.strides.empty()) return false;
        long long total = 0;
        for (int s : spec_.strides) {
            if (s <= 0 || model_w % s != 0 || model_h % s != 0) return false;
            total += static_cast<long long>(model_w / s) * (model_h / s);
        }
        return total == anchors;
    }

    // [tx,ty,tw,th,obj,c...] rows -> [x1,y1,x2,y2,obj,c...] pixel corners.
    // Rows pass through untouched beyond the first four cells.
    void grid_decode(const float* src, float* dst, int anchors, int ch,
                     int model_w, int model_h) const {
        const size_t row = static_cast<size_t>(ch);
        long long start = 0;
        for (int s : spec_.strides) {
            const int gw = model_w / s, gh = model_h / s;
            for (int gy = 0; gy < gh; ++gy) {
                for (int gx = 0; gx < gw; ++gx) {
                    const float* p = src + static_cast<size_t>(start) * row;
                    float* q = dst + static_cast<size_t>(start) * row;
                    const float tx = spec_.grid_center_activation ==
                                             GridCenterActivation::Sigmoid
                                         ? sigmoid(p[0])
                                         : p[0];
                    const float ty = spec_.grid_center_activation ==
                                             GridCenterActivation::Sigmoid
                                         ? sigmoid(p[1])
                                         : p[1];
                    const float cx = (tx + gx) * s;
                    const float cy = (ty + gy) * s;
                    const float bw = std::exp(p[2]) * s;
                    const float bh = std::exp(p[3]) * s;
                    q[0] = cx - bw / 2.0f;
                    q[1] = cy - bh / 2.0f;
                    q[2] = cx + bw / 2.0f;
                    q[3] = cy + bh / 2.0f;
                    for (int c = 4; c < ch; ++c) q[c] = p[c];
                    ++start;
                }
            }
        }
    }

    static float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

    DecodeSpec spec_;
};

// ---- raw: no decode, VBR1 n_det = 0 (§6.12 dev mode) ----
class RawDecoder : public Decoder {
public:
    int keypoints() const override { return 0; }
    bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                float score, float nms_th, DetectionResult& out,
                std::string& err) override {
        (void)outs; (void)n; (void)model_w; (void)model_h;
        (void)score; (void)nms_th; (void)err;
        out.dets.clear();
        out.kpts.clear();
        return true;
    }
};

bool parse_spec(const Json& j, DecodeSpec& s, std::string& err) {
    s.type = j.at("type").get<std::string>();
    auto get_int = [&](const char* key, int& v, int lo) -> bool {
        auto it = j.find(key);
        if (it == j.end() || it->is_null()) return true;
        if (!it->is_number_integer() || it->get<long long>() < lo) {
            err = std::string("decoder: ") + key + " must be an integer >= " +
                  std::to_string(lo);
            return false;
        }
        v = it->get<int>();
        return true;
    };
    if (!get_int("num_classes", s.num_classes, 1)) return false;
    if (!get_int("reg_max", s.reg_max, 1)) return false;
    if (!get_int("keypoints", s.keypoints, 0)) return false;
    if (!get_int("top_k", s.top_k, 1)) return false;
    auto st = j.find("strides");
    if (st != j.end() && !st->is_null()) {
        if (!st->is_array() || st->empty()) {
            err = "decoder: strides must be a non-empty array of ints";
            return false;
        }
        s.strides.clear();
        for (const auto& v : *st) {
            if (!v.is_number_integer() || v.get<int>() <= 0) {
                err = "decoder: strides must be a non-empty array of ints";
                return false;
            }
            s.strides.push_back(v.get<int>());
        }
    }
    auto sm = j.find("softmax");
    if (sm != j.end() && !sm->is_null()) {
        if (!sm->is_boolean()) {
            err = "decoder: softmax must be a boolean";
            return false;
        }
        s.softmax = sm->get<bool>();
    }
    auto ga = j.find("grid_center_activation");
    if (ga != j.end() && !ga->is_null()) {
        if (s.type != "yolox") {
            err = "decoder: grid_center_activation only applies to yolox";
            return false;
        }
        if (!ga->is_string()) {
            err = "decoder: grid_center_activation must be \"none\" or \"sigmoid\"";
            return false;
        }
        const std::string mode = ga->get<std::string>();
        if (mode == "none") s.grid_center_activation = GridCenterActivation::None;
        else if (mode == "sigmoid") s.grid_center_activation = GridCenterActivation::Sigmoid;
        else {
            err = "decoder: grid_center_activation must be \"none\" or \"sigmoid\"";
            return false;
        }
    }
    return true;
}

}  // namespace

// Implemented in yolov8_decode.cpp / classify.cpp.
std::unique_ptr<Decoder> make_yolov8_decoder(DecodeSpec s);
std::unique_ptr<Decoder> make_yolov8_dfl_decoder(DecodeSpec s);
// yolo_pose validates `keypoints` itself: it is the only parameter that
// decides the shape of the keypoint head, so a zero/absent value is an error
// rather than a silently keypoint-less pose decoder.
std::unique_ptr<Decoder> make_yolo_pose_decoder(DecodeSpec s, std::string& err);
std::unique_ptr<Decoder> make_classify_decoder(DecodeSpec s);

std::unique_ptr<Decoder> make_decoder(const std::string& decoder_json,
                                      std::string& err) {
    err.clear();
    if (decoder_json.empty()) return nullptr;
    Json j;
    try {
        j = Json::parse(decoder_json);
    } catch (const std::exception& e) {
        err = std::string("decoder: invalid json: ") + e.what();
        return nullptr;
    }
    if (j.is_null()) return nullptr;
    if (!j.is_object() || !j.contains("type") || !j.at("type").is_string()) {
        err = "decoder: config must be an object with a \"type\" string";
        return nullptr;
    }
    DecodeSpec spec;
    if (!parse_spec(j, spec, err)) return nullptr;
    if (spec.type == "ctc") {
        err = "ctc is only valid in stage2";
        return nullptr;
    }
    if (spec.type == "yolox") return std::make_unique<YoloXDecoder>(spec);
    if (spec.type == "yolov8") return make_yolov8_decoder(std::move(spec));
    if (spec.type == "yolov8_dfl") return make_yolov8_dfl_decoder(std::move(spec));
    if (spec.type == "yolo_pose") return make_yolo_pose_decoder(std::move(spec), err);
    if (spec.type == "classify") return make_classify_decoder(std::move(spec));
    if (spec.type == "raw") return std::make_unique<RawDecoder>();
    err = "decoder: unknown type " + spec.type;
    return nullptr;
}

}  // namespace vb
