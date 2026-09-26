// Decoder selection factory (spec BASE-1 §6.11, M1.15).
#include <cmath>
#include <string>

#include "vb/decoder.h"
#include "vb/json.h"
#include "vb/post.h"

namespace vb {
namespace {

// ---- yolox: [N, 5+nc] (obj*cls), delegates to the shared yolox_decode ----
class YoloXDecoder : public Decoder {
public:
    explicit YoloXDecoder(DecodeSpec s) : spec_(std::move(s)) {}

    int keypoints() const override { return 0; }

    bool decode(const TensorView* outs, size_t n, int model_w, int model_h,
                float score, float nms_th, DetectionResult& out,
                std::string& err) override {
        if (n != 1 || outs[0].dims.size() != 2) {
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
        out.dets.clear();
        out.kpts.clear();
        yolox_decode(data, anchors, nc, model_w, model_h, score, out.dets);
        nms(out.dets, out.kpts, nms_th, true);
        return true;
    }

private:
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
    return true;
}

}  // namespace

// Implemented in yolov8_decode.cpp / classify.cpp.
std::unique_ptr<Decoder> make_yolov8_decoder(DecodeSpec s);
std::unique_ptr<Decoder> make_yolov8_dfl_decoder(DecodeSpec s);
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
    if (spec.type == "classify") return make_classify_decoder(std::move(spec));
    if (spec.type == "raw") return std::make_unique<RawDecoder>();
    err = "decoder: unknown type " + spec.type;
    return nullptr;
}

}  // namespace vb
