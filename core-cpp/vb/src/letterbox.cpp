#include "vb/letterbox.h"

#include <cmath>

namespace vb {

LetterboxGeom LetterboxGeom::fit(int src_w, int src_h, int model_w, int model_h, Align a) {
    return letterbox_fit(src_w, src_h, model_w, model_h, a);
}

LetterboxGeom letterbox_fit(int src_w, int src_h, int model_w, int model_h, Align a) {
    if (src_w <= 0 || src_h <= 0 || model_w <= 0 || model_h <= 0) {
        throw std::invalid_argument("letterbox sizes must be > 0");
    }
    LetterboxGeom g;
    g.src_w = static_cast<int32_t>(src_w);
    g.src_h = static_cast<int32_t>(src_h);
    g.model_w = static_cast<int32_t>(model_w);
    g.model_h = static_cast<int32_t>(model_h);
    g.align = a;
    g.scale = std::min(static_cast<float>(model_w) / src_w, static_cast<float>(model_h) / src_h);
    // Python round() is banker's rounding; replicate it for exact cross-language parity.
    auto py_round = [](float v) {
        float f = std::floor(v);
        float r = v - f;
        if (r > 0.5f) return static_cast<int>(f) + 1;
        if (r < 0.5f) return static_cast<int>(f);
        return (static_cast<int>(f) % 2 == 0) ? static_cast<int>(f) : static_cast<int>(f) + 1;
    };
    int sw = py_round(src_w * g.scale);
    int sh = py_round(src_h * g.scale);
    if (a == Align::Center) {
        g.pad_x = static_cast<float>((model_w - sw) / 2);
        g.pad_y = static_cast<float>((model_h - sh) / 2);
    } else {
        g.pad_x = 0;
        g.pad_y = 0;
    }
    return g;
}

void LetterboxGeom::to_source_norm(float mx, float my, float& sx, float& sy) const {
    float px = mx * model_w;
    float py = my * model_h;
    sx = ((px - pad_x) / scale) / src_w;
    sy = ((py - pad_y) / scale) / src_h;
}

void LetterboxGeom::box_to_source_norm(float cx, float cy, float w, float h,
                                       float& scx, float& scy, float& sw, float& sh) const {
    to_source_norm(cx, cy, scx, scy);
    sw = w * model_w / (scale * src_w);
    sh = h * model_h / (scale * src_h);
}

void LetterboxGeom::to_model_norm(float sx, float sy, float& mx, float& my) const {
    float px = (sx * src_w) * scale + pad_x;
    float py = (sy * src_h) * scale + pad_y;
    mx = px / model_w;
    my = py / model_h;
}

}  // namespace vb
