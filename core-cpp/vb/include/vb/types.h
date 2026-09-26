// Core shared types of the native compute core (spec BASE-1 §6.1, subset for M1.3/M1.4).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vb {

enum class Align : uint8_t { Center = 0, TopLeft = 1 };

inline Align align_from_int(int v) {
    if (v == 0) return Align::Center;
    if (v == 1) return Align::TopLeft;
    throw std::invalid_argument("align must be 0 (center) or 1 (top_left)");
}

struct LetterboxGeom {
    int32_t src_w = 0, src_h = 0, model_w = 0, model_h = 0;
    float scale = 0, pad_x = 0, pad_y = 0;
    Align align = Align::Center;

    static LetterboxGeom fit(int src_w, int src_h, int model_w, int model_h, Align a);
    void to_source_norm(float mx, float my, float& sx, float& sy) const;
    void box_to_source_norm(float cx, float cy, float w, float h,
                            float& scx, float& scy, float& sw, float& sh) const;
    void to_model_norm(float sx, float sy, float& mx, float& my) const;
};

struct Keypoint {
    float x = 0, y = 0, conf = 0;
};

struct Detection {
    float cx = 0, cy = 0, w = 0, h = 0, score = 0;
    int32_t class_id = 0;
    uint32_t track_id = 0;
    uint32_t kpt_offset = 0, kpt_count = 0;
};

}  // namespace vb
