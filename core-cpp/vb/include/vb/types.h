// Core shared types of the native compute core (spec BASE-1 §6.1, subset for M1.3/M1.4).
#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

namespace vb {

enum class Align : uint8_t { Center = 0, TopLeft = 1 };

inline Align align_from_int(int v) {
    if (v == 0) return Align::Center;
    if (v == 1) return Align::TopLeft;
    throw std::invalid_argument("align must be 0 (center) or 1 (top_left)");
}

inline Align align_from_string(const std::string& v) {
    if (v == "center") return Align::Center;
    if (v == "top_left") return Align::TopLeft;
    throw std::invalid_argument("align must be center or top_left");
}

// What the model's input tensor expects. The frame format from the pipeline
// (RGB or BGR888) says nothing about the model, so this is declared rather
// than inferred from FrameBuf::fmt.
enum class ColorOrder : uint8_t { BGR = 0, RGB = 1 };

struct InputSpec {
    ColorOrder color_order = ColorOrder::BGR;
    // 255.0 => the model takes 0..1; 1.0 => it takes 0..255 unchanged.
    float divide = 1.0f;
    Align align = Align::Center;

    // Defaults follow each family's own convention: YOLOX's released ONNX
    // takes BGR 0-255 (its demo feeds cv2 output unscaled), Ultralytics YOLOv8
    // and classifier exports take RGB 0-1.
    static InputSpec default_for_decoder(const std::string& type) {
        if (type == "yolov8" || type == "yolov8_dfl" || type == "yolo_pose" ||
            type == "classify")
            return InputSpec{ColorOrder::RGB, 255.0f};
        return InputSpec{ColorOrder::BGR, 1.0f};  // yolox, raw
    }
};

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
