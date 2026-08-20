#include "detector.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace retail_vision {
namespace {

bool isNmsOrder(hailo_format_order_t order) {
    return order == HAILO_FORMAT_ORDER_HAILO_NMS ||
           order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS ||
           order == HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP;
}

const char* orderName(hailo_format_order_t order) {
    switch (order) {
        case HAILO_FORMAT_ORDER_HAILO_NMS:          return "HAILO_NMS(deprecated)";
        case HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS: return "HAILO_NMS_BY_CLASS";
        case HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE: return "HAILO_NMS_BY_SCORE";
        case HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP:  return "HAILO_NMS_ON_CHIP";
        case HAILO_FORMAT_ORDER_NHWC:               return "NHWC";
        case HAILO_FORMAT_ORDER_NHCW:               return "NHCW";
        case HAILO_FORMAT_ORDER_FCR:                return "FCR";
        default:                                    return "OTHER";
    }
}

}  // namespace

bool hasNmsOutput(const std::vector<RawTensor>& tensors) {
    for (const auto& t : tensors) {
        if (isNmsOrder(t.info.format.order)) return true;
    }
    return false;
}

std::string describeTensors(const std::vector<RawTensor>& tensors) {
    std::ostringstream o;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        if (i) o << "; ";
        o << t.info.name << " order=" << orderName(t.info.format.order)
          << " type=" << static_cast<int>(t.info.format.type)
          << " bytes=" << t.size;
        if (isNmsOrder(t.info.format.order)) {
            o << " classes=" << t.info.nms_shape.number_of_classes
              << " max_bboxes_per_class=" << t.info.nms_shape.max_bboxes_per_class;
        } else {
            o << " shape=" << t.info.shape.height << "x" << t.info.shape.width
              << "x" << t.info.shape.features;
        }
    }
    return o.str();
}

std::vector<DetectionBox> decodeHailoNms(const std::vector<RawTensor>& tensors,
                                         float score_threshold,
                                         int class_filter) {
    std::vector<DetectionBox> out;

    for (const auto& t : tensors) {
        if (!isNmsOrder(t.info.format.order)) continue;
        if (t.info.format.type != HAILO_FORMAT_TYPE_FLOAT32) continue;
        if (t.data == nullptr) continue;

        const uint32_t classes = t.info.nms_shape.number_of_classes;
        const uint32_t max_per_class = t.info.nms_shape.max_bboxes_per_class;
        if (class_filter < 0 || static_cast<uint32_t>(class_filter) >= classes) continue;

        // Each class block is a float32 count followed by max_per_class slots,
        // so the offset of a class is computable without walking the ones before
        // it. Reading the counts sequentially would work too but touches every
        // block; the fixed stride keeps the person lookup to two cache lines.
        const size_t stride = sizeof(float) + static_cast<size_t>(max_per_class) * 5 * sizeof(float);
        const size_t base = static_cast<size_t>(class_filter) * stride;
        if (base + sizeof(float) > t.size) continue;

        float count_f = 0.0f;
        std::memcpy(&count_f, t.data + base, sizeof(float));
        if (!(count_f > 0.0f)) continue;

        uint32_t count = static_cast<uint32_t>(count_f);
        count = std::min(count, max_per_class);

        for (uint32_t i = 0; i < count; ++i) {
            const size_t off = base + sizeof(float) + static_cast<size_t>(i) * 5 * sizeof(float);
            if (off + 5 * sizeof(float) > t.size) break;

            float b[5];
            std::memcpy(b, t.data + off, sizeof(b));
            const float y_min = b[0], x_min = b[1], y_max = b[2], x_max = b[3], score = b[4];
            if (score < score_threshold) continue;

            DetectionBox d;
            d.x = (x_min + x_max) * 0.5f;
            d.y = (y_min + y_max) * 0.5f;
            d.w = x_max - x_min;
            d.h = y_max - y_min;
            d.score = score;
            d.target = class_filter;
            if (d.w <= 0.0f || d.h <= 0.0f) continue;
            out.push_back(d);
        }
    }

    return out;
}

}  // namespace retail_vision
