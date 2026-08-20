#ifndef _HAILO_DETECTOR_H_
#define _HAILO_DETECTOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include <hailo/hailort.h>

#include "detection_box.h"

namespace retail_vision {

struct RawTensor {
    const uint8_t* data = nullptr;
    size_t size = 0;
    hailo_vstream_info_t info{};
};

// Decodes the output of a HEF whose NMS runs on-chip (the Hailo Model Zoo
// yolov8n/yolo11n detection HEFs are compiled that way, so nothing here has to
// re-implement DFL decoding or NMS on the CPU).
//
// Layout of a HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS float32 buffer:
//   for class in [0, number_of_classes):
//       float32 bbox_count
//       bbox_count x hailo_bbox_float32_t { y_min, x_min, y_max, x_max, score }
// Coordinates are already normalized to the model input square.
//
// Only `class_filter` is read: the retail pipeline needs COCO class 0 (person),
// and walking the other 79 class blocks would cost a scan of a 160 KB buffer
// per frame for boxes that are discarded immediately after.
std::vector<DetectionBox> decodeHailoNms(const std::vector<RawTensor>& tensors,
                                         float score_threshold,
                                         int class_filter);

// One-line description of every tensor, for the startup log. This is how a
// deployment proves which output format the HEF actually produced rather than
// assuming the one the decoder was written against.
std::string describeTensors(const std::vector<RawTensor>& tensors);

// True when at least one tensor carries an NMS format order, i.e. the decoder
// above is applicable to this HEF.
bool hasNmsOutput(const std::vector<RawTensor>& tensors);

}  // namespace retail_vision

#endif  // _HAILO_DETECTOR_H_
