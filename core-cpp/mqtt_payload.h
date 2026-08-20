#ifndef _MQTT_PAYLOAD_H_
#define _MQTT_PAYLOAD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "person_tracker.h"
#include "zone_metrics.h"

namespace retail_vision {

// Build the VisionPayload JSON published to the MQTT results topic:
// zone metrics + per-person data.
// frame_width/frame_height: display resolution for absolute pixel coordinate output
// model_width/model_height: inference input resolution (for letterbox correction)
std::string buildVisionJson(uint64_t timestamp_ms, uint32_t frame_id,
                            float fps, float inference_time_ms,
                            const ZoneSnapshot& zone,
                            const std::vector<TrackedPerson>& persons,
                            int frame_width, int frame_height,
                            int model_width, int model_height);

}  // namespace retail_vision

#endif  // _MQTT_PAYLOAD_H_
