// Shared helper: build a WireFrameRec from a wire_case*.json fixture.
#pragma once

#include <stdexcept>
#include <string>

#include "vb/json.h"
#include "vb/wire.h"

namespace vb {

inline WireFrameRec wire_frame_from_case_json(const Json& j) {
    WireFrameRec r;
    r.stream_index = j.at("stream_index").get<uint32_t>();
    r.seq = j.at("seq").get<uint64_t>();
    r.wall_ms = j.at("wall_ms").get<double>();
    r.src_w = j.at("src_w").get<int32_t>();
    r.src_h = j.at("src_h").get<int32_t>();
    r.model_w = j.at("model_w").get<int32_t>();
    r.model_h = j.at("model_h").get<int32_t>();
    r.scale = j.at("scale").get<float>();
    r.pad_x = j.at("pad_x").get<float>();
    r.pad_y = j.at("pad_y").get<float>();
    r.align = j.at("align").get<uint8_t>();
    r.inference_ms = j.at("inference_ms").get<float>();
    r.queue_delay_ms = j.at("queue_delay_ms").get<float>();

    const auto& dets = j.at("detections");
    uint8_t kpt_per_det = 0, attr_per_det = 0;
    if (!dets.empty()) {
        kpt_per_det = static_cast<uint8_t>(dets[0].at("keypoints").size());
        attr_per_det = static_cast<uint8_t>(dets[0].at("attrs").size());
        for (const auto& d : dets) {
            if (d.at("keypoints").size() != kpt_per_det ||
                d.at("attrs").size() != attr_per_det) {
                throw std::invalid_argument("inconsistent per-det keypoint/attr counts");
            }
        }
    }
    r.kpt_per_det = kpt_per_det;
    r.attr_per_det = attr_per_det;
    for (const auto& d : dets) {
        WireDet wd;
        wd.cx = d.at("cx").get<float>();
        wd.cy = d.at("cy").get<float>();
        wd.w = d.at("w").get<float>();
        wd.h = d.at("h").get<float>();
        wd.score = d.at("score").get<float>();
        wd.class_id = d.at("class_id").get<int32_t>();
        wd.track_id = d.at("track_id").get<uint32_t>();
        r.dets.push_back(wd);
        for (const auto& k : d.at("keypoints")) {
            r.kpts.push_back(k[0].get<float>());
            r.kpts.push_back(k[1].get<float>());
            r.kpts.push_back(k[2].get<float>());
        }
        for (const auto& a : d.at("attrs")) r.attrs.push_back(a.get<float>());
    }
    return r;
}

}  // namespace vb
