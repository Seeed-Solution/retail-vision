// Shared helper: build a WireFrameRec from a wire_case*.json fixture.
#pragma once

#include <stdexcept>
#include <string>

#include "vb/json.h"
#include "vb/wire.h"

#include "dev_tensor.h"

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

// ---- VBT1 dev-mode tensor frame (§6.12; wire_tensor_case*.json) ----

inline std::vector<uint8_t> bytes_from_hex(const std::string& hex) {
    if (hex.size() % 2 != 0)
        throw std::invalid_argument("data_hex must have even length");
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::invalid_argument("bad hex digit");
    };
    for (size_t i = 0; i < hex.size(); i += 2)
        out.push_back(static_cast<uint8_t>(nib(hex[i]) * 16 + nib(hex[i + 1])));
    return out;
}

inline DevTensorFrame dev_tensor_frame_from_case_json(const Json& j) {
    DevTensorFrame f;
    f.stream_index = j.at("stream_index").get<uint32_t>();
    f.seq = j.at("seq").get<uint64_t>();
    f.wall_ms = j.at("wall_ms").get<double>();
    f.src_w = j.at("src_w").get<int32_t>();
    f.src_h = j.at("src_h").get<int32_t>();
    f.model_w = j.at("model_w").get<int32_t>();
    f.model_h = j.at("model_h").get<int32_t>();
    f.scale = j.at("scale").get<float>();
    f.pad_x = j.at("pad_x").get<float>();
    f.pad_y = j.at("pad_y").get<float>();
    f.align = j.at("align").get<uint8_t>();
    for (const auto& tj : j.at("tensors")) {
        DevTensor t;
        t.name = tj.at("name").get<std::string>();
        t.dtype = tj.at("dtype").get<uint8_t>();
        t.nhwc = tj.at("nhwc").get<bool>();
        for (const auto& d : tj.at("dims")) t.dims.push_back(d.get<int32_t>());
        t.scale = tj.at("scale").get<float>();
        t.zero_point = tj.at("zero_point").get<int32_t>();
        t.data = bytes_from_hex(tj.at("data_hex").get<std::string>());
        f.tensors.push_back(std::move(t));
    }
    return f;
}

}  // namespace vb
