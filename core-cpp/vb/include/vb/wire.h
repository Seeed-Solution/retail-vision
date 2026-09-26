// Wire record encoding VBR1/VBE1/VBC1/VBS1 (spec BASE-1 §6.3).
//
// All records: little-endian, first 8 bytes are char[4] magic + u32 body_len
// (body_len excludes the 8 header bytes).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vb {

// Frame-result record payload (VBR1 body). Kept standalone (independent of the
// runtime Detection type) so the encoder is a pure function of its inputs.
struct WireDet {
    float cx = 0, cy = 0, w = 0, h = 0, score = 0;
    int32_t class_id = 0;
    uint32_t track_id = 0;
};

struct WireFrameRec {
    uint32_t stream_index = 0;
    uint64_t seq = 0;
    double wall_ms = 0;
    int32_t src_w = 0, src_h = 0, model_w = 0, model_h = 0;
    float scale = 0, pad_x = 0, pad_y = 0;
    uint8_t align = 0;      // 0 center, 1 top-left
    uint8_t kpt_per_det = 0;
    float inference_ms = 0, queue_delay_ms = 0;
    uint8_t attr_per_det = 0;
    std::vector<WireDet> dets;
    std::vector<float> kpts;   // n_det * kpt_per_det * 3 (x, y, conf)
    std::vector<float> attrs;  // n_det * attr_per_det
};

// VBR1: append the full record (header + body) to out.
void wire_encode_vbr1(const WireFrameRec& r, std::vector<uint8_t>& out);

// VBE1 / VBC1: append header (given 4-char magic, e.g. "VBE1"/"VBC1") + UTF-8
// JSON body.
void wire_encode_json_record(const char magic[4], const std::string& json_body,
                             std::vector<uint8_t>& out);

// VBS1: body = u32 json_len + JSON meta + payload bytes.
void wire_encode_vbs1(const std::string& meta_json, const uint8_t* payload,
                      size_t payload_len, std::vector<uint8_t>& out);

}  // namespace vb
