#include "vb/wire.h"

#include <cstring>

namespace vb {
namespace {

void put_u16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xff));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
}

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

void put_u64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

void put_i32(std::vector<uint8_t>& out, int32_t v) {
    put_u32(out, static_cast<uint32_t>(v));
}

void put_f32(std::vector<uint8_t>& out, float v) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(v), "float must be 32-bit");
    std::memcpy(&bits, &v, sizeof(bits));
    put_u32(out, bits);
}

void put_f64(std::vector<uint8_t>& out, double v) {
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(v), "double must be 64-bit");
    std::memcpy(&bits, &v, sizeof(bits));
    put_u64(out, bits);
}

void put_header(std::vector<uint8_t>& out, const char magic[4], uint32_t body_len) {
    out.insert(out.end(), magic, magic + 4);
    put_u32(out, body_len);
}

}  // namespace

void wire_encode_vbr1(const WireFrameRec& r, std::vector<uint8_t>& out) {
    const uint16_t n_det = static_cast<uint16_t>(r.dets.size());
    const size_t body_len =
        64 + size_t(n_det) * 28 +
        size_t(n_det) * r.kpt_per_det * 3 * 4 +
        size_t(n_det) * r.attr_per_det * 4;

    put_header(out, "VBR1", static_cast<uint32_t>(body_len));
    put_u32(out, r.stream_index);
    put_u64(out, r.seq);
    put_f64(out, r.wall_ms);
    put_i32(out, r.src_w);
    put_i32(out, r.src_h);
    put_i32(out, r.model_w);
    put_i32(out, r.model_h);
    put_f32(out, r.scale);
    put_f32(out, r.pad_x);
    put_f32(out, r.pad_y);
    out.push_back(r.align);
    out.push_back(r.kpt_per_det);
    put_u16(out, n_det);
    put_f32(out, r.inference_ms);
    put_f32(out, r.queue_delay_ms);
    out.push_back(r.attr_per_det);
    out.push_back(0);  // reserved u8
    put_u16(out, 0);   // reserved u16
    for (const auto& d : r.dets) {
        put_f32(out, d.cx);
        put_f32(out, d.cy);
        put_f32(out, d.w);
        put_f32(out, d.h);
        put_f32(out, d.score);
        put_i32(out, d.class_id);
        put_u32(out, d.track_id);
    }
    for (float v : r.kpts) put_f32(out, v);
    for (float v : r.attrs) put_f32(out, v);
}

void wire_encode_json_record(const char magic[4], const std::string& json_body,
                             std::vector<uint8_t>& out) {
    put_header(out, magic, static_cast<uint32_t>(json_body.size()));
    out.insert(out.end(), json_body.begin(), json_body.end());
}

void wire_encode_vbs1(const std::string& meta_json, const uint8_t* payload,
                      size_t payload_len, std::vector<uint8_t>& out) {
    put_header(out, "VBS1", static_cast<uint32_t>(4 + meta_json.size() + payload_len));
    put_u32(out, static_cast<uint32_t>(meta_json.size()));
    out.insert(out.end(), meta_json.begin(), meta_json.end());
    out.insert(out.end(), payload, payload + payload_len);
}

}  // namespace vb
