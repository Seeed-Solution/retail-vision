// Dev-mode raw tensor passthrough VBT1 (spec BASE-1 §6.12, M1.23a).
//
// VBT1 body layout (little-endian):
//   u32 stream_index, u64 seq, f64 wall_ms,
//   i32 src_w, i32 src_h, i32 model_w, i32 model_h,
//   f32 scale, f32 pad_x, f32 pad_y,
//   u8 align, u8 0, u8 0, u8 0, u16 n_tensors, u16 0          (56 bytes)
// then per tensor:
//   u8 dtype, u8 n_dims, u8 nhwc, u8 0,
//   i32 dim0..dim3 (zero-padded to 4),
//   f32 scale, i32 zero_point,
//   u16 name_len, name bytes, u32 data_len, data bytes.
#include "dev_tensor.h"

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

}  // namespace

bool wire_encode_vbt1(const DevTensorFrame& f, std::vector<uint8_t>& out,
                      std::string& err) {
    err.clear();
    // The header carries n_tensors as u16, n_dims per tensor as u8 (≤ 4 by
    // contract), name_len as u16 and data_len as u32. Refuse anything that
    // would be narrowed: a truncated length field leaves the reader parsing
    // the payload as the next field.
    if (f.tensors.size() > 0xFFFFu) {
        err = "VBT1 too many tensors: " + std::to_string(f.tensors.size());
        return false;
    }
    for (const auto& t : f.tensors) {
        if (t.dims.size() > 4) {
            err = "VBT1 tensor has more than 4 dims: " + std::to_string(t.dims.size());
            return false;
        }
        if (t.name.size() > 0xFFFFu) {
            err = "VBT1 tensor name too long: " + std::to_string(t.name.size());
            return false;
        }
        if (t.data.size() > 0xFFFFFFFFull) {
            err = "VBT1 tensor data too large: " + std::to_string(t.data.size());
            return false;
        }
    }
    std::vector<uint8_t> body;
    put_u32(body, f.stream_index);
    put_u64(body, f.seq);
    put_f64(body, f.wall_ms);
    put_i32(body, f.src_w);
    put_i32(body, f.src_h);
    put_i32(body, f.model_w);
    put_i32(body, f.model_h);
    put_f32(body, f.scale);
    put_f32(body, f.pad_x);
    put_f32(body, f.pad_y);
    body.push_back(f.align);
    body.push_back(0);
    body.push_back(0);
    body.push_back(0);
    put_u16(body, static_cast<uint16_t>(f.tensors.size()));
    put_u16(body, 0);
    for (const auto& t : f.tensors) {
        body.push_back(t.dtype);
        body.push_back(static_cast<uint8_t>(t.dims.size()));
        body.push_back(t.nhwc ? 1 : 0);
        body.push_back(0);
        for (size_t i = 0; i < 4; ++i)
            put_i32(body, i < t.dims.size() ? t.dims[i] : 0);
        put_f32(body, t.scale);
        put_i32(body, t.zero_point);
        put_u16(body, static_cast<uint16_t>(t.name.size()));
        body.insert(body.end(), t.name.begin(), t.name.end());
        put_u32(body, static_cast<uint32_t>(t.data.size()));
        body.insert(body.end(), t.data.begin(), t.data.end());
    }
    if (8 + body.size() > kDevTensorMaxRecord) {
        err = "dev tensor record exceeds 16 MiB";
        return false;
    }
    out.insert(out.end(), {'V', 'B', 'T', '1'});
    put_u32(out, static_cast<uint32_t>(body.size()));
    out.insert(out.end(), body.begin(), body.end());
    return true;
}

bool dev_config_from_json(const Json& config, bool allow_dev, DevConfig& out,
                          std::string& err) {
    auto it = config.find("dev");
    if (it == config.end() || it->is_null()) return true;
    if (!it->is_object()) {
        err = "dev must be an object";
        return false;
    }
    const Json& d = *it;
    DevConfig c;
    auto rt = d.find("raw_tensors");
    if (rt != d.end() && !rt->is_null()) {
        if (!rt->is_boolean()) {
            err = "dev.raw_tensors must be a boolean";
            return false;
        }
        c.raw_tensors = rt->get<bool>();
    }
    auto mf = d.find("max_fps");
    if (mf != d.end() && !mf->is_null()) {
        if (!mf->is_number()) {
            err = "dev.max_fps must be a number in (0, 2]";
            return false;
        }
        c.max_fps = mf->get<double>();
        if (!(c.max_fps > 0.0) || c.max_fps > 2.0) {
            err = "dev.max_fps must be in (0, 2]";
            return false;
        }
    }
    auto ms = d.find("max_streams");
    if (ms != d.end() && !ms->is_null()) {
        if (!ms->is_number_integer()) {
            err = "dev.max_streams must be 1";
            return false;
        }
        c.max_streams = ms->get<int>();
        if (c.max_streams != 1) {
            err = "dev.max_streams must be 1";
            return false;
        }
    }
    if (c.raw_tensors && !allow_dev) {
        err = "dev.raw_tensors is not allowed in production configs";
        return false;
    }
    out = c;
    return true;
}

}  // namespace vb
