#include "check.h"
#include "cpu_backend.h"
#include "dev_tensor.h"
#include "vb/backend.h"
#include "vb/json.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

vb::FrameBuf frame_from(const std::vector<uint8_t>& pixels, int w, int h,
                        int stride, vb::PixFmt fmt) {
    auto owned = std::make_shared<std::vector<uint8_t>>(pixels);
    vb::FrameBuf f;
    f.w = w;
    f.h = h;
    f.stride = stride;
    f.fmt = fmt;
    f.mem = vb::Mem::Host;
    f.host = owned->data();
    f.hold = owned;
    return f;
}

std::vector<float> infer_raw(const std::string& model, const vb::FrameBuf& frame,
                             const char* input_json) {
    vb::Json cfg = vb::json_parse(input_json);
    cfg["model_path"] = model;
    std::string err;
    auto backend = vb::make_cpu_backend(vb::json_dump(cfg), err);
    CHECK(backend != nullptr);
    auto ctx = backend->create_context(0, err);
    CHECK(ctx != nullptr);
    const vb::FrameBuf* frames[1] = {&frame};
    vb::DetectionResult result;
    CHECK(ctx->infer(frames, 1, 0.25f, 0.5f, &result, err) == 0);
    auto* raw = dynamic_cast<vb::RawTensorSource*>(ctx.get());
    CHECK(raw != nullptr);
    std::vector<vb::DevTensor> tensors;
    CHECK(raw->last_raw_tensors(tensors));
    CHECK(tensors.size() == 1 && tensors[0].dtype == 0);
    CHECK(tensors[0].data.size() % sizeof(float) == 0);
    std::vector<float> result_pixels(tensors[0].data.size() / sizeof(float));
    std::memcpy(result_pixels.data(), tensors[0].data.data(), tensors[0].data.size());
    return result_pixels;
}

float at(const std::vector<float>& v, int side, int channel, int y, int x) {
    return v[(static_cast<size_t>(channel) * side + y) * side + x];
}

}  // namespace

int main(int argc, char** argv) {
    CHECK(argc == 3);
    const std::string model4 = argv[1];
    const std::string model5 = argv[2];

    // A 4x2 RGB frame into a 4x4 canvas has one padding row at each edge.
    // The stride includes four bytes of row padding to prove that sampling
    // never walks into the next row's alignment bytes.
    const int stride = 16;
    std::vector<uint8_t> rgb(static_cast<size_t>(stride) * 2, 0xee);
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 4; ++x) {
        uint8_t* p = rgb.data() + y * stride + x * 3;
        p[0] = static_cast<uint8_t>(10 + y * 100 + x);  // R
        p[1] = static_cast<uint8_t>(50 + y * 100 + x);  // G
        p[2] = static_cast<uint8_t>(90 + y * 100 + x);  // B
    }
    auto f = frame_from(rgb, 4, 2, stride, vb::PixFmt::RGB888);
    const char* bgr_255 = R"({"decoder":{"type":"raw"},"input":{"color_order":"bgr","divide":1}})";
    auto bgr = infer_raw(model4, f, bgr_255);
    CHECK(bgr.size() == 3u * 4u * 4u);
    for (int c = 0; c < 3; ++c) for (int x = 0; x < 4; ++x) {
        CHECK_NEAR(at(bgr, 4, c, 0, x), 114.0, 1e-6);
        CHECK_NEAR(at(bgr, 4, c, 3, x), 114.0, 1e-6);
    }
    CHECK_NEAR(at(bgr, 4, 0, 1, 0), 90, 1e-6);
    CHECK_NEAR(at(bgr, 4, 1, 1, 2), 52, 1e-6);
    CHECK_NEAR(at(bgr, 4, 2, 2, 3), 113, 1e-6);

    // A BGR source with RGB model input must produce the same semantic
    // channels, while divide=255 is applied after the channel conversion.
    std::vector<uint8_t> bgr_source = rgb;
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 4; ++x)
        std::swap(bgr_source[y * stride + x * 3], bgr_source[y * stride + x * 3 + 2]);
    auto fb = frame_from(bgr_source, 4, 2, stride, vb::PixFmt::BGR888);
    const char* rgb_01 = R"({"decoder":{"type":"raw"},"input":{"color_order":"rgb","divide":255}})";
    auto rgb01 = infer_raw(model4, fb, rgb_01);
    CHECK_NEAR(at(rgb01, 4, 0, 1, 0), 10.0 / 255.0, 1e-6);
    CHECK_NEAR(at(rgb01, 4, 1, 2, 2), 152.0 / 255.0, 1e-6);
    CHECK_NEAR(at(rgb01, 4, 2, 2, 3), 193.0 / 255.0, 1e-6);

    // Odd 16:9-like geometry: 7x3 -> 5x5 rounds the resized height to 2,
    // leaving three complete padding rows (top=1, bottom=2).
    std::vector<uint8_t> odd(7 * 3 * 3, 7);
    auto fo = frame_from(odd, 7, 3, 21, vb::PixFmt::RGB888);
    auto odd_out = infer_raw(model5, fo, bgr_255);
    CHECK(odd_out.size() == 3u * 5u * 5u);
    CHECK_NEAR(at(odd_out, 5, 0, 0, 2), 114.0, 1e-6);
    CHECK_NEAR(at(odd_out, 5, 0, 1, 2), 7.0, 1e-6);
    CHECK_NEAR(at(odd_out, 5, 0, 3, 2), 114.0, 1e-6);
    CHECK_NEAR(at(odd_out, 5, 2, 4, 4), 114.0, 1e-6);

    std::puts("cpu_preprocess: production CpuBackend identity oracles PASS");
    return 0;
}
