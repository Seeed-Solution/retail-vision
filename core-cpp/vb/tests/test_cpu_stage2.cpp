#include "check.h"
#include "cpu_backend.h"
#include "vb/decoder.h"
#include <algorithm>
#include <limits>
#include <vector>
using namespace vb;
void expected(const TensorView& view, const std::vector<float>& values) {
    CHECK(view.count == values.size());
    for (size_t i = 0; i < values.size(); ++i) CHECK_NEAR(view.data[i], values[i], 1e-5);
}
int main(int argc, char** argv) {
    CHECK(argc == 2);
    Stage2Spec spec; spec.model_path = std::string(argv[1]) + "/pixels2.onnx";
    spec.in_h = 2; spec.in_w = 2; spec.scale = .01f;
    spec.mean[0] = .1f; spec.mean[1] = .2f; spec.mean[2] = .3f;
    std::string err; auto a = make_cpu_stage2(spec, err); auto b = make_cpu_stage2(spec, err);
    CHECK(a); CHECK(b);
    int stride = 17;
    std::vector<uint8_t> pixels(stride * 4, 255);
    for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) {
        auto p = pixels.data() + y * stride + x * 3;
        p[0] = x * 10 + y * 40; p[1] = 100 + x * 5; p[2] = 200 - y * 10;
    }
    FrameBuf frame; frame.w = 4; frame.h = 4; frame.stride = stride;
    frame.mem = Mem::Host; frame.fmt = PixFmt::RGB888; frame.host = pixels.data();
    CropReq crop{.25f,.25f,.75f,.75f}; TensorView out;
    CHECK(a->infer_crops(frame, &crop, 1, &out, err) == 0);
    const std::vector<float> oracle{.4f,.5f,.8f,.9f, .85f,.9f,.85f,.9f, 1.6f,1.6f,1.5f,1.5f};
    expected(out, oracle);
    auto old = out.data;
    TensorView second; CHECK(b->infer_crops(frame, &crop, 1, &second, err) == 0);
    CHECK(old != second.data); expected(out, oracle);
    // Raw ROI buffer still uses full-source crop coordinates.
    std::vector<uint8_t> tile(16, 255);
    for (int y = 0; y < 2; ++y) std::copy_n(pixels.data() + (y+1)*stride + 3, 6, tile.data() + y*8);
    frame.w = frame.h = 2; frame.stride = 8; frame.full_w = frame.full_h = 4;
    frame.crop_x0 = frame.crop_y0 = 1; frame.host = tile.data();
    CHECK(a->infer_crops(frame, &crop, 1, &out, err) == 0); expected(out, oracle);
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) std::swap(tile[y*8+x*3], tile[y*8+x*3+2]);
    frame.fmt = PixFmt::BGR888;
    CHECK(a->infer_crops(frame, &crop, 1, &out, err) == 0); expected(out, oracle);
    frame.stride = 5; CHECK(a->infer_crops(frame, &crop, 1, &out, err) != 0);
    // Upscaling: negative half-pixel clamps to the first pixel, not a blend.
    spec.model_path = std::string(argv[1]) + "/pixels4.onnx"; spec.in_w = 4;
    spec.scale = 1; spec.mean[0] = spec.mean[1] = spec.mean[2] = 0;
    auto up = make_cpu_stage2(spec, err); CHECK(up);
    const uint8_t small[]{0,40,80, 100,140,180, 20,60,100, 120,160,200};
    CHECK(up->infer_rgb(small, 2, 2, 6, &out, err) == 0);
    expected(out, {0,25,75,100,20,45,95,120, 40,65,115,140,60,85,135,160, 80,105,155,180,100,125,175,200});
    // Letterboxed canvas: only rows 1..2 contain source pixels.
    std::vector<uint8_t> canvas(4*4*3, 250);
    for (int y = 1; y <= 2; ++y) for (int x = 0; x < 4; ++x) {
        auto p = canvas.data() + (y*4+x)*3; p[0] = x*10+y*40; p[1]=100+x*5; p[2]=200-y*10;
    }
    spec.model_path = std::string(argv[1]) + "/pixels2.onnx"; spec.in_w=2;
    auto boxed = make_cpu_stage2(spec, err); CHECK(boxed);
    frame = FrameBuf(); frame.w=frame.h=4; frame.stride=12; frame.mem=Mem::Host;
    frame.fmt=PixFmt::RGB888; frame.host=canvas.data(); frame.letterboxed=true;
    frame.geom.src_w=2; frame.geom.src_h=1; frame.geom.model_w=frame.geom.model_h=4;
    frame.geom.scale=2; frame.geom.pad_y=1;
    CropReq whole{0,0,1,1};
    CHECK(boxed->infer_crops(frame,&whole,1,&out,err)==0);
    expected(out,{45,65,85,105, 102.5,112.5,102.5,112.5, 190,190,180,180});
    for (auto name : {"badbatch", "badtype"}) {
        spec.model_path = std::string(argv[1]) + "/" + name + ".onnx";
        CHECK(!make_cpu_stage2(spec,err)); CHECK(!err.empty());
    }
    return 0;
}
