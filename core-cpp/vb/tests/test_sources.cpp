// Frame-source, wire-boundary and raw-tensor limits (BASE-1 review fixes).
//
//   test_sources <fixtures_dir>
//
// Covers: synthetic frame-size bounds (B3) incl. the overflow url; the
// GStreamer url policy — allowed schemes, and a raw launch string refused in
// production (C1); the snapshot crop intersection (B2); VBT1 protocol bounds
// (name/count/dims); and the raw-tensor dtype mapping used by the CPU
// backend's dev-mode passthrough (B5).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "dev_tensor.h"
#include "snapshot.h"
#include "sources/gst_source.h"
#include "sources/synthetic.h"
#include "vb/backend.h"
#include "vb/runtime.h"
#include "vb/wire.h"

// cpu_backend.h lives outside the vb_core include path; the helper it exposes
// is inline, so this test needs no link against the backend object library.
#include "../backends/cpu/cpu_backend.h"

namespace {

// ---- B3: synthetic frame sizes ----

void test_synthetic_size_bounds() {
    std::string err;
    auto backend = vb::make_synthetic_backend("{}", err);
    CHECK(backend != nullptr);
    auto open_src = [&](const char* url) {
        vb::StreamSpec s;
        s.url = url;
        err.clear();
        return backend->create_source(s, err);
    };

    // w * h * 3 used to be computed in int and to allocate whatever the url
    // asked for; both faces of the fix are checked here.
    CHECK(open_src("synthetic://?w=2147483647&h=2147483647") == nullptr);
    CHECK(!err.empty());
    err.clear();
    CHECK(open_src("synthetic://?w=99999999999&h=720") == nullptr);  // > int range
    CHECK(!err.empty());
    CHECK(open_src("synthetic://?w=4097&h=1") == nullptr);
    CHECK(open_src("synthetic://?w=1&h=4097") == nullptr);
    CHECK(open_src("synthetic://?w=100000&h=100000") == nullptr);

    // The bound itself is still usable, and ordinary urls are untouched.
    CHECK(open_src("synthetic://?w=4096&h=1") != nullptr);
    auto ok = open_src("synthetic://?w=320&h=240&fps=60&boxes=2");
    CHECK(ok != nullptr);
    if (ok) {
        CHECK(ok->open(err));
        vb::FrameBuf f;
        CHECK(ok->read(f, 1000) == 1);
        CHECK(f.w == 320 && f.h == 240 && f.stride == 320 * 3);
        CHECK(f.host != nullptr);
        ok->close();
    }

    CHECK(vb::synthetic_dims_ok(640, 480, err));
    CHECK(!vb::synthetic_dims_ok(0, 480, err));
    CHECK(!vb::synthetic_dims_ok(-1, 480, err));
    CHECK(!vb::synthetic_dims_ok(4096, 4097, err));
    std::printf("synthetic: oversized frames refused, bounded sizes accepted\n");
}

// ---- C1: GStreamer url policy ----

void test_source_url_policy() {
    ::unsetenv("VB_PRODUCTION");
    std::string err;

    CHECK(vb::classify_source_url("rtsp://cam.local:554/s1", err) == vb::SourceUrlKind::Uri);
    CHECK(err.empty());
    CHECK(vb::classify_source_url("rtsps://cam.local/s1", err) == vb::SourceUrlKind::Uri);
    CHECK(vb::classify_source_url("http://cam.local/s.mjpg", err) == vb::SourceUrlKind::Uri);
    CHECK(vb::classify_source_url("https://cam.local/s", err) == vb::SourceUrlKind::Uri);
    CHECK(vb::classify_source_url("file:///tmp/clip.mp4", err) == vb::SourceUrlKind::Uri);
    CHECK(vb::classify_source_url("RTSP://cam.local/s", err) == vb::SourceUrlKind::Uri);
    // An element the pipeline grammar should never see: only reachable in dev.
    CHECK(vb::classify_source_url("videotestsrc num-buffers=3", err) ==
          vb::SourceUrlKind::DevPipeline);
    CHECK(vb::classify_source_url("fakesrc ! fakesink", err) ==
          vb::SourceUrlKind::DevPipeline);
    CHECK(vb::classify_source_url("uridecodebin uri=rtsp://cam/s ! fakesink", err) ==
          vb::SourceUrlKind::DevPipeline);

    // Unknown or malformed schemes are refused in every mode.
    CHECK(vb::classify_source_url("evil://cam/s", err) == vb::SourceUrlKind::Rejected);
    CHECK(!err.empty());
    CHECK(vb::classify_source_url("synthetic://0", err) == vb::SourceUrlKind::Rejected);
    CHECK(vb::classify_source_url("", err) == vb::SourceUrlKind::Rejected);

    // §6.12 production gate: the same VB_PRODUCTION switch that refuses
    // dev.raw_tensors refuses raw pipelines, while urls keep working.
    ::setenv("VB_PRODUCTION", "1", 1);
    err.clear();
    CHECK(vb::classify_source_url("videotestsrc num-buffers=3", err) ==
          vb::SourceUrlKind::Rejected);
    CHECK(!err.empty());
    CHECK(vb::classify_source_url("rtsp://cam.local/s1", err) == vb::SourceUrlKind::Uri);
    CHECK(vb::classify_source_url("file:///tmp/clip.mp4", err) == vb::SourceUrlKind::Uri);
    ::unsetenv("VB_PRODUCTION");

    // Control characters cannot be smuggled through either path.
    CHECK(vb::classify_source_url(std::string("rtsp://cam/") + '\x01' + "s", err) ==
          vb::SourceUrlKind::Rejected);
    CHECK(vb::classify_source_url(std::string("fakesrc\nfakesink"), err) ==
          vb::SourceUrlKind::Rejected);
    std::printf("gst url: schemes allow-listed, launch strings gated on dev mode\n");
}

// ---- B2: snapshot crop intersection ----

void test_snapshot_clip() {
    int x0 = 0, y0 = 0, cw = 0, ch = 0;

    // A box starting at the bottom/right edge used to clip to y0 == h (x0 == w)
    // and then force a 1-pixel extent: a memcpy from outside the frame.
    CHECK(!vb::snapshot_clip_box(10, 100, 40, 130, 320, 100, x0, y0, cw, ch));
    CHECK(!vb::snapshot_clip_box(10, 110, 40, 130, 320, 100, x0, y0, cw, ch));
    CHECK(!vb::snapshot_clip_box(320, 10, 340, 40, 320, 100, x0, y0, cw, ch));
    CHECK(!vb::snapshot_clip_box(340, 10, 360, 40, 320, 100, x0, y0, cw, ch));
    CHECK(!vb::snapshot_clip_box(0, 0, 0, 0, 320, 100, x0, y0, cw, ch));
    CHECK(!vb::snapshot_clip_box(50, 50, 40, 60, 320, 100, x0, y0, cw, ch));
    CHECK(!vb::snapshot_clip_box(std::nan(""), 0, 10, 10, 320, 100, x0, y0, cw, ch));
    CHECK(!vb::snapshot_clip_box(0, 0, 10, 10, 0, 0, x0, y0, cw, ch));

    // A partial overlap is clipped to the frame, never refused or extended.
    CHECK(vb::snapshot_clip_box(-10, -10, 20.5f, 30.5f, 320, 100, x0, y0, cw, ch));
    CHECK(x0 == 0 && y0 == 0 && cw == 21 && ch == 31);
    CHECK(vb::snapshot_clip_box(300, 80, 400, 200, 320, 100, x0, y0, cw, ch));
    CHECK(x0 == 300 && y0 == 80 && cw == 20 && ch == 20);
    CHECK(vb::snapshot_clip_box(10, 20, 30, 40, 320, 100, x0, y0, cw, ch));
    CHECK(x0 == 10 && y0 == 20 && cw == 20 && ch == 20);
    CHECK(x0 + cw <= 320 && y0 + ch <= 100);

    // Encode level: with a JPEG encoder in the build the same box must fail the
    // whole snapshot rather than encode garbage. Without one the request is
    // refused as unsupported, which is the same outcome for a different reason
    // (the pure clipping check above covers the bound either way).
    vb::SnapshotRingEntry e;
    e.seq = 1;
    e.w = 64;
    e.h = 48;
    e.pixels.assign(static_cast<size_t>(e.w) * e.h * 3, 0);
    vb::SnapBox outside{};
    outside.track_id = 7;
    outside.x0 = 100;
    outside.y0 = 100;
    outside.x1 = 140;
    outside.y1 = 130;
    e.boxes.push_back(outside);
    std::vector<uint8_t> jpeg;
    int jw = 0, jh = 0;
    std::string err;
    if (vb::snapshot_encode_jpeg(e, 7, true, 640, jpeg, jw, jh, err)) {
        std::fprintf(stderr, "crop outside the frame must not encode\n");
        std::exit(1);
    }
    CHECK(!err.empty());
    if (err == "snapshot unsupported") {
        std::printf("snapshot: JPEG encoder not built, encode-level check skipped\n");
    } else {
        // A box inside the frame still encodes, so the refusal above is the
        // crop bound and not a broken encoder.
        e.boxes[0].x0 = 8;
        e.boxes[0].y0 = 8;
        e.boxes[0].x1 = 40;
        e.boxes[0].y1 = 32;
        jpeg.clear();
        std::string err2;
        CHECK(vb::snapshot_encode_jpeg(e, 7, true, 640, jpeg, jw, jh, err2));
        CHECK(jw == 32 && jh == 24);
        CHECK(!jpeg.empty());
        std::printf("snapshot: crop bounds enforced (%s)\n", err.c_str());
    }
    std::printf("snapshot: empty crop intersections refused\n");
}

// ---- VBT1 protocol bounds (§6.12) ----

void test_vbt1_bounds() {
    std::string err;
    std::vector<uint8_t> enc;

    // A name longer than the u16 length field would be declared truncated
    // while the full bytes are still written.
    {
        vb::DevTensorFrame f;
        vb::DevTensor t;
        t.name.assign(0x10000, 'n');
        t.dims = {1};
        t.data.assign(4, 0);
        f.tensors.push_back(std::move(t));
        enc.clear();
        CHECK(!vb::wire_encode_vbt1(f, enc, err));
        CHECK(!err.empty());
        CHECK(enc.empty());
    }
    // More than 65535 tensors cannot be declared in n_tensors.
    {
        vb::DevTensorFrame f;
        f.tensors.resize(0x10000);
        enc.clear();
        CHECK(!vb::wire_encode_vbt1(f, enc, err));
        CHECK(err.find("too many tensors") != std::string::npos);
        CHECK(enc.empty());
    }
    // VBT1 carries at most 4 dims per tensor.
    {
        vb::DevTensorFrame f;
        vb::DevTensor t;
        t.dims = {1, 2, 3, 4, 5};
        f.tensors.push_back(std::move(t));
        enc.clear();
        CHECK(!vb::wire_encode_vbt1(f, enc, err));
        CHECK(enc.empty());
    }
    // The boundaries themselves still encode.
    {
        vb::DevTensorFrame f;
        vb::DevTensor t;
        t.name.assign(0xFFFF, 'n');
        t.dims = {1, 2, 3, 4};
        t.data.assign(8, 0);
        f.tensors.push_back(std::move(t));
        enc.clear();
        CHECK(vb::wire_encode_vbt1(f, enc, err));
        CHECK(err.empty());
        // 8 header + 56 fixed body + 28 per-tensor fixed (dtype/n_dims/nhwc/
        // reserved, 4 dims, scale, zero_point) + name_len + name + data_len +
        // data.
        CHECK(enc.size() == 8 + 56 + 28 + 2 + 0xFFFF + 4 + 8);
    }
    std::printf("vbt1: names, counts and dims bounded before encoding\n");
}

// ---- B5: raw tensor dtype mapping ----

void test_cpu_raw_dtype() {
    // ONNX_TENSOR_ELEMENT_DATA_TYPE_* values (onnxruntime_c_api.h).
    const int32_t kFloat = 1, kUint8 = 2, kInt8 = 3, kInt32 = 6, kInt64 = 7,
                  kFloat16 = 10, kDouble = 11;
    uint8_t dtype = 0x5a;
    size_t elem_bytes = 0;
    std::string err;

    CHECK(vb::cpu_vbt1_dtype(kFloat, dtype, elem_bytes, err));
    CHECK(err.empty());
    CHECK(dtype == 0);  // VBT1 dtype 0 = f32
    CHECK(elem_bytes == 4);

    for (int32_t t : {kUint8, kInt8, kInt32, kInt64, kFloat16, kDouble}) {
        err.clear();
        CHECK(!vb::cpu_vbt1_dtype(t, dtype, elem_bytes, err));
        CHECK(!err.empty());
    }
    std::printf("cpu raw: non-float32 tensor types refused, f32 width is 4\n");
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    test_synthetic_size_bounds();
    test_source_url_policy();
    test_snapshot_clip();
    test_vbt1_bounds();
    test_cpu_raw_dtype();
    std::printf("sources: all checks passed\n");
    return 0;
}
