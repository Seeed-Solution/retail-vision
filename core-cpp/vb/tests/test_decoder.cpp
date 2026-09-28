// M1.15 decoder tests: make_decoder error cases, yolov8 transposed-vs-rows
// layout parity, classify softmax/top_k, yolo_pose (both layouts), plus the
// §6.11 decode fixtures and the review-fix cases:
//   D2 input buffer / null pointer validation
//   D3 yolov8_dfl box-vs-cls role ambiguity (4*reg_max == num_classes)
//   D4 non-finite candidates must not reach the detections
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "check.h"
#include "vb/decoder.h"
#include "vb/json.h"

using namespace vb;

namespace {

bool all_finite(const DetectionResult& r) {
    for (const Detection& d : r.dets)
        if (!std::isfinite(d.cx) || !std::isfinite(d.cy) || !std::isfinite(d.w) ||
            !std::isfinite(d.h) || !std::isfinite(d.score))
            return false;
    for (const Keypoint& k : r.kpts)
        if (!std::isfinite(k.x) || !std::isfinite(k.y) || !std::isfinite(k.conf))
            return false;
    return true;
}

// Row-major [C, H, W] float tensor for the split-head decoders.
struct Plane3 {
    std::vector<float> d;
    int64_t C, H, W;

    Plane3(int64_t c, int64_t h, int64_t w)
        : d(static_cast<size_t>(c * h * w), 0.0f), C(c), H(h), W(w) {}
    float& at(int64_t c, int64_t y, int64_t x) {
        return d[static_cast<size_t>((c * H + y) * W + x)];
    }
    TensorView view(const char* name = "") {
        return TensorView{d.data(), d.size(), {C, H, W}, name};
    }
};

// Small fixture runner (same format as `vb_selftest decode`, §6.11).
bool run_fixture(const std::string& path) {
    auto j = json_parse(read_file(path));
    std::string err;
    auto dec = make_decoder(j.at("decoder").dump(), err);
    if (!dec) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    const int mw = j.at("model_hw")[0].get<int>();
    const int mh = j.at("model_hw")[1].get<int>();
    const float score = j.value("score", 0.35f);
    const float nms = j.value("nms", 0.45f);
    std::vector<TensorView> views;
    std::vector<std::vector<float>> storage;
    for (const auto& t : j.at("tensors")) {
        storage.emplace_back();
        for (const auto& v : t.at("data")) storage.back().push_back(v.get<float>());
        TensorView tv;
        tv.data = storage.back().data();
        tv.count = storage.back().size();
        for (const auto& d : t.at("dims")) tv.dims.push_back(d.get<int64_t>());
        if (t.contains("name")) tv.name = t.at("name").get<std::string>();
        views.push_back(std::move(tv));
    }
    DetectionResult out;
    if (!dec->decode(views.data(), views.size(), mw, mh, score, nms, out, err)) {
        std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    const auto& expect = j.at("expect");
    if (out.dets.size() != expect.size()) {
        std::fprintf(stderr, "%s: expected %zu dets, got %zu\n", path.c_str(),
                     expect.size(), out.dets.size());
        return false;
    }
    for (size_t i = 0; i < expect.size(); ++i) {
        const auto& e = expect[i];
        const Detection& d = out.dets[i];
        if (std::fabs(d.cx - e.at("cx").get<double>()) > 1e-5 ||
            std::fabs(d.cy - e.at("cy").get<double>()) > 1e-5 ||
            std::fabs(d.w - e.at("w").get<double>()) > 1e-5 ||
            std::fabs(d.h - e.at("h").get<double>()) > 1e-5 ||
            std::fabs(d.score - e.at("score").get<double>()) > 1e-5 ||
            d.class_id != e.at("class_id").get<int>()) {
            std::fprintf(stderr, "%s: det %zu mismatch\n", path.c_str(), i);
            return false;
        }
        if (e.contains("kpts")) {  // pose fixtures carry the keypoint slice
            const auto& ek = e.at("kpts");
            if (d.kpt_count != ek.size() ||
                d.kpt_offset + d.kpt_count > out.kpts.size()) {
                std::fprintf(stderr, "%s: det %zu kpt slice mismatch\n", path.c_str(), i);
                return false;
            }
            for (size_t k = 0; k < ek.size(); ++k) {
                const Keypoint& kp = out.kpts[d.kpt_offset + k];
                if (std::fabs(kp.x - ek[k][0].get<double>()) > 1e-5 ||
                    std::fabs(kp.y - ek[k][1].get<double>()) > 1e-5 ||
                    std::fabs(kp.conf - ek[k][2].get<double>()) > 1e-5) {
                    std::fprintf(stderr, "%s: det %zu kpt %zu mismatch\n", path.c_str(),
                                 i, k);
                    return false;
                }
            }
        }
    }
    if (!all_finite(out)) {
        std::fprintf(stderr, "%s: non-finite value in result\n", path.c_str());
        return false;
    }
    std::printf("  %s: OK\n", path.c_str());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_decoder <fixtures-dir>\n");
        return 2;
    }
    const std::string dir = argv[1];

    // error cases
    {
        std::string err;
        CHECK(make_decoder("", err) == nullptr && err.empty());  // adapter default
        CHECK(make_decoder("null", err) == nullptr && err.empty());
        CHECK(make_decoder(R"({"type":"ctc"})", err) == nullptr);
        CHECK(err.find("ctc is only valid in stage2") != std::string::npos);
        err.clear();
        CHECK(make_decoder(R"({"type":"nope"})", err) == nullptr);
        CHECK(err.find("unknown type") != std::string::npos);
        err.clear();
        CHECK(make_decoder("not json", err) == nullptr);
        // yolo_pose is a registered decoder (A3) and validates `keypoints`.
        err.clear();
        auto pose = make_decoder(R"({"type":"yolo_pose","keypoints":2})", err);
        CHECK(pose != nullptr);
        CHECK(pose->keypoints() == 2);
        auto pose_default = make_decoder(R"({"type":"yolo_pose"})", err);
        CHECK(pose_default != nullptr);
        CHECK(pose_default->keypoints() == 17);  // §6.11 DecodeSpec default
        err.clear();
        CHECK(make_decoder(R"({"type":"yolo_pose","keypoints":0})", err) == nullptr);
        CHECK(err.find("keypoints") != std::string::npos);
    }

    // yolov8: [4+nc, N] and [N, 4+nc] layouts decode identically
    {
        // nc=1, 3 anchors: box in pixels on a 100x100 canvas
        const int nc = 1, N = 3;
        // rows: cx, cy, w, h, cls0
        const float rows[N][5] = {
            {50, 50, 20, 40, 0.9}, {20, 80, 10, 10, 0.8}, {80, 20, 10, 10, 0.2}};
        std::vector<float> chw, nhw;
        for (int c = 0; c < 4 + nc; ++c)
            for (int i = 0; i < N; ++i) {
                chw.push_back(rows[i][c]);
                nhw.push_back(rows[i][c]);  // same order, dims differ below
            }
        // nhw must be row-major [N, 5]: rebuild properly
        nhw.clear();
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < 4 + nc; ++c) nhw.push_back(rows[i][c]);

        std::string err;
        auto dec = make_decoder(R"({"type":"yolov8","num_classes":1})", err);
        CHECK(dec != nullptr);
        TensorView tv_chw{chw.data(), chw.size(), {4 + nc, N}, ""};
        TensorView tv_nhw{nhw.data(), nhw.size(), {N, 4 + nc}, ""};
        DetectionResult a, b;
        CHECK(dec->decode(&tv_chw, 1, 100, 100, 0.5f, 0.45f, a, err));
        CHECK(dec->decode(&tv_nhw, 1, 100, 100, 0.5f, 0.45f, b, err));
        CHECK(a.dets.size() == 2 && b.dets.size() == 2);
        for (size_t i = 0; i < a.dets.size(); ++i) {
            CHECK_NEAR(a.dets[i].cx, b.dets[i].cx, 1e-6);
            CHECK_NEAR(a.dets[i].cy, b.dets[i].cy, 1e-6);
            CHECK_NEAR(a.dets[i].w, b.dets[i].w, 1e-6);
            CHECK_NEAR(a.dets[i].h, b.dets[i].h, 1e-6);
            CHECK_NEAR(a.dets[i].score, b.dets[i].score, 1e-6);
        }
        CHECK_NEAR(a.dets[0].cx, 0.5, 1e-6);
        CHECK_NEAR(a.dets[0].h, 0.4, 1e-6);
        CHECK_NEAR(a.dets[0].score, 0.9, 1e-6);
        CHECK(a.dets[0].class_id == 0);
    }

    // classify: softmax + top_k ordering, whole-frame boxes
    {
        std::string err;
        auto dec = make_decoder(R"({"type":"classify","num_classes":5,"top_k":2})", err);
        CHECK(dec != nullptr);
        const float logits[5] = {0.0f, 3.0f, 1.0f, 2.0f, -1.0f};  // 1 > 3 > 2 > 0 > 4
        TensorView tv{logits, 5, {5}, ""};
        DetectionResult out;
        CHECK(dec->decode(&tv, 1, 224, 224, 0.01f, 0.45f, out, err));
        CHECK(out.dets.size() == 2);
        CHECK(out.dets[0].class_id == 1 && out.dets[1].class_id == 3);
        // softmax(0,3,1,2,-1): p1 = 1/1.571317 ~ 0.636409
        CHECK(out.dets[0].score > out.dets[1].score);
        CHECK_NEAR(out.dets[0].score, 0.636409, 1e-4);
        CHECK_NEAR(out.dets[0].cx, 0.5, 1e-6);
        CHECK_NEAR(out.dets[0].w, 1.0, 1e-6);
        // softmax=false keeps raw logits
        auto dec2 = make_decoder(R"({"type":"classify","num_classes":5,"top_k":1,"softmax":false})", err);
        CHECK(dec2 != nullptr);
        DetectionResult out2;
        CHECK(dec2->decode(&tv, 1, 224, 224, 0.01f, 0.45f, out2, err));
        CHECK(out2.dets.size() == 1);
        CHECK_NEAR(out2.dets[0].score, 3.0, 1e-6);
    }

    // raw: no detections, keypoints 0
    {
        std::string err;
        auto dec = make_decoder(R"({"type":"raw"})", err);
        CHECK(dec != nullptr);
        CHECK(dec->keypoints() == 0);
        const float whatever[4] = {0};
        TensorView tv{whatever, 4, {2, 2}, ""};
        DetectionResult out;
        CHECK(dec->decode(&tv, 1, 64, 64, 0.3f, 0.45f, out, err));
        CHECK(out.dets.empty());
    }

    // D2: every decoder validates data pointer, dims and element count before
    // reading. Count 1 against dims [5,3] used to read 15 floats out of bounds.
    {
        std::string err;
        float box[15] = {0};
        DetectionResult out;

        auto y8 = make_decoder(R"({"type":"yolov8","num_classes":1})", err);
        CHECK(y8 != nullptr);
        TensorView short_view{box, 1, {5, 3}, ""};
        CHECK(!y8->decode(&short_view, 1, 100, 100, 0.5f, 0.45f, out, err));
        CHECK(err.find("elements") != std::string::npos);
        err.clear();
        TensorView null_view{nullptr, 15, {5, 3}, ""};
        CHECK(!y8->decode(&null_view, 1, 100, 100, 0.5f, 0.45f, out, err));
        CHECK(err.find("null") != std::string::npos);
        err.clear();
        TensorView zero_dim{box, 15, {0, 3}, ""};
        CHECK(!y8->decode(&zero_dim, 1, 100, 100, 0.5f, 0.45f, out, err));
        CHECK(!err.empty());
        err.clear();
        TensorView no_dims{box, 15, {}, ""};
        CHECK(!y8->decode(&no_dims, 1, 100, 100, 0.5f, 0.45f, out, err));
        CHECK(!err.empty());
        err.clear();
        const int64_t huge = 1LL << 40;  // dims product overflows size_t
        TensorView overflow{box, 15, {huge, huge}, ""};
        CHECK(!y8->decode(&overflow, 1, 100, 100, 0.5f, 0.45f, out, err));
        CHECK(err.find("overflow") != std::string::npos);
        err.clear();
        TensorView ok_view{box, 15, {5, 3}, ""};
        CHECK(!y8->decode(&ok_view, 1, 0, 100, 0.5f, 0.45f, out, err));
        CHECK(err.find("canvas") != std::string::npos);
        err.clear();
        CHECK(!y8->decode(nullptr, 1, 100, 100, 0.5f, 0.45f, out, err));
        CHECK(!err.empty());

        auto cls = make_decoder(R"({"type":"classify","num_classes":5,"top_k":1})", err);
        CHECK(cls != nullptr);
        err.clear();
        TensorView cls_short{box, 3, {5}, ""};
        CHECK(!cls->decode(&cls_short, 1, 224, 224, 0.01f, 0.45f, out, err));
        CHECK(!err.empty());
        err.clear();
        TensorView cls_null{nullptr, 5, {5}, ""};
        CHECK(!cls->decode(&cls_null, 1, 224, 224, 0.01f, 0.45f, out, err));
        CHECK(!err.empty());

        auto yx = make_decoder(R"({"type":"yolox","num_classes":1})", err);
        CHECK(yx != nullptr);
        err.clear();
        TensorView yx_short{box, 6, {2, 6}, ""};
        CHECK(!yx->decode(&yx_short, 1, 100, 100, 0.5f, 0.45f, out, err));
        CHECK(!err.empty());
        err.clear();
        TensorView yx_ok{box, 12, {2, 6}, ""};
        CHECK(yx->decode(&yx_ok, 1, 100, 100, 0.5f, 0.45f, out, err));

        auto dfl = make_decoder(
            R"({"type":"yolov8_dfl","num_classes":1,"reg_max":4,"strides":[8]})", err);
        CHECK(dfl != nullptr);
        err.clear();
        Plane3 dfl_plane(17, 2, 2);
        TensorView dfl_short{dfl_plane.d.data(), dfl_plane.d.size() - 1,
                             {17, 2, 2}, ""};
        CHECK(!dfl->decode(&dfl_short, 1, 16, 16, 0.3f, 0.45f, out, err));
        CHECK(err.find("elements") != std::string::npos);

        auto pose = make_decoder(
            R"({"type":"yolo_pose","keypoints":2,"reg_max":4,"num_classes":1,"strides":[8]})",
            err);
        CHECK(pose != nullptr);
        err.clear();
        Plane3 pose_plane(16, 2, 2);
        TensorView pose_short{pose_plane.d.data(), pose_plane.d.size() - 1, {16, 2, 2}, ""};
        CHECK(!pose->decode(&pose_short, 1, 16, 16, 0.3f, 0.45f, out, err));
        CHECK(err.find("elements") != std::string::npos);
    }

    // D3: reg_max=16 with num_classes=64 makes box and cls tensors the same
    // shape. Roles are resolved by output name, then by the documented order
    // (box first), and a duplicate claim fails instead of overwriting.
    {
        std::string err;
        auto dec = make_decoder(
            R"({"type":"yolov8_dfl","num_classes":64,"reg_max":16,"strides":[8]})", err);
        CHECK(dec != nullptr);
        const float kLog4 = 1.3862943611198906f;
        auto make_box = [] {
            Plane3 p(64, 2, 2);
            for (int side = 0; side < 4; ++side) p.at(side * 16 + 1, 1, 1) = 20.0f;
            return p;
        };
        auto make_cls = [&] {
            Plane3 p(64, 2, 2);
            for (int64_t c = 0; c < 64; ++c)
                for (int64_t y = 0; y < 2; ++y)
                    for (int64_t x = 0; x < 2; ++x) p.at(c, y, x) = -10.0f;
            p.at(5, 1, 1) = kLog4;  // sigmoid -> 0.8, class 5
            return p;
        };

        // Documented order: box first, then cls. dist = 1 bin = 8 px ->
        // x0 = (1.5-1)*8 = 4, x1 = (1.5+1)*8 = 20 -> cx = cy = 0.75, w = h = 1
        {
            Plane3 b = make_box(), c = make_cls();
            TensorView views[2] = {b.view(), c.view()};
            DetectionResult out;
            CHECK(dec->decode(views, 2, 16, 16, 0.3f, 0.45f, out, err));
            CHECK(out.dets.size() == 1);
            CHECK_NEAR(out.dets[0].cx, 0.75, 1e-5);
            CHECK_NEAR(out.dets[0].cy, 0.75, 1e-5);
            CHECK_NEAR(out.dets[0].w, 1.0, 1e-5);
            CHECK_NEAR(out.dets[0].h, 1.0, 1e-5);
            CHECK_NEAR(out.dets[0].score, 0.8, 1e-5);
            CHECK(out.dets[0].class_id == 5);
        }
        // Same pair in the opposite order, disambiguated by output name.
        {
            Plane3 b = make_box(), c = make_cls();
            TensorView views[2] = {c.view("cls_64"), b.view("reg_64")};
            DetectionResult out;
            CHECK(dec->decode(views, 2, 16, 16, 0.3f, 0.45f, out, err));
            CHECK(out.dets.size() == 1);
            CHECK_NEAR(out.dets[0].score, 0.8, 1e-5);
            CHECK(out.dets[0].class_id == 5);
            CHECK_NEAR(out.dets[0].cx, 0.75, 1e-5);
        }
        // A third same-shaped tensor cannot claim an occupied role.
        {
            Plane3 b = make_box(), c = make_cls(), extra = make_box();
            TensorView views[3] = {b.view(), c.view(), extra.view()};
            DetectionResult out;
            err.clear();
            CHECK(!dec->decode(views, 3, 16, 16, 0.3f, 0.45f, out, err));
            CHECK(err.find("two cls tensors") != std::string::npos);
        }
    }

    // D4: non-finite candidates are dropped, never emitted.
    {
        std::string err;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float inf = std::numeric_limits<float>::infinity();

        // yolov8: one NaN score row among finite ones
        {
            auto dec = make_decoder(R"({"type":"yolov8","num_classes":1})", err);
            CHECK(dec != nullptr);
            const float rows[3][5] = {{50, 50, 20, 40, 0.9f},
                                      {20, 80, 10, 10, nan},
                                      {80, 20, 10, 10, 0.8f}};
            std::vector<float> chw;
            for (int c = 0; c < 5; ++c)
                for (int i = 0; i < 3; ++i) chw.push_back(rows[i][c]);
            TensorView tv{chw.data(), chw.size(), {5, 3}, ""};
            DetectionResult out;
            err.clear();
            CHECK(dec->decode(&tv, 1, 100, 100, 0.5f, 0.45f, out, err));
            CHECK(out.dets.size() == 2);
            CHECK(all_finite(out));
            CHECK(err.find("non-finite") != std::string::npos);

            // NaN box coordinates are dropped the same way
            const float rows2[1][5] = {{nan, 50, 20, 40, 0.9f}};
            std::vector<float> t2;
            for (int c = 0; c < 5; ++c) t2.push_back(rows2[0][c]);
            TensorView tv2{t2.data(), t2.size(), {5, 1}, ""};
            err.clear();
            CHECK(dec->decode(&tv2, 1, 100, 100, 0.5f, 0.45f, out, err));
            CHECK(out.dets.empty());
            CHECK(err.find("non-finite") != std::string::npos);
        }

        // yolov8_dfl: NaN bins, all-+inf bins and a NaN class logit all drop
        // their cell; the finite cell still decodes.
        {
            auto dec = make_decoder(
                R"({"type":"yolov8_dfl","num_classes":1,"reg_max":4,"strides":[8]})", err);
            CHECK(dec != nullptr);
            Plane3 p(17, 2, 2);
            // (0,0): every bin NaN
            for (int d = 0; d < 4; ++d)
                for (int k = 0; k < 4; ++k) p.at(d * 4 + k, 0, 0) = nan;
            // (0,1): every bin +inf
            for (int d = 0; d < 4; ++d)
                for (int k = 0; k < 4; ++k) p.at(d * 4 + k, 0, 1) = inf;
            // (1,1): one-hot bin 1 -> dist 1 cell = 8 px, cls ln4 -> 0.8
            for (int d = 0; d < 4; ++d) p.at(d * 4 + 1, 1, 1) = 20.0f;
            p.at(16, 1, 1) = 1.3862943611198906f;
            // (1,0): finite bins, NaN class logit
            for (int d = 0; d < 4; ++d) p.at(d * 4 + 1, 1, 0) = 20.0f;
            p.at(16, 1, 0) = nan;
            TensorView tv = p.view();
            DetectionResult out;
            err.clear();
            CHECK(dec->decode(&tv, 1, 16, 16, 0.3f, 0.45f, out, err));
            CHECK(out.dets.size() == 1);
            CHECK_NEAR(out.dets[0].cx, 0.75, 1e-5);
            CHECK_NEAR(out.dets[0].score, 0.8, 1e-5);
            CHECK(all_finite(out));
            CHECK(err.find("dropped 3 non-finite") != std::string::npos);
        }

        // classify: a NaN logit is dropped, the finite classes still come out
        {
            auto dec = make_decoder(
                R"({"type":"classify","num_classes":3,"top_k":3,"softmax":false})", err);
            CHECK(dec != nullptr);
            const float logits[3] = {nan, 1.0f, 2.0f};
            TensorView tv{logits, 3, {3}, ""};
            DetectionResult out;
            err.clear();
            CHECK(dec->decode(&tv, 1, 224, 224, 0.01f, 0.45f, out, err));
            CHECK(out.dets.size() == 2);
            CHECK(out.dets[0].class_id == 2 && out.dets[1].class_id == 1);
            CHECK(all_finite(out));
            CHECK(err.find("non-finite") != std::string::npos);
        }
    }

    // A3: yolo_pose, split DFL pose head (box/cls/kpt per stride). Hand math
    // on a 16x16 canvas (stride 8, reg_max 4, 2 keypoints): cell (1,1) has a
    // one-hot bin 1 on every side -> dist 1 cell = 8 px, so x0 = y0 = 4 and
    // x1 = y1 = 20 (cx = cy = 0.75, w = h = 1.0); cls logit ln4 -> 0.8;
    // keypoints x = (2*px + gx - 0.5)*stride.
    {
        std::string err;
        auto dec = make_decoder(
            R"({"type":"yolo_pose","keypoints":2,"reg_max":4,"num_classes":1,"strides":[8]})",
            err);
        CHECK(dec != nullptr);
        CHECK(dec->keypoints() == 2);
        Plane3 box(16, 2, 2), cls(1, 2, 2), kpt(6, 2, 2);
        // Scores: cell (0,0) also clears the threshold with a distant box.
        for (int64_t y = 0; y < 2; ++y)
            for (int64_t x = 0; x < 2; ++x) cls.at(0, y, x) = -10.0f;
        cls.at(0, 1, 1) = 1.3862943611198906f;  // -> 0.8
        cls.at(0, 0, 0) = 0.0f;                 // -> 0.5
        for (int d = 0; d < 4; ++d) box.at(d * 4 + 1, 1, 1) = 20.0f;
        for (int d = 0; d < 4; ++d) box.at(d * 4 + 1, 0, 0) = 20.0f;
        kpt.at(0, 1, 1) = 0.5f;   // k0.x = (1 + 1 - 0.5)*8 = 12 px -> 0.75
        kpt.at(1, 1, 1) = 0.25f;  // k0.y = (0.5 + 1 - 0.5)*8 = 8 px -> 0.5
        kpt.at(2, 1, 1) = 0.0f;   // sigmoid(0) = 0.5
        kpt.at(3, 1, 1) = 0.25f;  // k1.x = 8 px -> 0.5
        kpt.at(4, 1, 1) = 0.75f;  // k1.y = 16 px -> 1.0
        kpt.at(5, 1, 1) = 1.3862943611198906f;  // sigmoid -> 0.8
        for (int64_t c = 0; c < 6; ++c) kpt.at(c, 0, 0) = 0.1f;

        TensorView views[3] = {box.view(), cls.view(), kpt.view()};
        DetectionResult out;
        CHECK(dec->decode(views, 3, 16, 16, 0.3f, 0.45f, out, err));
        CHECK(out.dets.size() == 2);  // both cells clear 0.3, IoU 0
        CHECK(out.kpts.size() == 4);  // 2 dets x 2 keypoints, compacted by NMS
        const Detection& hi = out.dets[0];
        CHECK_NEAR(hi.score, 0.8, 1e-5);
        CHECK_NEAR(hi.cx, 0.75, 1e-5);
        CHECK_NEAR(hi.cy, 0.75, 1e-5);
        CHECK_NEAR(hi.w, 1.0, 1e-5);
        CHECK_NEAR(hi.h, 1.0, 1e-5);
        CHECK(hi.class_id == 0);
        CHECK(hi.kpt_count == 2);
        // Ultralytics keypoint form is (2*k + g)*stride — the same cell origin
        // the box path uses (gx + 0.5). Written with g - 0.5 these were one half
        // cell short (4 px at stride 8): 0.75/0.5 and 0.5/1.0 before the fix.
        CHECK_NEAR(out.kpts[hi.kpt_offset + 0].x, 1.0, 1e-5);
        CHECK_NEAR(out.kpts[hi.kpt_offset + 0].y, 0.75, 1e-5);
        CHECK_NEAR(out.kpts[hi.kpt_offset + 0].conf, 0.5, 1e-5);
        CHECK_NEAR(out.kpts[hi.kpt_offset + 1].x, 0.75, 1e-5);
        CHECK_NEAR(out.kpts[hi.kpt_offset + 1].y, 1.25, 1e-5);
        CHECK_NEAR(out.kpts[hi.kpt_offset + 1].conf, 0.8, 1e-5);
        CHECK(out.dets[1].kpt_count == 2);
        CHECK(out.dets[1].kpt_offset == 2);
        CHECK(all_finite(out));

        // Non-finite keypoints drop the whole candidate (and its slice).
        {
            Plane3 kbad = kpt;
            kbad.at(2, 1, 1) = std::numeric_limits<float>::quiet_NaN();
            TensorView bad[3] = {box.view(), cls.view(), kbad.view()};
            DetectionResult o2;
            err.clear();
            CHECK(dec->decode(bad, 3, 16, 16, 0.3f, 0.45f, o2, err));
            CHECK(o2.dets.size() == 1);
            CHECK(o2.dets[0].kpt_count == 2);
            CHECK(o2.kpts.size() == 2);
            CHECK(all_finite(o2));
            CHECK(err.find("non-finite") != std::string::npos);
        }

        // A second tensor of each role is rejected, and an incomplete set too.
        {
            TensorView dup[4] = {box.view(), cls.view(), kpt.view(), kpt.view()};
            DetectionResult o3;
            err.clear();
            CHECK(!dec->decode(dup, 4, 16, 16, 0.3f, 0.45f, o3, err));
            CHECK(err.find("two keypoint tensors") != std::string::npos);
            TensorView missing[2] = {box.view(), cls.view()};
            err.clear();
            CHECK(!dec->decode(missing, 2, 16, 16, 0.3f, 0.45f, o3, err));
            CHECK(err.find("incomplete") != std::string::npos);
        }
    }

    // A3: yolo_pose, end-to-end export [5+3K, N] / [N, 5+3K]. Boxes are
    // model-input pixels here; a normalized export (|v| <= 2) must decode to
    // the same detections.
    {
        std::string err;
        auto dec = make_decoder(R"({"type":"yolo_pose","keypoints":2})", err);
        CHECK(dec != nullptr);
        const int K = 2, N = 3, F = 5 + 3 * K;
        // anchor rows: cx, cy, w, h, score, k0(x,y,conf), k1(x,y,conf)
        const float rows[N][F] = {
            {50, 50, 20, 40, 0.9f, 10, 20, 0.9f, 30, 40, 0.5f},
            {20, 80, 10, 10, 0.8f, 15, 85, 0.8f, 25, 75, 0.6f},
            {80, 20, 10, 10, 0.2f, 80, 20, 0.2f, 85, 25, 0.2f}};
        // `scale` applies to coordinates only: a normalized export scales the
        // box/keypoint values, not the probabilities.
        auto fill = [&](bool feature_major, float scale) {
            std::vector<float> d(static_cast<size_t>(F * N), 0.0f);
            for (int a = 0; a < N; ++a)
                for (int f = 0; f < F; ++f) {
                    const bool coord = f < 4 || (f >= 5 && (f - 5) % 3 != 2);
                    const float v = rows[a][f] * (coord ? scale : 1.0f);
                    const size_t idx = feature_major
                                           ? static_cast<size_t>(f * N + a)
                                           : static_cast<size_t>(a * F + f);
                    d[idx] = v;
                }
            return d;
        };
        std::vector<float> chw = fill(true, 1.0f);    // [F, N]
        std::vector<float> nhw = fill(false, 1.0f);   // [N, F]
        std::vector<float> norm = fill(true, 0.01f);  // normalized coords
        TensorView v_feat{chw.data(), chw.size(), {F, N}, ""};
        TensorView v_rows{nhw.data(), nhw.size(), {N, F}, ""};
        TensorView v_norm{norm.data(), norm.size(), {F, N}, ""};
        DetectionResult feat, rws, nm;
        CHECK(dec->decode(&v_feat, 1, 100, 100, 0.5f, 0.45f, feat, err));
        CHECK(dec->decode(&v_rows, 1, 100, 100, 0.5f, 0.45f, rws, err));
        CHECK(dec->decode(&v_norm, 1, 100, 100, 0.5f, 0.45f, nm, err));
        CHECK(feat.dets.size() == 2 && rws.dets.size() == 2 && nm.dets.size() == 2);
        CHECK(feat.kpts.size() == 4);
        CHECK_NEAR(feat.dets[0].cx, 0.5, 1e-5);
        CHECK_NEAR(feat.dets[0].cy, 0.5, 1e-5);
        CHECK_NEAR(feat.dets[0].w, 0.2, 1e-5);
        CHECK_NEAR(feat.dets[0].h, 0.4, 1e-5);
        CHECK_NEAR(feat.dets[0].score, 0.9, 1e-5);
        CHECK(feat.dets[0].kpt_count == 2);
        CHECK_NEAR(feat.kpts[0].x, 0.1, 1e-5);
        CHECK_NEAR(feat.kpts[0].y, 0.2, 1e-5);
        CHECK_NEAR(feat.kpts[0].conf, 0.9, 1e-5);
        CHECK_NEAR(feat.kpts[1].x, 0.3, 1e-5);
        CHECK_NEAR(feat.kpts[1].y, 0.4, 1e-5);
        CHECK_NEAR(feat.kpts[1].conf, 0.5, 1e-5);
        CHECK_NEAR(feat.dets[1].cx, 0.2, 1e-5);
        CHECK_NEAR(feat.dets[1].score, 0.8, 1e-5);
        for (size_t i = 0; i < 2; ++i) {
            const Detection& a = feat.dets[i];
            const Detection& b = rws.dets[i];
            CHECK_NEAR(a.cx, b.cx, 1e-6);
            CHECK_NEAR(a.cy, b.cy, 1e-6);
            CHECK_NEAR(a.w, b.w, 1e-6);
            CHECK_NEAR(a.h, b.h, 1e-6);
            CHECK_NEAR(a.score, b.score, 1e-6);
            CHECK_NEAR(a.cx, nm.dets[i].cx, 1e-5);
            CHECK_NEAR(a.cy, nm.dets[i].cy, 1e-5);
        }
        // A single 2-D tensor with the wrong feature count is rejected.
        {
            std::vector<float> bad(static_cast<size_t>(F + 1) * 2, 0.0f);
            TensorView v_bad{bad.data(), bad.size(), {F + 1, 2}, ""};
            DetectionResult o;
            err.clear();
            CHECK(!dec->decode(&v_bad, 1, 100, 100, 0.5f, 0.45f, o, err));
            CHECK(err.find("5 + 3*keypoints") != std::string::npos);
        }
    }

    for (const char* f : {"decode_yolov8_case1.json", "decode_yolov8_dfl_case1.json",
                          "decode_classify_case1.json", "decode_yolo_pose_case1.json"}) {
        if (!run_fixture(dir + "/" + f)) return 1;
    }
    std::printf("test_decoder: all OK\n");
    return 0;
}
