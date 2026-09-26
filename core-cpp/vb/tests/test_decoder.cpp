// M1.15 decoder tests: make_decoder error cases, yolov8 transposed-vs-rows
// layout parity, classify softmax/top_k, plus the §6.11 decode fixtures.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "vb/decoder.h"
#include "vb/json.h"

using namespace vb;

namespace {

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

    for (const char* f : {"decode_yolov8_case1.json", "decode_yolov8_dfl_case1.json",
                          "decode_classify_case1.json"}) {
        if (!run_fixture(dir + "/" + f)) return 1;
    }
    std::printf("test_decoder: all OK\n");
    return 0;
}
