// vb_selftest: cross-language fixture runner (spec BASE-1 §8 M1.3+, §6.2.4).
//
// Usage:
//   vb_selftest letterbox <letterbox_cases.json>
//   vb_selftest analyzer <name> <fixture.json> [--plugin <path.so>]
//   vb_selftest decode <fixture.json>       (§6.11 decode fixtures, M1.15)
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "analyzer_fixture.h"
#include "check.h"
#include "vb/analyzer.h"
#include "vb/decoder.h"
#include "vb/json.h"
#include "vb/letterbox.h"

namespace {

int run_letterbox(const std::string& path) {
    auto j = vb::json_parse(read_file(path));
    int n_cases = 0, n_points = 0;
    for (const auto& c : j.at("cases")) {
        auto src = c.at("src");
        auto model = c.at("model");
        vb::Align a = (c.at("align").get<std::string>() == "center") ? vb::Align::Center
                                                                     : vb::Align::TopLeft;
        auto g = vb::letterbox_fit(src[0].get<int>(), src[1].get<int>(),
                                   model[0].get<int>(), model[1].get<int>(), a);
        for (const auto& p : c.at("points")) {
            float sx, sy;
            g.to_source_norm(p.at("model")[0].get<float>(),
                             p.at("model")[1].get<float>(), sx, sy);
            auto expect = p.at("expect");
            CHECK_NEAR(sx, expect[0].get<double>(), 1e-5);
            CHECK_NEAR(sy, expect[1].get<double>(), 1e-5);
            ++n_points;
        }
        // round-trip through to_model_norm must land back on the input point
        for (const auto& p : c.at("points")) {
            float mx = p.at("model")[0].get<float>(), my = p.at("model")[1].get<float>();
            float sx, sy, rx, ry;
            g.to_source_norm(mx, my, sx, sy);
            g.to_model_norm(sx, sy, rx, ry);
            CHECK_NEAR(rx, mx, 1e-6);
            CHECK_NEAR(ry, my, 1e-6);
        }
        ++n_cases;
    }
    std::printf("letterbox: %d cases, %d points OK\n", n_cases, n_points);
    return 0;
}

}  // namespace

namespace {

// §6.11 decode fixture: {"decoder": {...}, "model_hw": [w,h], "score": s,
// "nms": i, "tensors": [{"dims": [...], "data": [...]}], "expect": [det...]}
// (batch dims removed from `dims`; data row-major; comparison tol 1e-5).
int run_decode(const std::string& path) {
    auto j = vb::json_parse(read_file(path));
    std::string err;
    auto dec = vb::make_decoder(j.at("decoder").dump(), err);
    if (!dec) {
        std::fprintf(stderr, "decode %s: %s\n", path.c_str(), err.c_str());
        return 1;
    }
    const int mw = j.at("model_hw")[0].get<int>();
    const int mh = j.at("model_hw")[1].get<int>();
    const float score = j.value("score", 0.35);
    const float nms = j.value("nms", 0.45);
    std::vector<vb::TensorView> views;
    std::vector<std::vector<float>> storage;
    for (const auto& t : j.at("tensors")) {
        storage.emplace_back();
        for (const auto& v : t.at("data")) storage.back().push_back(v.get<float>());
        vb::TensorView tv;
        tv.data = storage.back().data();
        tv.count = storage.back().size();
        for (const auto& d : t.at("dims")) tv.dims.push_back(d.get<int64_t>());
        views.push_back(std::move(tv));
    }
    vb::DetectionResult out;
    if (!dec->decode(views.data(), views.size(), mw, mh, score, nms, out, err)) {
        std::fprintf(stderr, "decode %s: %s\n", path.c_str(), err.c_str());
        return 1;
    }
    const auto& expect = j.at("expect");
    if (out.dets.size() != expect.size()) {
        std::fprintf(stderr, "decode %s: expected %zu dets, got %zu\n", path.c_str(),
                     expect.size(), out.dets.size());
        return 1;
    }
    for (size_t i = 0; i < expect.size(); ++i) {
        const auto& e = expect[i];
        const vb::Detection& d = out.dets[i];
        bool bad = std::fabs(d.cx - e.at("cx").get<double>()) > 1e-5 ||
                   std::fabs(d.cy - e.at("cy").get<double>()) > 1e-5 ||
                   std::fabs(d.w - e.at("w").get<double>()) > 1e-5 ||
                   std::fabs(d.h - e.at("h").get<double>()) > 1e-5 ||
                   std::fabs(d.score - e.at("score").get<double>()) > 1e-5 ||
                   d.class_id != e.at("class_id").get<int>();
        if (bad) {
            std::fprintf(stderr,
                         "decode %s: det %zu mismatch: got cx=%g cy=%g w=%g h=%g "
                         "score=%g class=%d\n",
                         path.c_str(), i, d.cx, d.cy, d.w, d.h, d.score, d.class_id);
            return 1;
        }
    }
    std::printf("decode %s: %zu dets OK (%s)\n", path.c_str(), out.dets.size(),
                j.at("decoder").at("type").get<std::string>().c_str());
    return 0;
}

}  // namespace

namespace {

int run_analyzer(const std::string& name, const std::string& path, bool plugin,
                 const std::string& plugin_path) {
    std::string err;
    std::unique_ptr<vb::Analyzer> a =
        plugin ? vb::load_plugin_analyzer(plugin_path, err) : vb::create_analyzer(name, err);
    if (!a) {
        std::fprintf(stderr, "analyzer %s: %s\n", name.c_str(), err.c_str());
        return 1;
    }
    auto j = vb::json_parse(read_file(path));
    if (j.contains("analyzer")) {
        std::string want = j.at("analyzer").get<std::string>();
        if (!plugin && want != name) {
            std::fprintf(stderr, "fixture is for analyzer \"%s\", run as \"%s\"\n",
                         want.c_str(), name.c_str());
            return 1;
        }
    }
    auto r = vb_fixture::run_analyzer_fixture(*a, j);
    if (!r.ok) {
        std::fprintf(stderr, "analyzer %s: first diff: %s\n", name.c_str(), r.fail.c_str());
        return 1;
    }
    std::printf("analyzer %s: %s OK\n", name.c_str(), path.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "letterbox") == 0) {
        return run_letterbox(argv[2]);
    }
    if (argc >= 3 && std::strcmp(argv[1], "decode") == 0) {
        return run_decode(argv[2]);
    }
    if (argc >= 4 && std::strcmp(argv[1], "analyzer") == 0) {
        bool plugin = (argc >= 6 && std::strcmp(argv[4], "--plugin") == 0);
        return run_analyzer(argv[2], argv[3], plugin,
                            plugin ? argv[5] : std::string());
    }
    std::fprintf(stderr,
                 "usage: vb_selftest letterbox <fixture.json>\n"
                 "       vb_selftest analyzer <name> <fixture.json> [--plugin <path.so>]\n"
                 "       vb_selftest decode <fixture.json>\n");
    return 2;
}
