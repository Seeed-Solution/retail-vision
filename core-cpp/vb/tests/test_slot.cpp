#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#include "analyzer_fixture.h"
#include "check.h"
#include "vb/analyzer.h"
#include "vb/geom2d.h"
#include "vb/json.h"

namespace vb { std::unique_ptr<Analyzer> make_slot_coverage_analyzer(); }

int main(int argc, char** argv) {
    vb::Polygon cw{{.2f,.2f},{.2f,.8f},{.8f,.8f},{.8f,.2f}};
    vb::Polygon ccw{{.2f,.2f},{.8f,.2f},{.8f,.8f},{.2f,.8f}};
    CHECK_NEAR(vb::polygon_area(cw), .36, 1e-6);
    CHECK_NEAR(vb::polygon_area(ccw), .36, 1e-6);
    CHECK(vb::valid_convex_polygon(cw));
    CHECK(vb::valid_convex_polygon(ccw));
    vb::Polygon star{{.5f,.05f},{.62f,.4f},{.98f,.4f},{.69f,.62f},{.82f,.96f},
                     {.5f,.75f},{.18f,.96f},{.31f,.62f},{.02f,.4f},{.38f,.4f}};
    CHECK(!vb::valid_convex_polygon(star));
    vb::Polygon triangle{{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}};
    vb::Polygon pentagon{{0.f, .3f}, {.25f, 0.f}, {.75f, 0.f}, {1.f, .3f}, {.5f, 1.f}};
    vb::Polygon collinear{{0.f, 0.f}, {.5f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
    vb::Polygon same_turn_star{{.5f, 0.f}, {0.024f, .345f}, {.794f, .905f}, {.206f, .905f}, {.976f, .345f}};
    CHECK(vb::valid_convex_polygon(triangle));
    CHECK(vb::valid_convex_polygon(pentagon));
    CHECK(vb::valid_convex_polygon(collinear));
    CHECK(!vb::valid_convex_polygon(same_turn_star));
    vb::Polygon tiny{{0.f, 0.f}, {.001f, 0.f}, {.001f, .0005f}, {0.f, .0005f}};
    CHECK(!vb::valid_convex_polygon(tiny));
    CHECK_NEAR(vb::polygon_intersection_area_rect(cw, .3f, .3f, .7f, .7f), .16, 1e-6);
    if (argc < 2) return 2;
    std::string err;
    auto a = vb::make_slot_coverage_analyzer();
    CHECK(a != nullptr);
    for (const char* name : {"slot_basic.json", "slot_m41_geometry.json", "slot_m41_hold.json", "slot_m41_reconfigure.json",
                             "slot_m41_clip_lower_occupied.json", "slot_m41_triangle_coverage.json",
                             "slot_m41_default_hold.json", "slot_m41_pending_transaction.json"}) {
        auto j = vb::json_parse(read_file(std::string(argv[1]) + "/" + name));
        auto r = vb_fixture::run_analyzer_fixture(*a, j);
        if (!r.ok) { std::fprintf(stderr, "%s: FAIL: %s\n", name, r.fail.c_str()); return 1; }
        std::printf("%s: OK\n", name);
        a = vb::make_slot_coverage_analyzer();
    }
    auto bad = vb::make_slot_coverage_analyzer();
    auto bj = vb::json_parse(read_file(std::string(argv[1]) + "/slot_invalid.json"));
    auto br = vb_fixture::run_analyzer_fixture(*bad, bj);
    CHECK(br.ok);
    std::printf("slot_invalid.json: OK\n");
    auto bad_m41 = vb::make_slot_coverage_analyzer();
    auto bad_m41_j = vb::json_parse(read_file(std::string(argv[1]) + "/slot_m41_invalid.json"));
    auto bad_m41_r = vb_fixture::run_analyzer_fixture(*bad_m41, bad_m41_j);
    CHECK(bad_m41_r.ok);
    for (const char* name : {"slot_m41_classes_uint64_max.json", "slot_m41_classes_uint64_max_minus_1.json",
                             "slot_m41_classes_null.json"}) {
        auto classes_bad = vb::make_slot_coverage_analyzer();
        auto classes_j = vb::json_parse(read_file(std::string(argv[1]) + "/" + name));
        auto classes_r = vb_fixture::run_analyzer_fixture(*classes_bad, classes_j);
        CHECK(classes_r.ok);
        std::printf("%s: OK\n", name);
    }
    std::printf("slot_m41_invalid.json: OK\n");
    auto strict = vb::make_slot_coverage_analyzer();
    std::string strict_err;
    CHECK(!strict->configure(R"({"slots":[{"id":"dot.id","polygon":[[0,0],[1,0],[1,1],[0,1]]}],"free_ratio":0.35,"occupied_ratio":0.35})", strict_err));
    CHECK(!strict->configure(R"({"slots":[{"id":"x","polygon":[[0,0],[1,0],[1,1],[0,1]]}],"hold_s":"3"})", strict_err));
    CHECK(!strict->configure(R"({"slots":[{"id":"x","polygon":[[0,0],[1,0],[1,1],[0,1]]}],"free_hold_s":-1})", strict_err));
    CHECK(!strict->configure(R"({"slots":[{"id":"x","polygon":[[0,0],[1,0],[1,1],[0,1]]}],"classes":null})", strict_err));
    CHECK(strict_err.find("classes must be an array of ints") != std::string::npos);
    CHECK(!strict->configure(R"({"slots":[{"id":"x","polygon":[[0,0],[1,0],[1,1],[0,1]]}],"classes":[18446744073709551615]})", strict_err));
    CHECK(!strict->configure(R"({"slots":[{"id":"x","polygon":[[0,0],[1,0],[1,1],[0,1]]}],"classes":[18446744073709551614]})", strict_err));
    CHECK(!strict->configure(R"({"slots":[{"id":"x","polygon":[[0,0],[1,0],[1,1],[0,1]]}],"classes":[-2147483649]})", strict_err));
    CHECK(!strict->configure(R"({"slots":[{"id":"x","polygon":[[0,0],[1,0],[1,1],[0,1]]}],"classes":[1.0]})", strict_err));
    CHECK(!strict->configure(R"({"slots":[{"id":"bad","polygon":[[0.5,0.05],[0.62,0.4],[0.98,0.4],[0.69,0.62],[0.82,0.96],[0.5,0.75],[0.18,0.96],[0.31,0.62],[0.02,0.4],[0.38,0.4]]}]})", strict_err));
    CHECK(strict_err == "slot bad: polygon not convex");
    return 0;
}
