// M1.6 analyzer tests: run every line_cross/zone external fixture
// (contracts/fixtures/vb/analyzers/*.json, §6.2.4 format) plus the plugin
// loader checks.
#include <cstdio>
#include <string>
#include <vector>

#include "analyzer_fixture.h"
#include "check.h"
#include "vb/analyzer.h"
#include "vb/json.h"

namespace {

const char* kLineFixtures[] = {
    "line_cross_basic.json",       // forward crossing; B4 values from the event-frame det
    "line_cross_backward.json",    // reverse direction
    "line_cross_on_line_jitter.json",  // x=0.5 band=0: on-line frames never double-count
    "line_cross_endpoint_bypass.json", // walking around a segment endpoint is not a crossing
    "line_cross_reverse.json",     // forward then backward (reversing vehicle): two events
    "line_cross_reborn.json",      // ID-switch inheritance: no double count after reborn
    "line_cross_min_frames.json",  // noise in the first frames is not counted
    "line_cross_min_travel.json",  // jitter around the line is not counted
    "line_cross_b4_coasting.json", // B4: coasting frame uses the last associated det
};
const char* kZoneFixtures[] = {
    "zone_basic.json",
    "zone_removal.json",
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: test_analyzers <fixtures-dir> <count_plugin.so> <bad_plugin.so>\n");
        return 2;
    }
    const std::string dir = argv[1];
    for (const char* f : kLineFixtures) {
        std::string err;
        auto a = vb::create_analyzer("line_cross", err);
        CHECK(a != nullptr);
        auto j = vb::json_parse(read_file(dir + "/" + f));
        auto r = vb_fixture::run_analyzer_fixture(*a, j);
        if (!r.ok) {
            std::fprintf(stderr, "%s: FAIL: %s\n", f, r.fail.c_str());
            return 1;
        }
        std::printf("  %s: OK\n", f);
    }
    for (const char* f : kZoneFixtures) {
        std::string err;
        auto a = vb::create_analyzer("zone", err);
        CHECK(a != nullptr);
        auto j = vb::json_parse(read_file(dir + "/" + f));
        auto r = vb_fixture::run_analyzer_fixture(*a, j);
        if (!r.ok) {
            std::fprintf(stderr, "%s: FAIL: %s\n", f, r.fail.c_str());
            return 1;
        }
        std::printf("  %s: OK\n", f);
    }

    // Unknown analyzer name fails loudly.
    {
        std::string err;
        CHECK(vb::create_analyzer("nope", err) == nullptr);
        CHECK(!err.empty());
    }

    // Plugin loader: good plugin runs the same external-fixture path.
    {
        std::string err;
        auto a = vb::load_plugin_analyzer(argv[2], err);
        CHECK(a != nullptr);
        CHECK_STREQ(a->name(), "count");
        CHECK(a->attr_count() == 1);
        auto j = vb::json_parse(read_file(dir + "/plugin_count.json"));
        auto r = vb_fixture::run_analyzer_fixture(*a, j);
        if (!r.ok) {
            std::fprintf(stderr, "plugin_count.json: FAIL: %s\n", r.fail.c_str());
            return 1;
        }
        std::printf("  plugin_count.json: OK\n");

        // attrs: per-track frame counter
        std::string cerr;
        CHECK(a->configure("{}", cerr));
        vb::FrameMeta m{};
        m.geom = vb::letterbox_fit(1000, 1000, 1000, 1000, vb::Align::Center);
        vb::Track tr;
        tr.track_id = 7;
        tr.det.cx = 0.5f;
        tr.det.cy = 0.5f;
        std::vector<vb::Track> tracks{tr};
        float attrs[3] = {0, 0, 0};
        std::vector<vb::AnalyzerEvent> evs;
        for (int i = 0; i < 3; ++i) {
            m.seq = i + 1;
            m.t_mono_s = i * 0.1;
            a->on_frame(m, tracks, attrs + i, evs);
        }
        CHECK_NEAR(attrs[0], 1.0, 1e-6);
        CHECK_NEAR(attrs[1], 2.0, 1e-6);
        CHECK_NEAR(attrs[2], 3.0, 1e-6);
        CHECK(evs.empty());

        // configure failure keeps the old instance usable
        CHECK(!a->configure("not json", cerr));
        CHECK(!cerr.empty());
        a->on_frame(m, tracks, attrs, evs);
        CHECK_NEAR(attrs[0], 4.0, 1e-6);
    }

    // ABI mismatch: load fails with an explicit error (never silently skipped).
    {
        std::string err;
        CHECK(vb::load_plugin_analyzer(argv[3], err) == nullptr);
        CHECK(err.find("abi") != std::string::npos);
        std::printf("  bad-abi plugin rejected: %s\n", err.c_str());
    }

    std::printf("analyzers: all OK\n");
    return 0;
}
