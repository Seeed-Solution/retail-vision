// M1.16 tests: motion_util unit checks + every dwell/speed/direction external
// fixture (contracts/fixtures/vb/analyzers/{dwell,speed,direction}_*.json,
// spec BASE-1 §6.2.4/§6.2.5).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "analyzers/motion_util.h"
#include "analyzer_fixture.h"
#include "check.h"
#include "vb/analyzer.h"
#include "vb/json.h"

namespace {

const char* kDwellFixtures[] = {
    "dwell_static_1fps.json",     // static 12 s -> dwell_start once at 10 s
    "dwell_static_15fps.json",    // same times at 15 fps (+-1 frame)
    "dwell_radius_boundary.json", // 0.029 stays, 0.031 resets
    "dwell_moved.json",           // moves out of radius -> reason "moved"
    "dwell_repeat.json",          // repeat_s=5 -> dwell_update at 15 s and 20 s
    "dwell_lost.json",            // track removal -> reason "lost"
    "dwell_zones.json",           // anchor outside the zone never accrues
};
const char* kSpeedFixtures[] = {
    "speed_over.json",       // 0.6 fh/s > 0.5 -> speed_over after hold
    "speed_normal.json",     // fast -> slow -> speed_normal
    "speed_window_short.json", // span < 0.5*window_s -> no judgement
    "speed_homography.json", // diag(x10) homography -> m/s, value x10
    "speed_aspect.json",     // 1920x1080: same pixel speed h/v -> same speed
};
const char* kDirectionFixtures[] = {
    "direction_basic.json",     // down (90) ok, up (270) -> wrong_way
    "direction_min_travel.json",// sub-min_travel jitter ignored
    "direction_recover.json",   // back to down -> wrong_way_end
    "direction_tolerance.json", // tolerance 45: 44 deg reports, 46 deg not
};

void run_fixtures(const char* analyzer, const char* const* fixtures, size_t n,
                  const std::string& dir) {
    for (size_t i = 0; i < n; ++i) {
        std::string err;
        auto a = vb::create_analyzer(analyzer, err);
        CHECK(a != nullptr);
        auto j = vb::json_parse(read_file(dir + "/" + fixtures[i]));
        auto r = vb_fixture::run_analyzer_fixture(*a, j);
        if (!r.ok) {
            std::fprintf(stderr, "%s: FAIL: %s\n", fixtures[i], r.fail.c_str());
            std::exit(1);
        }
        std::printf("  %s: OK\n", fixtures[i]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_motion <fixtures-dir>\n");
        return 2;
    }
    const std::string dir = argv[1];

    // wrap_deg / ang_diff (§6.2.5)
    CHECK_NEAR(vb::wrap_deg(0.0f), 0.0, 1e-6);
    CHECK_NEAR(vb::wrap_deg(-90.0f), 270.0, 1e-6);
    CHECK_NEAR(vb::wrap_deg(450.0f), 90.0, 1e-6);
    CHECK_NEAR(vb::ang_diff(90.0f, 270.0f), 180.0, 1e-6);
    CHECK_NEAR(vb::ang_diff(0.0f, 50.0f), 50.0, 1e-6);
    CHECK_NEAR(vb::ang_diff(350.0f, 10.0f), 20.0, 1e-6);

    // Debounce: needs hold_s of a differing raw before committing.
    {
        vb::Debounce d("normal");
        std::string prev, next;
        CHECK(!d.step("over", 0.0, 0.5, prev, next));
        CHECK(!d.step("over", 0.4, 0.5, prev, next));
        CHECK(d.step("over", 0.5, 0.5, prev, next));
        CHECK(prev == "normal" && next == "over");
        CHECK(d.state == "over");
        // raw == state clears pending
        vb::Debounce e("ok");
        CHECK(!e.step("wrong", 0.0, 1.0, prev, next));
        CHECK(!e.step("ok", 0.9, 1.0, prev, next));
        CHECK(!e.step("wrong", 1.5, 1.0, prev, next));  // pending restarted
        CHECK(!e.step("wrong", 2.4, 1.0, prev, next));
        CHECK(e.step("wrong", 2.5, 1.0, prev, next));
    }

    // point_in_polygon (geom2d, M1.16)
    {
        vb::Polygon sq{{0.2f, 0.2f}, {0.8f, 0.2f}, {0.8f, 0.8f}, {0.2f, 0.8f}};
        CHECK(vb::point_in_polygon(0.5f, 0.5f, sq));
        CHECK(!vb::point_in_polygon(0.1f, 0.5f, sq));
        CHECK(!vb::point_in_polygon(0.9f, 0.9f, sq));
        vb::Polygon tri{{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
        CHECK(vb::point_in_polygon(0.25f, 0.25f, tri));
        CHECK(!vb::point_in_polygon(0.75f, 0.75f, tri));
    }

    run_fixtures("dwell", kDwellFixtures, sizeof kDwellFixtures / sizeof(*kDwellFixtures), dir);
    run_fixtures("speed", kSpeedFixtures, sizeof kSpeedFixtures / sizeof(*kSpeedFixtures), dir);
    run_fixtures("direction", kDirectionFixtures,
                 sizeof kDirectionFixtures / sizeof(*kDirectionFixtures), dir);

    std::printf("test_motion: all OK\n");
    return 0;
}
