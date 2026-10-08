#include "check.h"
#include "stage2.h"
#include <limits>

int main() {
    using namespace vb;
    std::vector<Detection> d(3);
    d[0].track_id = 1; d[0].score = .7f; d[0].cx = .5f; d[0].cy = .5f; d[0].w = .4f; d[0].h = .2f;
    d[1].track_id = 2; d[1].score = .9f; d[1].cx = .1f; d[1].cy = .1f; d[1].w = .4f; d[1].h = .4f;
    d[2].track_id = 3; d[2].score = .1f; d[2].cx = .5f; d[2].cy = .5f; d[2].w = .2f; d[2].h = .2f;
    Stage2FilterConfig c; c.min_score = .5f; c.max_crops_per_frame = 2; c.expand = .25f;
    std::map<uint32_t, Stage2TrackState> s; auto a = stage2_select_crops(d, 1.0, c, s);
    CHECK(a.size() == 2); CHECK(a[0].track_id == 2); CHECK(a[1].track_id == 1);
    CHECK(a[0].x0 == 0.0f); CHECK(a[0].x1 <= 0.4f);
    c.max_per_track = 1; auto b = stage2_select_crops(d, 1.1, c, s); CHECK(b.empty());
    c.max_per_track = 0; c.min_gap_s = 2.0; auto e = stage2_select_crops(d, 1.2, c, s); CHECK(e.empty());
    auto f = stage2_select_crops(d, 3.2, c, s); CHECK(f.size() == 2);
    s.clear(); c.min_gap_s = 0; c.classes = {7};
    d[0].class_id = 7; d[1].class_id = 2;
    auto classes = stage2_select_crops(d, 10, c, s);
    CHECK(classes.size() == 1); CHECK(classes[0].track_id == 1);
    c.classes.clear(); s.clear();
    c.roi_polygon = {.3f,.3f,.9f,.3f,.9f,.9f,.3f,.9f};
    auto roi = stage2_select_crops(d, 10, c, s);
    CHECK(roi.size() == 1); CHECK(roi[0].track_id == 1);
    c.roi_polygon.clear(); s.clear();
    d[0].cx = std::numeric_limits<float>::quiet_NaN();
    auto finite = stage2_select_crops(d, 10, c, s);
    CHECK(finite.size() == 1); CHECK(finite[0].track_id == 2);
    d[0].cx = .5f; d[0].score = d[1].score;
    s.clear(); c.max_crops_per_frame = 1;
    auto stable = stage2_select_crops(d, 10, c, s);
    CHECK(stable.size() == 1); CHECK(stable[0].track_id == 1);
    return 0;
}
