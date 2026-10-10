// M1.5 tracker test: reproduces the retail-vision behaviour fixture
// core-py/retail_core/tests/fixtures/tracker_sequence.json (expectations from
// the authoritative C++ person_tracker) plus the generic-tracker cases in
// contracts/fixtures/vb/tracker_generic.json.
//
// Note on ids: the retail fixture predates the base track_id semantics
// (§6.2: ids start at 1, 0 = unmatched), so its 0-based expectations are
// compared with a +1 offset.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "check.h"
#include "vb/json.h"
#include "vb/tracker.h"

namespace {

vb::TrackerConfig config_from_json(const vb::Json& j) {
    vb::TrackerConfig c;
    if (j.contains("iou_threshold")) c.iou_threshold = j.at("iou_threshold").get<float>();
    if (j.contains("dist_threshold")) c.dist_threshold = j.at("dist_threshold").get<float>();
    if (j.contains("max_misses")) c.max_misses = j.at("max_misses").get<int>();
    if (j.contains("max_lost_s")) c.max_lost_s = j.at("max_lost_s").get<float>();
    if (j.contains("min_hits")) c.min_hits = j.at("min_hits").get<int>();
    if (j.contains("class_aware")) c.class_aware = j.at("class_aware").get<bool>();
    if (j.contains("anchor") && j.at("anchor").get<std::string>() == "bottom_center") {
        c.anchor = vb::TrackerConfig::BottomCenter;
    }
    return c;
}

std::vector<uint32_t> visible_ids(const std::vector<vb::Track>& alive) {
    std::vector<uint32_t> ids;
    for (const auto& t : alive) {
        if (t.misses == 0) ids.push_back(t.track_id);  // "visible" in the retail sense
    }
    return ids;  // alive is in ascending-id order
}

void run_retail_sequence(const vb::Json& j) {
    const double fps = j.at("fps").get<double>();
    vb::Tracker tr(vb::TrackerConfig{});
    std::vector<uint32_t> removed;
    size_t ei = 0;
    const auto& expect = j.at("expect_frames");
    const auto& frames = j.at("frames");
    for (size_t f = 0; f < frames.size(); ++f) {
        std::vector<vb::Detection> dets;
        for (const auto& d : frames[f]) {
            vb::Detection det;
            det.cx = d[0].get<float>();
            det.cy = d[1].get<float>();
            det.w = d[2].get<float>();
            det.h = d[3].get<float>();
            det.score = d[4].get<float>();
            det.class_id = 0;
            dets.push_back(det);
        }
        const std::vector<vb::Track>& alive =
            tr.update(dets, {}, static_cast<double>(f) / fps, removed);
        if (ei < expect.size() && expect[ei].at("frame").get<size_t>() == f) {
            auto got = visible_ids(alive);
            std::vector<uint32_t> want;
            for (int id : expect[ei].at("track_ids")) want.push_back(static_cast<uint32_t>(id + 1));
            CHECK(got == want);
            ++ei;
        }
    }
    CHECK(ei == expect.size());
    std::printf("  retail sequence: %zu frames, %zu checkpoints OK\n", frames.size(), expect.size());
}

void run_generic_cases(const vb::Json& j) {
    for (const auto& c : j.at("cases")) {
        const std::string name = c.at("name").get<std::string>();
        const double fps = c.at("fps").get<double>();
        vb::Tracker tr(config_from_json(c.value("config", vb::Json::object())));
        std::vector<uint32_t> removed;
        struct Rem { size_t frame; std::vector<uint32_t> ids; };
        std::vector<Rem> rems;
        for (const auto& e : c.value("expect_removed", vb::Json::array())) {
            Rem r{e.at("frame").get<size_t>(), {}};
            for (uint32_t id : e.at("removed")) r.ids.push_back(id);
            std::sort(r.ids.begin(), r.ids.end());
            rems.push_back(r);
        }
        size_t ei = 0, ri = 0;
        const auto& expect = c.at("expect");
        const auto& frames = c.at("frames");
        for (size_t f = 0; f < frames.size(); ++f) {
            std::vector<vb::Detection> dets;
            for (const auto& d : frames[f]) {
                vb::Detection det;
                det.cx = d[0].get<float>();
                det.cy = d[1].get<float>();
                det.w = d[2].get<float>();
                det.h = d[3].get<float>();
                det.score = d[4].get<float>();
                det.class_id = d.size() > 5 ? d[5].get<int>() : 0;
                dets.push_back(det);
            }
            const std::vector<vb::Track>& alive =
                tr.update(dets, {}, static_cast<double>(f) / fps, removed);
            if (ei < expect.size() && expect[ei].at("frame").get<size_t>() == f) {
                auto got = visible_ids(alive);
                std::vector<uint32_t> want;
                for (uint32_t id : expect[ei].at("track_ids")) want.push_back(id);
                CHECK(got == want);
                ++ei;
            }
            if (ri < rems.size() && rems[ri].frame == f) {
                auto got = removed;
                std::sort(got.begin(), got.end());
                CHECK(got == rems[ri].ids);
                ++ri;
            }
        }
        CHECK(ei == expect.size());
        CHECK(ri == rems.size());
        std::printf("  case %s: OK\n", name.c_str());
    }
}

void check_anchor_cases_differ(const vb::Json& j) {
    // M1.5 review finding / M1.9 fix: the anchor_center_no_fallback_match and
    // anchor_bottom_center_fallback_match pair must assign different ids on
    // frame 1 (the anchors change the fallback reference point only).
    std::vector<uint32_t> center, bottom;
    for (const auto& c : j.at("cases")) {
        const std::string name = c.at("name").get<std::string>();
        if (name != "anchor_center_no_fallback_match" &&
            name != "anchor_bottom_center_fallback_match")
            continue;
        for (const auto& e : c.at("expect"))
            if (e.at("frame").get<size_t>() == 1)
                for (uint32_t id : e.at("track_ids"))
                    (name.rfind("anchor_center", 0) == 0 ? center : bottom)
                        .push_back(id);
    }
    CHECK(!center.empty() && !bottom.empty());
    CHECK(center != bottom);
    std::printf("  anchor pair: center -> id %u, bottom_center -> id %u (differ)\n",
                center[0], bottom[0]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_tracker <tracker_sequence.json> <tracker_generic.json>\n");
        return 2;
    }
    std::printf("retail fixture (%s):\n", argv[1]);
    run_retail_sequence(vb::json_parse(read_file(argv[1])));
    std::printf("generic fixture (%s):\n", argv[2]);
    run_generic_cases(vb::json_parse(read_file(argv[2])));
    check_anchor_cases_differ(vb::json_parse(read_file(argv[2])));

    // reset(): ids restart from 1
    {
        vb::Tracker tr;
        std::vector<uint32_t> removed;
        std::vector<vb::Detection> d{{0.5f, 0.5f, 0.1f, 0.1f, 0.9f, 0}};
        const auto& a1 = tr.update(d, {}, 0.0, removed);
        CHECK(a1.size() == 1 && a1[0].track_id == 1);
        tr.reset();
        const auto& a2 = tr.update(d, {}, 1.0, removed);
        CHECK(a2.size() == 1 && a2[0].track_id == 1);
        // unmatched dets always receive a non-zero id (0 is reserved)
        CHECK(d[0].track_id == 1);
    }
    // keypoints follow the detection into the track
    {
        vb::Tracker tr;
        std::vector<uint32_t> removed;
        std::vector<vb::Detection> d{{0.5f, 0.5f, 0.1f, 0.1f, 0.9f, 0}};
        d[0].kpt_offset = 0;
        d[0].kpt_count = 2;
        std::vector<vb::Keypoint> kp{{0.4f, 0.4f, 0.9f}, {0.6f, 0.6f, 0.8f}};
        const auto& alive = tr.update(d, kp, 0.0, removed);
        CHECK(alive.size() == 1 && alive[0].kpts.size() == 2);
        CHECK_NEAR(alive[0].kpts[1].x, 0.6f, 1e-6);
    }
    std::printf("tracker: all OK\n");
    return 0;
}
