// Host-side self test for the shared core. core-cpp/ has no Hailo, GStreamer or
// sscma dependency, so the exact tracking / dwell / payload code that runs on
// the board also builds and runs on a laptop.
//
// It drives two stationary people through the tracker at 15 fps and prints the
// VisionPayload, which is what pins down the parts of the contract that do not
// need an NPU to be wrong: one batched message per cycle, slot numbering within
// the batch, centre coordinates as percentages, and the dwell state machine
// crossing into "engaged" after the configured threshold.
//
//   c++ -std=c++17 -I core-cpp tools/core_selftest.cpp \
//       core-cpp/person_tracker.cpp core-cpp/zone_metrics.cpp \
//       core-cpp/mqtt_payload.cpp -o /tmp/core_selftest

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mqtt_payload.h"
#include "person_tracker.h"
#include "zone_metrics.h"

using namespace retail_vision;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

bool contains(const std::string& h, const std::string& n) {
    return h.find(n) != std::string::npos;
}

}  // namespace

int main() {
    PersonTracker tracker;
    ZoneMetrics zone;

    TrackerConfig cfg;
    cfg.frame_width = 640;
    cfg.frame_height = 640;
    tracker.setConfig(cfg);
    zone.setWindowDuration(60.0f);
    tracker.setTrackRemovedCallback([&zone](const TrackRecord& r) { zone.onTrackRemoved(r); });

    // Two people, both standing still, in the same frame. Coordinates are in the
    // model's normalized space, which is what the Hailo decoder emits.
    std::vector<DetectionBox> dets = {
        {0.30f, 0.55f, 0.14f, 0.60f, 0.88f, 0},
        {0.68f, 0.52f, 0.15f, 0.62f, 0.81f, 0},
    };

    std::vector<TrackedPerson> persons;
    const float dt = 1.0f / 15.0f;
    float t = 0.0f;
    // 3 s is past dwell_min_duration (1.5 s) and short of the assistance
    // threshold (20 s), so both people should land in "engaged".
    for (int i = 0; i < 45; ++i) {
        t += dt;
        persons = tracker.update(dets, t);
        zone.update(tracker.getStateCounts(), tracker.getEntryCount(), tracker.getExitCount(), t);
    }

    check(persons.size() == 2, "two detections in one frame produce two tracked persons");

    const std::string json = buildVisionJson(1709500000000ULL, 45, 14.9f, 7.3f, zone.getSnapshot(),
                                             persons, 1280, 720, 640, 640);
    std::printf("\n%s\n\n", json.c_str());

    check(contains(json, "\"slot\":0"), "first person carries slot 0");
    check(contains(json, "\"slot\":1"), "second person carries slot 1");
    check(!contains(json, "\"slot\":2"), "slot numbering is bounded by people in the batch");
    check(contains(json, "\"frame_width\":1280") && contains(json, "\"frame_height\":720"),
          "payload reports source resolution, not the model square");
    check(contains(json, "\"occupancy_count\":2"), "zone occupancy counts both people");
    check(contains(json, "\"state\":\"engaged\""), "dwell state machine reaches engaged after 3 s");
    check(contains(json, "cx_pct") && contains(json, "cy_pct"), "centre is reported as percentages");
    check(contains(json, "\"velocity\":{") && contains(json, "speed_m_s"), "velocity block present");
    check(contains(json, "\"entry_count\":2"), "both arrivals counted as entries");

    // Letterbox regression. A 16:9 source in a square model leaves padding bands
    // top and bottom; a box whose predicted height spills into the padding used
    // to be published as y=-0.05, h=1.10. Intersecting with the frame rectangle
    // is the right answer rather than a clamp of convenience: a person cannot be
    // taller than the picture, so the part outside it was never a real detection.
    struct Case {
        const char* name;
        float h;           // height in model-normalized space
        bool expect_full;  // whether the corrected box should fill the frame height
    };
    const Case cases[] = {
        {"mid-shot, wholly inside the content band", 0.30f, false},
        {"box exactly filling the content band", 0.5625f, true},
        {"box overflowing into the padding", 0.62f, true},
    };

    for (const auto& c : cases) {
        PersonTracker tk;
        ZoneMetrics zm;
        tk.setConfig(cfg);
        std::vector<DetectionBox> one = {{0.5f, 0.5f, 0.12f, c.h, 0.9f, 0}};
        std::vector<TrackedPerson> tp;
        float tt = 0.0f;
        for (int i = 0; i < 20; ++i) {
            tt += dt;
            tp = tk.update(one, tt);
            zm.update(tk.getStateCounts(), tk.getEntryCount(), tk.getExitCount(), tt);
        }
        const std::string js =
            buildVisionJson(1709500000000ULL, 20, 15.0f, 7.0f, zm.getSnapshot(), tp, 1280, 720, 640, 640);

        // Pull bbox numbers straight back out of the emitted JSON: the contract is
        // the text on the wire, not the intermediate floats.
        const size_t at = js.find("\"bbox\":{");
        float bx = 9, by = 9, bw = 9, bh = 9;
        if (at != std::string::npos) {
            std::sscanf(js.c_str() + at, "\"bbox\":{\"x\":%f,\"y\":%f,\"w\":%f,\"h\":%f", &bx, &by, &bw, &bh);
        }
        std::printf("     %-44s -> x=%.4f y=%.4f w=%.4f h=%.4f\n", c.name, bx, by, bw, bh);
        const bool in_range = bx >= 0.f && by >= 0.f && bw >= 0.f && bh >= 0.f &&
                              bx + bw <= 1.0001f && by + bh <= 1.0001f;
        check(in_range, std::string("bbox stays inside [0,1]: ") + c.name);
        if (c.expect_full) {
            check(bh > 0.999f, std::string("overflow converges to full frame height: ") + c.name);
        } else {
            check(bh > 0.4f && bh < 0.7f, std::string("ordinary box is left alone: ") + c.name);
        }
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "OK", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
