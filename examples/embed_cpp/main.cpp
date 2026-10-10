// vb_algo embed example (spec BASE-1 §6.13.2, M1.24): feeds 10 frames of
// synthetic detections (one box walking from y=0.4 to y=0.8) through
// vb::Tracker + a line_cross analyzer over a y=0.6 horizontal line and
// prints each emitted event as one JSON line; expect exactly one
// {"type":"line_cross",...}.
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "vb/analyzer.h"
#include "vb/json.h"
#include "vb/letterbox.h"
#include "vb/tracker.h"

int main() {
    using namespace vb;
    std::string err;
    auto analyzer = create_analyzer("line_cross", err);
    if (!analyzer || !analyzer->configure(
            R"({"lines":[{"id":"demo","a":[0.1,0.6],"b":[0.9,0.6]}]})", err)) {
        std::fprintf(stderr, "line_cross setup failed: %s\n", err.c_str());
        return 1;
    }
    Tracker tracker;
    const LetterboxGeom geom = LetterboxGeom::fit(1000, 1000, 1000, 1000, Align::Center);
    std::vector<AnalyzerEvent> events;
    for (int i = 0; i < 10; ++i) {
        Detection d;
        d.cx = 0.5f; d.cy = 0.40f + 0.045f * i; d.w = 0.1f; d.h = 0.1f;
        d.score = 0.9f; d.class_id = 0;
        std::vector<Detection> dets{d};
        std::vector<uint32_t> removed;
        const auto& tracks = tracker.update(dets, {}, 0.1 * i, removed);
        FrameMeta m{};
        m.seq = static_cast<uint64_t>(i + 1);
        m.t_mono_s = 0.1 * i;
        m.geom = geom;
        analyzer->on_frame(m, tracks, nullptr, events);
    }
    for (const auto& ev : events) {
        Json out = Json{{"type", ev.type}, {"track_id", ev.track_id}};
        out["fields"] = Json::parse(ev.fields_json);
        std::printf("%s\n", out.dump().c_str());
    }
    return events.size() == 1 ? 0 : 1;
}
