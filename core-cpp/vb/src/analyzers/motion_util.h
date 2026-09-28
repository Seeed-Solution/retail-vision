// Internal helpers shared by the §6.2.5 motion analyzers (dwell / speed /
// direction / count_threshold / pose_angle, spec BASE-1 §6.2.5, M1.16/M1.17).
#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

#include "vb/analyzer.h"
#include "vb/geom2d.h"
#include "vb/json.h"

namespace vb {

// Anchor point of a track box in source-normalized coords, letterbox
// back-transformed. anchor: "bottom_center" | "center".
void anchor_point(const Track& t, const LetterboxGeom& g, const std::string& anchor,
                  float& x, float& y);

// "Height-normalized" coords: distances and angles are computed here so
// aspect-ratio distortion cancels: (x * src_w / src_h, y).
inline void to_h(float& x, float& y, const LetterboxGeom& g) {
    (void)y;  // y is already frame-height normalized
    x = x * static_cast<float>(g.src_w) / static_cast<float>(g.src_h);
}

inline double wrap_deg(double a) {
    a = std::fmod(a, 360.0);
    if (a < 0) a += 360.0;
    return a;
}

inline double ang_diff(double a, double b) {
    double d = std::fabs(a - b);
    return d > 180.0 ? 360.0 - d : d;
}

// Hysteresis debouncer, one per judged object (§6.2.5). step() returns true
// exactly when the committed state changes; prev/next then hold the
// transition ("raw == state" clears any pending change).
struct Debounce {
    explicit Debounce(std::string initial) : state(std::move(initial)) {}

    bool step(const std::string& raw, double now, double hold_s,
              std::string& prev, std::string& next) {
        if (raw == state) {
            pending = false;
            return false;
        }
        if (!pending || pending_raw != raw) {
            pending = true;
            pending_raw = raw;
            pending_since = now;
            return false;
        }
        if (now - pending_since >= hold_s) {
            prev = state;
            state = raw;
            pending = false;
            next = state;
            return true;
        }
        return false;
    }

    std::string state;
    bool pending = false;
    std::string pending_raw;
    double pending_since = 0;
};

// Shared config pieces. zones: [{"id","polygon"}], 3..64 vertices, coords in
// [0,1] (source-normalized), unique non-empty ids.
struct ZoneCfg {
    std::string id;
    Polygon poly;
};

bool parse_zones(const Json& j, const char* who, std::vector<ZoneCfg>& out,
                 std::string& err);
bool parse_classes(const Json& j, const char* who, std::vector<int>& out,
                   std::string& err);
inline bool class_ok(const std::vector<int>& classes, int32_t cid) {
    return classes.empty() ||
           std::find(classes.begin(), classes.end(), cid) != classes.end();
}
// First zone whose polygon contains (x, y); nullptr when none. zone_id is ""
// when zones is empty, otherwise the hit zone's id ("" when no hit).
const ZoneCfg* zone_of(const std::vector<ZoneCfg>& zones, float x, float y,
                       std::string& zone_id);
// "anchor" key, default "bottom_center".
bool parse_anchor(const Json& j, const char* who, std::string& out, std::string& err);

// Emit helper: pushes one event with a JSON object body.
inline void emit_event(std::vector<AnalyzerEvent>& out, const char* type,
                       uint32_t track_id, const Json& fields) {
    AnalyzerEvent ev;
    ev.type = type;
    ev.track_id = track_id;
    ev.fields_json = fields.dump();
    out.push_back(std::move(ev));
}

}  // namespace vb
