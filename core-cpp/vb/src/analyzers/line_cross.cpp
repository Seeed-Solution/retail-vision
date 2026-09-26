// Built-in line_cross analyzer (spec BASE-1 §6.2, M1.6; design source:
// NRD-2 §6.2 LineCounter pseudocode + esk contracts/MQTT.md side==0 rule).
//
// Semantics:
//  - Anchor point = bbox bottom-centre mapped to source-normalized coords.
//    For coasting tracks (misses > 0) the anchor is extrapolated with the
//    track velocity so a track that disappears mid-crossing is still
//    counted on a coasting frame (B4: class_id/score then come from the
//    last associated detection, which Track.det already holds).
//  - side = sign of the signed distance to the directed line a->b, with a
//    hysteresis band: |d| <= band leaves the last explicit side unchanged
//    (frames "on the line" never flip the side and never double-count).
//  - A crossing is counted only when the projection parameter t is within
//    [-segment_margin, 1+segment_margin]; walking around a segment endpoint
//    updates the side without emitting an event.
//  - Crossings before min_track_frames observed frames update the side but
//    are not counted; a second crossing within min_travel of the last
//    counted anchor is jitter and not counted; the same direction is only
//    counted once per track per line.
//  - "forward" = positive side -> negative side, "backward" = the reverse.
//  - ID-switch inheritance: when a track is removed after being counted, a
//    new track whose anchor appears within reborn_dist and reborn_window_s
//    inherits side + counted direction (no double count).
#include <array>
#include <cmath>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "vb/analyzer.h"
#include "vb/json.h"

namespace vb {
namespace {

struct LineCrossAnalyzer : public Analyzer {
    struct Line {
        std::string id;
        float ax = 0, ay = 0, bx = 0, by = 0;
    };
    struct Cfg {
        std::vector<Line> lines;
        float band = 0.0f;
        float segment_margin = 0.05f;
        float min_travel = 0.02f;
        float reborn_window_s = 1.5f;
        float reborn_dist = 0.05f;
        int min_track_frames = 3;
        std::vector<int> classes;  // empty = all classes
    };
    struct PerTrack {
        int side = 0;             // -1/0/+1; 0 until the first explicit side
        int counted = 0;          // +1 forward, -1 backward, 0 never
        bool has_anchor = false;  // anchor at last count (min_travel guard)
        float ax = 0, ay = 0;
        float lx = 0, ly = 0;     // last computed anchor (for removal memory)
        uint32_t age = 0;         // frames observed (including coasting)
    };
    struct Lost {
        double t;
        float ax, ay;
        int counted, side;
    };
    struct LineState {
        std::map<uint32_t, PerTrack> st;
        std::deque<Lost> lost;
    };

    const char* name() const override { return "line_cross"; }

    bool parse_cfg(const Json& j, Cfg& c, std::string& err) {
        if (!j.is_object()) {
            err = "line_cross: config must be an object";
            return false;
        }
        auto it = j.find("lines");
        if (it == j.end() || !it->is_array() || it->empty() || it->size() > 64) {
            err = "line_cross: lines must be a non-empty array (1..64)";
            return false;
        }
        for (const auto& lj : *it) {
            if (!lj.is_object() || !lj.contains("id") || !lj.contains("a") || !lj.contains("b")) {
                err = "line_cross: each line needs id, a, b";
                return false;
            }
            Line ln;
            ln.id = lj.at("id").get<std::string>();
            if (ln.id.empty() || ln.id.size() > 32) {
                err = "line_cross: line id must be 1..32 chars";
                return false;
            }
            for (const auto& l : c.lines) {
                if (l.id == ln.id) {
                    err = "line_cross: duplicate line id " + ln.id;
                    return false;
                }
            }
            auto rd = [&lj](const char* k, float& x, float& y) {
                const auto& v = lj.at(k);
                x = v.at(0).get<float>();
                y = v.at(1).get<float>();
            };
            rd("a", ln.ax, ln.ay);
            rd("b", ln.bx, ln.by);
            if (std::hypot(ln.bx - ln.ax, ln.by - ln.ay) < 1e-9f) {
                err = "line_cross: line " + ln.id + " is degenerate";
                return false;
            }
            c.lines.push_back(std::move(ln));
        }
        auto getf = [&j, &err](const char* k, float& out, float lo, float hi) {
            auto f = j.find(k);
            if (f == j.end()) return true;
            if (!f->is_number()) {
                err = std::string("line_cross: ") + k + " must be a number";
                return false;
            }
            double v = f->get<double>();
            if (v < lo || v > hi) {
                err = std::string("line_cross: ") + k + " out of range";
                return false;
            }
            out = static_cast<float>(v);
            return true;
        };
        if (!getf("band", c.band, 0, 1)) return false;
        if (!getf("segment_margin", c.segment_margin, 0, 0.5)) return false;
        if (!getf("min_travel", c.min_travel, 0, 1)) return false;
        if (!getf("reborn_window_s", c.reborn_window_s, 0, 3600)) return false;
        if (!getf("reborn_dist", c.reborn_dist, 0, 1)) return false;
        auto f = j.find("min_track_frames");
        if (f != j.end()) {
            if (!f->is_number_integer() || f->get<int>() < 1) {
                err = "line_cross: min_track_frames must be an int >= 1";
                return false;
            }
            c.min_track_frames = f->get<int>();
        }
        f = j.find("classes");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_array()) {
                err = "line_cross: classes must be an array of ints";
                return false;
            }
            for (const auto& cl : *f) {
                if (!cl.is_number_integer()) {
                    err = "line_cross: classes must be an array of ints";
                    return false;
                }
                c.classes.push_back(cl.get<int>());
            }
        }
        return true;
    }

    bool configure(const std::string& json, std::string& err) override {
        Json j;
        try {
            j = Json::parse(json);
        } catch (const std::exception& e) {
            err = std::string("line_cross: invalid json: ") + e.what();
            return false;
        }
        Cfg c;
        if (!parse_cfg(j, c, err)) return false;
        cfg_ = std::move(c);
        lst_.assign(cfg_.lines.size(), LineState{});
        return true;
    }

    static bool class_ok(const Cfg& c, int32_t cid) {
        return c.classes.empty() ||
               std::find(c.classes.begin(), c.classes.end(), cid) != c.classes.end();
    }

    void on_frame(const FrameMeta& m, const std::vector<Track>& tracks, float* attrs,
                  std::vector<AnalyzerEvent>& out) override {
        (void)attrs;
        for (const auto& tr : tracks) {
            if (!class_ok(cfg_, tr.det.class_id)) continue;
            // bottom-centre anchor in model coords, extrapolated while coasting
            float mx = tr.det.cx, my = tr.det.cy + tr.det.h * 0.5f;
            if (tr.misses > 0) {
                double dts = m.t_mono_s - tr.last_seen_s;
                if (dts < 0) dts = 0;
                mx += static_cast<float>(tr.vx * dts);
                my += static_cast<float>(tr.vy * dts);
            }
            float px, py;
            m.geom.to_source_norm(mx, my, px, py);
            for (size_t li = 0; li < cfg_.lines.size(); ++li) {
                step(li, tr.track_id, px, py, m.t_mono_s, tr, out);
            }
        }
    }

    void step(size_t li, uint32_t tid, float px, float py, double now, const Track& tr,
              std::vector<AnalyzerEvent>& out) {
        const Line& ln = cfg_.lines[li];
        LineState& ls = lst_[li];
        PerTrack& s = ls.st[tid];

        if (s.age == 0) {  // new track: try ID-switch inheritance
            for (auto it = ls.lost.begin(); it != ls.lost.end(); ++it) {
                if (now - it->t <= cfg_.reborn_window_s &&
                    std::hypot(px - it->ax, py - it->ay) <= cfg_.reborn_dist) {
                    s.counted = it->counted;
                    s.side = it->side;
                    ls.lost.erase(it);
                    break;
                }
            }
        }
        s.age++;
        s.lx = px;
        s.ly = py;

        float dx = ln.bx - ln.ax, dy = ln.by - ln.ay;
        float len = std::hypot(dx, dy);
        float d = (dx * (py - ln.ay) - dy * (px - ln.ax)) / len;  // signed distance
        float tproj = ((px - ln.ax) * dx + (py - ln.ay) * dy) / (len * len);

        int cur = 0;
        if (d > cfg_.band) cur = 1;
        else if (d < -cfg_.band) cur = -1;
        else return;  // inside the hysteresis band: keep the last explicit side

        if (s.side == 0) {
            s.side = cur;  // first explicit side: never counts
            return;
        }
        if (cur == s.side) return;

        // Crossing from s.side to cur.
        if (tproj < -cfg_.segment_margin || tproj > 1 + cfg_.segment_margin) {
            s.side = cur;  // walked around a segment endpoint
            return;
        }
        if (static_cast<int>(s.age) < cfg_.min_track_frames) {
            s.side = cur;  // too young
            return;
        }
        const char* direction = (s.side > 0) ? "forward" : "backward";
        int dir = (s.side > 0) ? 1 : -1;
        if (s.counted == dir) {
            s.side = cur;  // same direction counted once per track
            return;
        }
        if (s.has_anchor && std::hypot(px - s.ax, py - s.ay) < cfg_.min_travel) {
            s.side = cur;  // jitter around the line
            return;
        }
        AnalyzerEvent ev;
        ev.type = "line_cross";
        ev.track_id = tid;
        ev.fields_json = Json{{"line_id", ln.id},
                              {"direction", direction},
                              {"anchor", Json::array({px, py})},
                              {"class_id", tr.det.class_id},
                              {"score", tr.det.score}}
                             .dump();
        out.push_back(std::move(ev));
        s.counted = dir;
        s.ax = px;
        s.ay = py;
        s.has_anchor = true;
        s.side = cur;
    }

    void on_track_removed(uint32_t track_id, double t_mono_s,
                          std::vector<AnalyzerEvent>& out) override {
        (void)out;  // line_cross emits nothing on removal; it only records memory
        for (size_t li = 0; li < cfg_.lines.size(); ++li) {
            LineState& ls = lst_[li];
            auto it = ls.st.find(track_id);
            if (it == ls.st.end()) continue;
            if (it->second.counted != 0) {
                ls.lost.push_back(Lost{t_mono_s, it->second.lx, it->second.ly,
                                       it->second.counted, it->second.side});
            }
            ls.st.erase(it);
            while (!ls.lost.empty() && t_mono_s - ls.lost.front().t > cfg_.reborn_window_s) {
                ls.lost.pop_front();
            }
        }
    }

    Cfg cfg_;
    std::vector<LineState> lst_;
};

}  // namespace

std::unique_ptr<Analyzer> make_line_cross_analyzer() {
    return std::make_unique<LineCrossAnalyzer>();
}

}  // namespace vb
