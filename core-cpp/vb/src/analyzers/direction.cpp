// Built-in direction analyzer (spec BASE-1 §6.2.5, M1.16): wrong-way
// detection. Angles follow image coords in height-normalized space:
// 0 deg = right, 90 deg = down. Events:
//   wrong_way     {angle_deg, expected_deg, diff_deg, travel, zone_id, class_id, score}
//   wrong_way_end {duration_s}
#include <cmath>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "analyzers/direction.h"
#include "analyzers/motion_util.h"

namespace vb {
namespace {

struct DirectionAnalyzer : public Analyzer {
    struct Cfg {
        float expected_deg = 90.0f;
        float tolerance_deg = 60.0f;
        float min_travel = 0.05f;
        double window_s = 1.5;
        double hold_s = 0.5;
        std::vector<ZoneCfg> zones;
        std::string anchor = "bottom_center";
        std::vector<int> classes;
    };
    struct Hist {
        std::deque<double> ts;
        std::deque<std::pair<float, float>> pts;
        Debounce deb{"ok"};
        bool wrong = false;
        double since = 0;
    };

    const char* name() const override { return "direction"; }

    bool parse_cfg(const Json& j, Cfg& c, std::string& err) {
        if (!j.is_object()) {
            err = "direction: config must be an object";
            return false;
        }
        if (!parse_zones(j, "direction", c.zones, err)) return false;
        if (!parse_classes(j, "direction", c.classes, err)) return false;
        if (!parse_anchor(j, "direction", c.anchor, err)) return false;
        auto f = j.find("expected_deg");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number()) {
                err = "direction: expected_deg must be a number";
                return false;
            }
            c.expected_deg = f->get<float>();
        }
        f = j.find("tolerance_deg");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0 || f->get<double>() > 180) {
                err = "direction: tolerance_deg must be in [0,180]";
                return false;
            }
            c.tolerance_deg = f->get<float>();
        }
        f = j.find("min_travel");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0) {
                err = "direction: min_travel must be >= 0";
                return false;
            }
            c.min_travel = f->get<float>();
        }
        f = j.find("window_s");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() <= 0) {
                err = "direction: window_s must be > 0";
                return false;
            }
            c.window_s = f->get<double>();
        }
        f = j.find("hold_s");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0) {
                err = "direction: hold_s must be >= 0";
                return false;
            }
            c.hold_s = f->get<double>();
        }
        return true;
    }

    bool configure(const std::string& json, std::string& err) override {
        Json j;
        try {
            j = Json::parse(json);
        } catch (const std::exception& e) {
            err = std::string("direction: invalid json: ") + e.what();
            return false;
        }
        Cfg c;
        if (!parse_cfg(j, c, err)) return false;
        cfg_ = std::move(c);
        hist_.clear();
        return true;
    }

    void on_frame(const FrameMeta& m, const std::vector<Track>& tracks, float* attrs,
                  std::vector<AnalyzerEvent>& out) override {
        (void)attrs;
        const double now = m.t_mono_s;
        for (const auto& tr : tracks) {
            if (tr.misses != 0) continue;
            if (!class_ok(cfg_.classes, tr.det.class_id)) continue;
            float px, py;
            anchor_point(tr, m.geom, cfg_.anchor, px, py);
            std::string zid;
            if (!cfg_.zones.empty()) {
                zone_of(cfg_.zones, px, py, zid);
                if (zid.empty()) continue;  // zones filter: anchor must be inside
            }
            float hx = px, hy = py;
            to_h(hx, hy, m.geom);
            Hist& h = hist_[tr.track_id];
            h.ts.push_back(now);
            h.pts.push_back({hx, hy});
            while (!h.ts.empty() && now - h.ts.front() > cfg_.window_s) {
                h.ts.pop_front();
                h.pts.pop_front();
            }
            if (h.ts.size() < 2) continue;
            double span = h.ts.back() - h.ts.front();
            if (span < 0.5 * cfg_.window_s) continue;  // not enough samples
            double vx = static_cast<double>(h.pts.back().first) - h.pts.front().first;
            double vy = static_cast<double>(h.pts.back().second) - h.pts.front().second;
            double travel = std::sqrt(vx * vx + vy * vy);
            if (travel < cfg_.min_travel) continue;
            double ang = wrap_deg(std::atan2(vy, vx) * 180.0 / M_PI);
            double diff = ang_diff(ang, cfg_.expected_deg);
            std::string raw = diff > cfg_.tolerance_deg ? "wrong" : "ok";
            std::string prev, next;
            if (!h.deb.step(raw, now, cfg_.hold_s, prev, next)) continue;
            if (next == "wrong") {
                h.wrong = true;
                h.since = now;
                emit_event(out, "wrong_way", tr.track_id,
                           Json{{"angle_deg", ang},
                                {"expected_deg", cfg_.expected_deg},
                                {"diff_deg", diff},
                                {"travel", travel},
                                {"zone_id", zid},
                                {"class_id", tr.det.class_id},
                                {"score", tr.det.score}});
            } else {  // ("wrong", "ok")
                h.wrong = false;
                emit_event(out, "wrong_way_end", tr.track_id,
                           Json{{"duration_s", now - h.since}});
            }
        }
    }

    void on_track_removed(uint32_t track_id, double t_mono_s,
                          std::vector<AnalyzerEvent>& out) override {
        auto it = hist_.find(track_id);
        if (it == hist_.end()) return;
        if (it->second.wrong)
            emit_event(out, "wrong_way_end", track_id,
                       Json{{"duration_s", t_mono_s - it->second.since}});
        hist_.erase(it);
    }

    Cfg cfg_;
    std::map<uint32_t, Hist> hist_;
};

}  // namespace

std::unique_ptr<Analyzer> make_direction_analyzer() {
    return std::make_unique<DirectionAnalyzer>();
}

}  // namespace vb
