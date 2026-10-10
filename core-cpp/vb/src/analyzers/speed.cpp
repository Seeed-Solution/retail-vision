// Built-in speed analyzer (spec BASE-1 §6.2.5, M1.16): displacement over a
// sliding window. Without `homography` the unit is "fh/s" (frame heights per
// second, computed in height-normalized coords so aspect ratio cancels);
// with a 3x3 row-major homography mapping source-normalized (x,y,1) to
// ground metres, the unit is "m/s".
// Events:
//   speed_over   {speed, limit, unit, class_id, score}
//   speed_under  {speed, limit, unit, class_id, score}
//   speed_normal {speed, unit}
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "analyzers/motion_util.h"
#include "analyzers/speed.h"

namespace vb {
namespace {

struct SpeedAnalyzer : public Analyzer {
    struct Cfg {
        double window_s = 1.0;
        bool has_max = false, has_min = false;
        double max_speed = 0, min_speed = 0;
        double hold_s = 0.5;
        std::string anchor = "bottom_center";
        bool has_h = false;
        double h[9] = {0};
        std::vector<int> classes;
    };
    struct Hist {
        std::deque<double> ts;
        std::deque<std::pair<float, float>> pts;
        Debounce deb{"normal"};
    };

    const char* name() const override { return "speed"; }

    bool parse_cfg(const Json& j, Cfg& c, std::string& err) {
        if (!j.is_object()) {
            err = "speed: config must be an object";
            return false;
        }
        if (!parse_classes(j, "speed", c.classes, err)) return false;
        if (!parse_anchor(j, "speed", c.anchor, err)) return false;
        auto f = j.find("window_s");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() <= 0) {
                err = "speed: window_s must be > 0";
                return false;
            }
            c.window_s = f->get<double>();
        }
        auto num = [&](const char* key, double& v, bool& has) -> bool {
            auto it = j.find(key);
            if (it == j.end() || it->is_null()) return true;
            if (!it->is_number() || it->get<double>() < 0) {
                err = std::string("speed: ") + key + " must be a number >= 0";
                return false;
            }
            v = it->get<double>();
            has = true;
            return true;
        };
        if (!num("max_speed", c.max_speed, c.has_max)) return false;
        if (!num("min_speed", c.min_speed, c.has_min)) return false;
        if (!c.has_max && !c.has_min) {
            err = "speed: at least one of max_speed / min_speed must be set";
            return false;
        }
        f = j.find("hold_s");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0) {
                err = "speed: hold_s must be >= 0";
                return false;
            }
            c.hold_s = f->get<double>();
        }
        f = j.find("homography");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_array() || f->size() != 9) {
                err = "speed: homography must be null or 9 numbers (row-major)";
                return false;
            }
            for (size_t i = 0; i < 9; ++i) {
                if (!(*f)[i].is_number()) {
                    err = "speed: homography must be null or 9 numbers (row-major)";
                    return false;
                }
                c.h[i] = (*f)[i].get<double>();
            }
            c.has_h = true;
        }
        return true;
    }

    bool configure(const std::string& json, std::string& err) override {
        Json j;
        try {
            j = Json::parse(json);
        } catch (const std::exception& e) {
            err = std::string("speed: invalid json: ") + e.what();
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
        const char* unit = cfg_.has_h ? "m/s" : "fh/s";
        for (const auto& tr : tracks) {
            if (tr.misses != 0) continue;
            if (!class_ok(cfg_.classes, tr.det.class_id)) continue;
            float px, py;
            anchor_point(tr, m.geom, cfg_.anchor, px, py);
            float fx = px, fy = py;
            if (cfg_.has_h) {
                const double* H = cfg_.h;
                double w = H[6] * px + H[7] * py + H[8];
                fx = static_cast<float>((H[0] * px + H[1] * py + H[2]) / w);
                fy = static_cast<float>((H[3] * px + H[4] * py + H[5]) / w);
            } else {
                to_h(fx, fy, m.geom);
            }
            Hist& h = hist_[tr.track_id];
            h.ts.push_back(now);
            h.pts.push_back({fx, fy});
            while (!h.ts.empty() && now - h.ts.front() > cfg_.window_s) {
                h.ts.pop_front();
                h.pts.pop_front();
            }
            if (h.ts.size() < 2) continue;
            double span = h.ts.back() - h.ts.front();
            if (span < 0.5 * cfg_.window_s) continue;  // not enough samples
            double dx = static_cast<double>(h.pts.back().first) - h.pts.front().first;
            double dy = static_cast<double>(h.pts.back().second) - h.pts.front().second;
            double v = std::sqrt(dx * dx + dy * dy) / span;
            std::string raw = "normal";
            if (cfg_.has_max && v > cfg_.max_speed)
                raw = "over";
            else if (cfg_.has_min && v < cfg_.min_speed)
                raw = "under";
            std::string prev, next;
            if (!h.deb.step(raw, now, cfg_.hold_s, prev, next)) continue;
            if (next == "over") {
                emit_event(out, "speed_over", tr.track_id,
                           Json{{"speed", v},
                                {"limit", cfg_.has_max ? Json(cfg_.max_speed) : Json()},
                                {"unit", unit},
                                {"class_id", tr.det.class_id},
                                {"score", tr.det.score}});
            } else if (next == "under") {
                emit_event(out, "speed_under", tr.track_id,
                           Json{{"speed", v},
                                {"limit", cfg_.has_min ? Json(cfg_.min_speed) : Json()},
                                {"unit", unit},
                                {"class_id", tr.det.class_id},
                                {"score", tr.det.score}});
            } else {
                emit_event(out, "speed_normal", tr.track_id,
                           Json{{"speed", v}, {"unit", unit}});
            }
        }
    }

    void on_track_removed(uint32_t track_id, double t_mono_s,
                          std::vector<AnalyzerEvent>& out) override {
        (void)t_mono_s;
        (void)out;
        hist_.erase(track_id);
    }

    Cfg cfg_;
    std::map<uint32_t, Hist> hist_;
};

}  // namespace

std::unique_ptr<Analyzer> make_speed_analyzer() {
    return std::make_unique<SpeedAnalyzer>();
}

}  // namespace vb
