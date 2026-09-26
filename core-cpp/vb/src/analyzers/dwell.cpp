// Built-in dwell analyzer (spec BASE-1 §6.2.5, M1.16): loitering detection.
// Unlike zone_dwell (still inside a zone), dwell requires the anchor to stay
// within `radius` (in units of frame height) of its entry position.
// Events:
//   dwell_start  {dwell_s, anchor:[x,y], zone_id, class_id, score}
//   dwell_update {dwell_s, anchor:[x,y], zone_id, class_id, score}  every repeat_s
//   dwell_end    {dwell_s, reason:"moved"|"left_zone"|"lost"}
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "analyzers/dwell.h"
#include "analyzers/motion_util.h"

namespace vb {
namespace {

struct DwellAnalyzer : public Analyzer {
    struct Cfg {
        float radius = 0.03f;
        double min_dwell_s = 10.0;
        double repeat_s = 0.0;
        std::vector<ZoneCfg> zones;
        std::string anchor = "bottom_center";
        std::vector<int> classes;
    };
    struct St {
        float hx = 0, hy = 0;        // entry anchor in height-normalized coords
        float sx = 0, sy = 0;        // entry anchor, source-normalized
        double t0 = 0;
        bool active = false;
        double last = 0;
        std::string zone;
    };

    const char* name() const override { return "dwell"; }

    bool parse_cfg(const Json& j, Cfg& c, std::string& err) {
        if (!j.is_object()) {
            err = "dwell: config must be an object";
            return false;
        }
        if (!parse_zones(j, "dwell", c.zones, err)) return false;
        if (!parse_classes(j, "dwell", c.classes, err)) return false;
        if (!parse_anchor(j, "dwell", c.anchor, err)) return false;
        auto f = j.find("radius");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() <= 0) {
                err = "dwell: radius must be > 0";
                return false;
            }
            c.radius = f->get<float>();
        }
        f = j.find("min_dwell_s");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0) {
                err = "dwell: min_dwell_s must be >= 0";
                return false;
            }
            c.min_dwell_s = f->get<double>();
        }
        f = j.find("repeat_s");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0) {
                err = "dwell: repeat_s must be >= 0";
                return false;
            }
            c.repeat_s = f->get<double>();
        }
        return true;
    }

    bool configure(const std::string& json, std::string& err) override {
        Json j;
        try {
            j = Json::parse(json);
        } catch (const std::exception& e) {
            err = std::string("dwell: invalid json: ") + e.what();
            return false;
        }
        Cfg c;
        if (!parse_cfg(j, c, err)) return false;
        cfg_ = std::move(c);
        st_.clear();
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
            zone_of(cfg_.zones, px, py, zid);

            float hx = px, hy = py;
            to_h(hx, hy, m.geom);

            auto it = st_.find(tr.track_id);
            if (it == st_.end()) {
                st_[tr.track_id] = St{hx, hy, px, py, now, false, 0.0, zid};
                continue;
            }
            St& s = it->second;

            bool left = (!cfg_.zones.empty() && zid.empty()) || zid != s.zone;
            if (left) {
                if (s.active)
                    end_event(out, tr.track_id, now - s.t0, "left_zone");
                st_[tr.track_id] = St{hx, hy, px, py, now, false, 0.0, zid};
                continue;
            }
            float dx = hx - s.hx, dy = hy - s.hy;
            if (std::sqrt(dx * dx + dy * dy) > cfg_.radius) {
                if (s.active)
                    end_event(out, tr.track_id, now - s.t0, "moved");
                st_[tr.track_id] = St{hx, hy, px, py, now, false, 0.0, zid};
                continue;
            }
            double d = now - s.t0;
            if (!s.active && d >= cfg_.min_dwell_s) {
                s.active = true;
                s.last = now;
                dwell_event(out, "dwell_start", tr, s, d, zid);
            } else if (s.active && cfg_.repeat_s > 0 && now - s.last >= cfg_.repeat_s) {
                s.last = now;
                dwell_event(out, "dwell_update", tr, s, d, zid);
            }
        }
    }

    void dwell_event(std::vector<AnalyzerEvent>& out, const char* type, const Track& tr,
                     const St& s, double d, const std::string& zid) const {
        Json f{{"dwell_s", d},
               {"anchor", Json{s.sx, s.sy}},
               {"zone_id", zid},
               {"class_id", tr.det.class_id},
               {"score", tr.det.score}};
        emit_event(out, type, tr.track_id, f);
    }

    void end_event(std::vector<AnalyzerEvent>& out, uint32_t id, double d,
                   const char* reason) const {
        emit_event(out, "dwell_end", id, Json{{"dwell_s", d}, {"reason", reason}});
    }

    void on_track_removed(uint32_t track_id, double t_mono_s,
                          std::vector<AnalyzerEvent>& out) override {
        auto it = st_.find(track_id);
        if (it == st_.end()) return;
        if (it->second.active)
            end_event(out, track_id, t_mono_s - it->second.t0, "lost");
        st_.erase(it);
    }

    Cfg cfg_;
    std::map<uint32_t, St> st_;
};

}  // namespace

std::unique_ptr<Analyzer> make_dwell_analyzer() {
    return std::make_unique<DwellAnalyzer>();
}

}  // namespace vb
