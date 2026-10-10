// Built-in zone analyzer (spec BASE-1 §6.2, M1.6).
//
// Anchor = bbox bottom-centre in source-normalized coords; point-in-polygon
// by ray casting. Events (B4: class_id/score from the track's last
// associated detection, which Track.det already holds):
//   zone_enter {zone_id, class_id, score}
//   zone_exit  {zone_id, dwell_s, class_id, score}
//   zone_dwell {zone_id, dwell_s, class_id, score}  - once per continuous
//              stay, on the first frame where dwell_s >= min_dwell_s
//              (min_dwell_s <= 0 never fires zone_dwell; zone_enter already
//              covers the degenerate case)
// A track removed while inside a zone emits zone_exit from
// on_track_removed with dwell_s up to the removal frame.
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "vb/analyzer.h"
#include "vb/json.h"

namespace vb {
namespace {

bool point_in_poly(float x, float y, const std::vector<std::array<float, 2>>& poly) {
    bool inside = false;
    size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        float xi = poly[i][0], yi = poly[i][1];
        float xj = poly[j][0], yj = poly[j][1];
        if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
}

struct ZoneAnalyzer : public Analyzer {
    struct Zone {
        std::string id;
        std::vector<std::array<float, 2>> poly;
    };
    struct Cfg {
        std::vector<Zone> zones;
        double min_dwell_s = 0;
        std::vector<int> classes;
    };
    struct ZState {
        bool inside = false;
        double entered_at = 0;
        bool dwell_fired = false;
        int32_t class_id = 0;   // last associated detection (B4)
        float score = 0;
    };
    struct ZoneState {
        std::map<uint32_t, ZState> st;
    };

    const char* name() const override { return "zone"; }

    bool parse_cfg(const Json& j, Cfg& c, std::string& err) {
        if (!j.is_object()) {
            err = "zone: config must be an object";
            return false;
        }
        auto it = j.find("zones");
        if (it == j.end() || !it->is_array() || it->empty() || it->size() > 64) {
            err = "zone: zones must be a non-empty array (1..64)";
            return false;
        }
        for (const auto& zj : *it) {
            if (!zj.is_object() || !zj.contains("id") || !zj.contains("polygon")) {
                err = "zone: each zone needs id and polygon";
                return false;
            }
            Zone z;
            z.id = zj.at("id").get<std::string>();
            if (z.id.empty() || z.id.size() > 32) {
                err = "zone: zone id must be 1..32 chars";
                return false;
            }
            for (const auto& o : c.zones) {
                if (o.id == z.id) {
                    err = "zone: duplicate zone id " + z.id;
                    return false;
                }
            }
            const auto& poly = zj.at("polygon");
            if (!poly.is_array() || poly.size() < 3 || poly.size() > 16) {
                err = "zone: polygon must have 3..16 vertices";
                return false;
            }
            for (const auto& p : poly) {
                if (!p.is_array() || p.size() != 2 || !p[0].is_number() || !p[1].is_number() ||
                    p[0].get<double>() < 0 || p[0].get<double>() > 1 ||
                    p[1].get<double>() < 0 || p[1].get<double>() > 1) {
                    err = "zone: polygon coords must be in [0,1] (source-normalized)";
                    return false;
                }
                z.poly.push_back({p[0].get<float>(), p[1].get<float>()});
            }
            c.zones.push_back(std::move(z));
        }
        auto f = j.find("min_dwell_s");
        if (f != j.end()) {
            if (!f->is_number() || f->get<double>() < 0) {
                err = "zone: min_dwell_s must be >= 0";
                return false;
            }
            c.min_dwell_s = f->get<double>();
        }
        f = j.find("classes");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_array()) {
                err = "zone: classes must be an array of ints";
                return false;
            }
            for (const auto& cl : *f) {
                if (!cl.is_number_integer()) {
                    err = "zone: classes must be an array of ints";
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
            err = std::string("zone: invalid json: ") + e.what();
            return false;
        }
        Cfg c;
        if (!parse_cfg(j, c, err)) return false;
        cfg_ = std::move(c);
        zst_.assign(cfg_.zones.size(), ZoneState{});
        return true;
    }

    void on_frame(const FrameMeta& m, const std::vector<Track>& tracks, float* attrs,
                  std::vector<AnalyzerEvent>& out) override {
        (void)attrs;
        for (const auto& tr : tracks) {
            if (!cfg_.classes.empty() &&
                std::find(cfg_.classes.begin(), cfg_.classes.end(), tr.det.class_id) ==
                    cfg_.classes.end()) {
                continue;
            }
            float mx = tr.det.cx, my = tr.det.cy + tr.det.h * 0.5f;
            float px, py;
            m.geom.to_source_norm(mx, my, px, py);
            for (size_t zi = 0; zi < cfg_.zones.size(); ++zi) {
                bool in = point_in_poly(px, py, cfg_.zones[zi].poly);
                step(zi, tr, in, m.t_mono_s, out);
            }
        }
    }

    void step(size_t zi, const Track& tr, bool in, double now, std::vector<AnalyzerEvent>& out) {
        ZState& s = zst_[zi].st[tr.track_id];
        s.class_id = tr.det.class_id;
        s.score = tr.det.score;
        const Zone& z = cfg_.zones[zi];
        auto emit = [&](const char* type, double dwell) {
            Json f{{"zone_id", z.id}, {"class_id", tr.det.class_id}, {"score", tr.det.score}};
            if (dwell >= 0) f["dwell_s"] = dwell;
            AnalyzerEvent ev;
            ev.type = type;
            ev.track_id = tr.track_id;
            ev.fields_json = f.dump();
            out.push_back(std::move(ev));
        };
        if (in && !s.inside) {
            s.inside = true;
            s.entered_at = now;
            s.dwell_fired = false;
            emit("zone_enter", -1);
        } else if (s.inside && !in) {
            s.inside = false;
            emit("zone_exit", now - s.entered_at);
        }
        if (s.inside) {
            double dwell = now - s.entered_at;
            if (!s.dwell_fired && cfg_.min_dwell_s > 0 && dwell >= cfg_.min_dwell_s) {
                s.dwell_fired = true;
                emit("zone_dwell", dwell);
            }
        }
    }

    void on_track_removed(uint32_t track_id, double t_mono_s,
                          std::vector<AnalyzerEvent>& out) override {
        for (size_t zi = 0; zi < cfg_.zones.size(); ++zi) {
            auto it = zst_[zi].st.find(track_id);
            if (it == zst_[zi].st.end()) continue;
            if (it->second.inside) {
                Json f{{"zone_id", cfg_.zones[zi].id},
                       {"dwell_s", t_mono_s - it->second.entered_at},
                       {"class_id", it->second.class_id},
                       {"score", it->second.score}};
                AnalyzerEvent ev;
                ev.type = "zone_exit";
                ev.track_id = track_id;
                ev.fields_json = f.dump();
                out.push_back(std::move(ev));
            }
            zst_[zi].st.erase(it);
        }
    }

    Cfg cfg_;
    std::vector<ZoneState> zst_;
};

}  // namespace

std::unique_ptr<Analyzer> make_zone_analyzer() {
    return std::make_unique<ZoneAnalyzer>();
}

}  // namespace vb
