// Built-in count_threshold analyzer (spec BASE-1 §6.2.5, M1.17): per-zone
// occupancy limits with hysteresis. zones empty = one "frame" zone covering
// the whole picture. Events (track_id = 0, needs_tracks() = false):
//   count_over   {zone_id, count, limit}
//   count_under  {zone_id, count, limit}
//   count_normal {zone_id, count}
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "analyzers/count_threshold.h"
#include "analyzers/motion_util.h"

namespace vb {
namespace {

struct CountThresholdAnalyzer : public Analyzer {
    struct Cfg {
        std::vector<ZoneCfg> zones;   // empty config -> "frame" full-picture zone
        bool has_max = false, has_min = false;
        double max_count = 0, min_count = 0;
        double hold_s = 2.0;
        std::string anchor = "bottom_center";
        std::vector<int> classes;
    };
    struct ZoneState {
        Debounce deb{"normal"};
    };

    const char* name() const override { return "count_threshold"; }
    bool needs_tracks() const override { return false; }

    bool parse_cfg(const Json& j, Cfg& c, std::string& err) {
        if (!j.is_object()) {
            err = "count_threshold: config must be an object";
            return false;
        }
        if (!parse_zones(j, "count_threshold", c.zones, err)) return false;
        if (!parse_classes(j, "count_threshold", c.classes, err)) return false;
        if (!parse_anchor(j, "count_threshold", c.anchor, err)) return false;
        auto num = [&](const char* key, double& v, bool& has) -> bool {
            auto it = j.find(key);
            if (it == j.end() || it->is_null()) return true;
            if (!it->is_number_integer() || it->get<long long>() < 0) {
                err = std::string("count_threshold: ") + key +
                      " must be a non-negative integer";
                return false;
            }
            v = it->get<double>();
            has = true;
            return true;
        };
        if (!num("max_count", c.max_count, c.has_max)) return false;
        if (!num("min_count", c.min_count, c.has_min)) return false;
        if (!c.has_max && !c.has_min) {
            err = "count_threshold: at least one of max_count / min_count must be set";
            return false;
        }
        auto f = j.find("hold_s");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0) {
                err = "count_threshold: hold_s must be >= 0";
                return false;
            }
            c.hold_s = f->get<double>();
        }
        return true;
    }

    // Effective zone list: configured zones, or the implicit full-frame zone.
    std::vector<ZoneCfg> effective_zones() const {
        if (!cfg_.zones.empty()) return cfg_.zones;
        return {ZoneCfg{"frame", Polygon{{0, 0}, {1, 0}, {1, 1}, {0, 1}}}};
    }

    bool configure(const std::string& json, std::string& err) override {
        Json j;
        try {
            j = Json::parse(json);
        } catch (const std::exception& e) {
            err = std::string("count_threshold: invalid json: ") + e.what();
            return false;
        }
        Cfg c;
        if (!parse_cfg(j, c, err)) return false;
        // Zones whose id and polygon are unchanged keep their debounce state;
        // everything else resets to "normal".
        std::map<std::string, ZoneState> keep;
        for (const auto& z : c.zones) {
            auto it = zst_.find(z.id);
            if (it != zst_.end()) {
                // polygon equality check against the *old* config
                for (const auto& oz : cfg_.zones) {
                    if (oz.id == z.id && oz.poly == z.poly) {
                        keep[z.id] = std::move(it->second);
                        break;
                    }
                }
            }
        }
        cfg_ = std::move(c);
        zst_ = std::move(keep);
        return true;
    }

    void on_frame(const FrameMeta& m, const std::vector<Track>& tracks, float* attrs,
                  std::vector<AnalyzerEvent>& out) override {
        (void)attrs;
        const double now = m.t_mono_s;
        // Anchor points of the counted tracks (misses == 0 + class filter).
        std::vector<std::pair<float, float>> anchors;
        anchors.reserve(tracks.size());
        for (const auto& tr : tracks) {
            if (tr.misses != 0) continue;
            if (!class_ok(cfg_.classes, tr.det.class_id)) continue;
            float px, py;
            anchor_point(tr, m.geom, cfg_.anchor, px, py);
            anchors.push_back({px, py});
        }
        for (const auto& z : effective_zones()) {
            int count = 0;
            for (const auto& a : anchors)
                if (point_in_polygon(a.first, a.second, z.poly)) ++count;
            std::string raw = "normal";
            if (cfg_.has_max && count > cfg_.max_count)
                raw = "over";
            else if (cfg_.has_min && count < cfg_.min_count)
                raw = "under";
            auto it2 = zst_.try_emplace(z.id, ZoneState{});
            ZoneState& st = it2.first->second;
            std::string prev, next;
            if (!st.deb.step(raw, now, cfg_.hold_s, prev, next)) continue;
            if (next == "over") {
                emit_event(out, "count_over", 0,
                           Json{{"zone_id", z.id},
                                {"count", count},
                                {"limit", cfg_.has_max ? Json(cfg_.max_count) : Json()}});
            } else if (next == "under") {
                emit_event(out, "count_under", 0,
                           Json{{"zone_id", z.id},
                                {"count", count},
                                {"limit", cfg_.has_min ? Json(cfg_.min_count) : Json()}});
            } else {
                emit_event(out, "count_normal", 0,
                           Json{{"zone_id", z.id}, {"count", count}});
            }
        }
    }

    void on_track_removed(uint32_t track_id, double t_mono_s,
                          std::vector<AnalyzerEvent>& out) override {
        (void)track_id;
        (void)t_mono_s;
        (void)out;
    }

    Cfg cfg_;
    std::map<std::string, ZoneState> zst_;
};

}  // namespace

std::unique_ptr<Analyzer> make_count_threshold_analyzer() {
    return std::make_unique<CountThresholdAnalyzer>();
}

}  // namespace vb
