// Parking-slot occupancy from the lower half of detection boxes (BASE-1 M4.1).
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <regex>
#include <string>
#include <vector>

#include "vb/analyzer.h"
#include "vb/geom2d.h"
#include "vb/json.h"

namespace vb {
namespace {

struct SlotAnalyzer final : Analyzer {
    struct Slot { std::string id; Polygon poly; double occupied = .5, free = .2;
                  double occupied_hold = 0, free_hold = 0; std::vector<int> classes; };
    struct State { std::string value = "unknown"; double since = 0, pending_since = 0;
                   std::string pending; uint32_t track_id = 0; };
    struct Config { std::vector<Slot> slots; std::vector<int> classes; };

    const char* name() const override { return "slot_coverage"; }
    bool needs_tracks() const override { return false; }

    static bool finite_num(const Json& x) { return x.is_number() && std::isfinite(x.get<double>()); }
    static bool valid_id(const std::string& id) {
        return !id.empty() && id.size() <= 32 && std::regex_match(id, std::regex("[A-Za-z0-9_.-]+"));
    }
    static bool parse_classes(const Json& j, std::vector<int>& out, std::string& err) {
        if (!j.is_array()) { err = "slot_coverage: classes must be an array of ints"; return false; }
        for (const auto& x : j) {
            if (!x.is_number_integer()) {
                err = "slot_coverage: classes must be an array of ints"; return false;
            }
            if (x.is_number_unsigned()) {
                const auto v = x.get<uint64_t>();
                if (v > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                    err = "slot_coverage: classes must be an array of ints"; return false;
                }
                out.push_back(static_cast<int>(v));
            } else {
                const auto v = x.get<int64_t>();
                if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
                    err = "slot_coverage: classes must be an array of ints"; return false;
                }
                out.push_back(static_cast<int>(v));
            }
        }
        return true;
    }
    static bool ratio(const Json& j, const char* key, double& out, std::string& err) {
        if (!finite_num(j) || j.get<double>() < 0 || j.get<double>() > 1) {
            err = std::string("slot_coverage: ") + key + " must be finite in [0,1]"; return false;
        }
        out = j.get<double>(); return true;
    }
    static bool hold(const Json& j, const char* key, double& out, std::string& err) {
        if (!finite_num(j) || j.get<double>() < 0) {
            err = std::string("slot_coverage: ") + key + " must be finite and >= 0"; return false;
        }
        out = j.get<double>(); return true;
    }
    static bool parse_hold(const Json& j, const char* key, double& out, const double inherited,
                           std::string& err) {
        if (!j.contains(key)) { out = inherited; return true; }
        if (j.at(key).is_null()) { out = inherited; return true; }
        return hold(j.at(key), key, out, err);
    }
    static bool parse(const Json& j, Config& c, std::string& err) {
        if (!j.is_object() || !j.contains("slots") || !j.at("slots").is_array() ||
            j.at("slots").empty() || j.at("slots").size() > 64) {
            err = "slot_coverage: slots must be a non-empty array (1..64)"; return false;
        }
        double def_occ = .35, def_free = .15, def_hold = 3.0, def_fh = 3.0;
        if (j.contains("occupied_ratio") && !ratio(j.at("occupied_ratio"), "occupied_ratio", def_occ, err)) return false;
        if (j.contains("free_ratio") && !ratio(j.at("free_ratio"), "free_ratio", def_free, err)) return false;
        if (def_free >= def_occ) { err = "slot_coverage: free_ratio must be < occupied_ratio"; return false; }
        if (j.contains("hold_s") && !hold(j.at("hold_s"), "hold_s", def_hold, err)) return false;
        if (j.contains("occupied_hold_s") && !j.contains("hold_s") &&
            !hold(j.at("occupied_hold_s"), "occupied_hold_s", def_hold, err)) return false;
        if (!parse_hold(j, "free_hold_s", def_fh, def_hold, err)) return false;
        if (j.contains("classes") && !parse_classes(j.at("classes"), c.classes, err)) return false;
        for (const auto& x : j.at("slots")) {
            if (!x.is_object() || !x.contains("id") || !x.contains("polygon") || !x.at("id").is_string()) {
                err = "slot_coverage: each slot needs id and polygon"; return false;
            }
            Slot s; s.id = x.at("id").get<std::string>(); s.occupied = def_occ; s.free = def_free;
            s.occupied_hold = def_hold; s.free_hold = def_fh;
            if (!valid_id(s.id)) { err = "slot_coverage: invalid slot id"; return false; }
            for (const auto& old : c.slots) if (old.id == s.id) { err = "slot_coverage: duplicate slot id"; return false; }
            const auto& p = x.at("polygon");
            if (!p.is_array() || p.size() < 3 || p.size() > 16) { err = "slot_coverage: polygon must have 3..16 vertices"; return false; }
            for (const auto& q : p) {
                if (!q.is_array() || q.size() != 2 || !finite_num(q[0]) || !finite_num(q[1]) ||
                    q[0].get<double>() < 0 || q[0].get<double>() > 1 || q[1].get<double>() < 0 || q[1].get<double>() > 1) {
                    err = "slot_coverage: polygon coords must be finite in [0,1]"; return false;
                }
                s.poly.push_back({q[0].get<float>(), q[1].get<float>()});
            }
            std::string polygon_err;
            if (!valid_convex_polygon(s.poly, &polygon_err)) {
                const bool degenerate = polygon_err.find("area is too small") != std::string::npos ||
                                        polygon_err.find("duplicate vertices") != std::string::npos;
                err = "slot " + s.id + ": " + (degenerate ? "degenerate polygon" : "polygon not convex");
                return false;
            }
            if (x.contains("occupied_ratio") && !ratio(x.at("occupied_ratio"), "occupied_ratio", s.occupied, err)) return false;
            if (x.contains("free_ratio") && !ratio(x.at("free_ratio"), "free_ratio", s.free, err)) return false;
            if (s.free >= s.occupied) { err = "slot_coverage: free_ratio must be < occupied_ratio"; return false; }
            if (x.contains("hold_s") && !hold(x.at("hold_s"), "hold_s", s.occupied_hold, err)) return false;
            if (x.contains("occupied_hold_s") && !x.contains("hold_s") &&
                !hold(x.at("occupied_hold_s"), "occupied_hold_s", s.occupied_hold, err)) return false;
            if (x.contains("hold_s")) s.free_hold = s.occupied_hold;
            if (x.contains("free_hold_s")) {
                if (x.at("free_hold_s").is_null()) s.free_hold = s.occupied_hold;
                else if (!hold(x.at("free_hold_s"), "free_hold_s", s.free_hold, err)) return false;
            }
            if (x.contains("classes") && !parse_classes(x.at("classes"), s.classes, err)) return false;
            c.slots.push_back(std::move(s));
        }
        return true;
    }
    static bool same_polygon(const Polygon& a, const Polygon& b) {
        if (a.size() != b.size()) return false;
        const size_t n = a.size();
        for (int dir : {1, -1}) for (size_t start = 0; start < n; ++start) {
            bool ok = true;
            for (size_t k = 0; k < n; ++k) {
                size_t j = (dir == 1) ? (start + k) % n : (start + n - k) % n;
                if (std::fabs(a[k][0] - b[j][0]) > 1e-6 || std::fabs(a[k][1] - b[j][1]) > 1e-6) { ok = false; break; }
            }
            if (ok) return true;
        }
        return false;
    }
    bool configure(const std::string& text, std::string& err) override {
        Config next; Json j;
        try { j = Json::parse(text); } catch (const std::exception& e) { err = std::string("slot_coverage: invalid json: ") + e.what(); return false; }
        try {
            if (!parse(j, next, err)) return false;
        } catch (const std::exception&) {
            err = "slot_coverage: invalid field type";
            return false;
        }
        std::vector<State> ns(next.slots.size());
        for (size_t i = 0; i < next.slots.size(); ++i) for (size_t k = 0; k < cfg_.slots.size(); ++k)
            if (next.slots[i].id == cfg_.slots[k].id && same_polygon(next.slots[i].poly, cfg_.slots[k].poly) && k < state_.size()) ns[i] = state_[k];
        cfg_ = std::move(next); state_ = std::move(ns); return true;
    }
    static bool class_ok(const std::vector<int>& classes, int id) {
        return classes.empty() || std::find(classes.begin(), classes.end(), id) != classes.end();
    }
    void on_frame(const FrameMeta& m, const std::vector<Track>& tracks, float*, std::vector<AnalyzerEvent>& out) override {
        for (size_t i = 0; i < cfg_.slots.size(); ++i) {
            const Slot& s = cfg_.slots[i]; float best = 0; uint32_t best_id = 0;
            for (const auto& tr : tracks) {
                if (tr.misses != 0 || !class_ok(s.classes.empty() ? cfg_.classes : s.classes, tr.det.class_id)) continue;
                float cx, cy, w, h; m.geom.box_to_source_norm(tr.det.cx, tr.det.cy, tr.det.w, tr.det.h, cx, cy, w, h);
                const float x0 = std::max(0.0f, cx - w * .5f), x1 = std::min(1.0f, cx + w * .5f);
                const float box_y0 = cy - h * .5f, box_y1 = cy + h * .5f;
                const float y0 = std::max(0.0f, box_y0), y1 = std::min(1.0f, box_y1);
                const float lower_y0 = y0 + (y1 - y0) * .5f;
                const float lower_y1 = y1;
                float cov = polygon_area(s.poly) > 0 ? polygon_intersection_area_rect(s.poly, x0, lower_y0, x1, lower_y1) / polygon_area(s.poly) : 0;
                if (cov > best) { best = cov; best_id = tr.track_id; }
            }
            step(i, best, best_id, m.t_mono_s, out);
        }
    }
    void step(size_t i, float cov, uint32_t tid, double now, std::vector<AnalyzerEvent>& out) {
        Slot& s = cfg_.slots[i]; State& st = state_[i];
        std::string want = st.value;
        if (st.value == "unknown") want = cov >= s.occupied ? "occupied" : "free";
        else if (st.value == "free") want = cov >= s.occupied ? "occupied" : "free";
        else want = cov <= s.free ? "free" : "occupied";
        if (st.value == "unknown") { emit(i, st.value, want, cov, want == "free" ? 0 : tid, out); st.value = want; st.since = now; st.pending.clear(); st.track_id = want == "free" ? 0 : tid; return; }
        if (want == st.value) { st.pending.clear(); st.track_id = st.value == "free" ? 0 : tid; return; }
        if (st.pending != want) { st.pending = want; st.pending_since = now; }
        double hold_s = want == "occupied" ? s.occupied_hold : s.free_hold;
        if (now >= st.pending_since && now - st.pending_since >= hold_s) {
            emit(i, st.value, want, cov, want == "free" ? 0 : tid, out); st.value = want; st.since = now; st.pending.clear(); st.track_id = want == "free" ? 0 : tid;
        }
    }
    void emit(size_t i, const std::string& prev, const std::string& state, float cov, uint32_t tid, std::vector<AnalyzerEvent>& out) {
        Json f{{"slot_id", cfg_.slots[i].id}, {"prev_state", prev}, {"state", state}, {"coverage", cov}};
        AnalyzerEvent e; e.type = "slot_change"; e.track_id = tid; e.fields_json = f.dump(); out.push_back(std::move(e));
    }
    void on_track_removed(uint32_t, double, std::vector<AnalyzerEvent>&) override {}
    Config cfg_; std::vector<State> state_;
};
} // namespace

std::unique_ptr<Analyzer> make_slot_coverage_analyzer() { return std::make_unique<SlotAnalyzer>(); }
} // namespace vb
