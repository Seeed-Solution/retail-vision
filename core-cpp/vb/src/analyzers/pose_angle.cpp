// Built-in pose_angle analyzer (spec BASE-1 §6.2.5, M1.17): joint angle
// limits from COCO-17 keypoints. The angle is ABC at B, 0..180 deg,
// computed in source *pixel* coords (keypoints are model-canvas normalized
// and letterbox back-transformed first). Events:
//   angle_out {joint_id, angle_deg, min_deg, max_deg, class_id, score}
//   angle_in  {joint_id, angle_deg}
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "analyzers/motion_util.h"
#include "analyzers/pose_angle.h"

namespace vb {
namespace {

struct Joint {
    std::string id;
    int a = 0, b = 0, c = 0;
    bool has_min = false, has_max = false;
    double min_deg = 0, max_deg = 0;
};

struct PoseAngleAnalyzer : public Analyzer {
    struct Cfg {
        std::vector<Joint> joints;
        double min_conf = 0.3;
        double hold_s = 0.5;
        std::vector<int> classes;
        int max_idx = -1;  // max(a,b,c) over joints
    };
    struct Key {
        uint32_t track_id;
        std::string joint;
        bool operator<(const Key& o) const {
            return track_id != o.track_id ? track_id < o.track_id : joint < o.joint;
        }
    };

    const char* name() const override { return "pose_angle"; }
    // Runtime rejects add when caps.keypoints < min_keypoints().
    uint32_t min_keypoints() const override {
        return cfg_.max_idx < 0 ? 0 : static_cast<uint32_t>(cfg_.max_idx + 1);
    }

    bool parse_cfg(const Json& j, Cfg& c, std::string& err) {
        if (!j.is_object()) {
            err = "pose_angle: config must be an object";
            return false;
        }
        if (!parse_classes(j, "pose_angle", c.classes, err)) return false;
        auto f = j.find("joints");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_array()) {
                err = "pose_angle: joints must be an array";
                return false;
            }
            for (const auto& jj : *f) {
                if (!jj.is_object() || !jj.contains("id") || !jj.contains("a") ||
                    !jj.contains("b") || !jj.contains("c")) {
                    err = "pose_angle: each joint needs id, a, b, c";
                    return false;
                }
                Joint jt;
                jt.id = jj.at("id").get<std::string>();
                if (jt.id.empty() || jt.id.size() > 32) {
                    err = "pose_angle: joint id must be 1..32 chars";
                    return false;
                }
                for (const char* k : {"a", "b", "c"}) {
                    if (!jj.at(k).is_number_integer() || jj.at(k).get<int>() < 0) {
                        err = std::string("pose_angle: joint ") + jt.id + ": " + k +
                              " must be a non-negative int";
                        return false;
                    }
                }
                jt.a = jj.at("a").get<int>();
                jt.b = jj.at("b").get<int>();
                jt.c = jj.at("c").get<int>();
                auto lim = [&](const char* k, double& v, bool& has) -> bool {
                    auto it = jj.find(k);
                    if (it == jj.end() || it->is_null()) return true;
                    if (!it->is_number() || it->get<double>() < 0 ||
                        it->get<double>() > 180) {
                        err = std::string("pose_angle: joint ") + jt.id + ": " + k +
                              " must be in [0,180]";
                        return false;
                    }
                    v = it->get<double>();
                    has = true;
                    return true;
                };
                if (!lim("min_deg", jt.min_deg, jt.has_min)) return false;
                if (!lim("max_deg", jt.max_deg, jt.has_max)) return false;
                if (!jt.has_min && !jt.has_max) {
                    err = "pose_angle: joint " + jt.id +
                          " needs at least one of min_deg / max_deg";
                    return false;
                }
                c.max_idx = std::max({c.max_idx, jt.a, jt.b, jt.c});
                c.joints.push_back(std::move(jt));
            }
        }
        f = j.find("min_conf");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0 || f->get<double>() > 1) {
                err = "pose_angle: min_conf must be in [0,1]";
                return false;
            }
            c.min_conf = f->get<double>();
        }
        f = j.find("hold_s");
        if (f != j.end() && !f->is_null()) {
            if (!f->is_number() || f->get<double>() < 0) {
                err = "pose_angle: hold_s must be >= 0";
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
            err = std::string("pose_angle: invalid json: ") + e.what();
            return false;
        }
        Cfg c;
        if (!parse_cfg(j, c, err)) return false;
        cfg_ = std::move(c);
        deb_.clear();
        return true;
    }

    void on_frame(const FrameMeta& m, const std::vector<Track>& tracks, float* attrs,
                  std::vector<AnalyzerEvent>& out) override {
        (void)attrs;
        const double now = m.t_mono_s;
        for (const auto& tr : tracks) {
            if (tr.misses != 0) continue;
            if (!class_ok(cfg_.classes, tr.det.class_id)) continue;
            for (const auto& jt : cfg_.joints) {
                if (static_cast<int>(tr.kpts.size()) <= jt.a ||
                    static_cast<int>(tr.kpts.size()) <= jt.b ||
                    static_cast<int>(tr.kpts.size()) <= jt.c)
                    continue;
                const Keypoint &A = tr.kpts[jt.a], &B = tr.kpts[jt.b], &C = tr.kpts[jt.c];
                if (A.conf < cfg_.min_conf || B.conf < cfg_.min_conf ||
                    C.conf < cfg_.min_conf)
                    continue;  // low confidence: no state change
                float sax, say, sbx, sby, scx, scy;
                m.geom.to_source_norm(A.x, A.y, sax, say);
                m.geom.to_source_norm(B.x, B.y, sbx, sby);
                m.geom.to_source_norm(C.x, C.y, scx, scy);
                double ax = sax * m.geom.src_w, ay = say * m.geom.src_h;
                double bx = sbx * m.geom.src_w, by = sby * m.geom.src_h;
                double cx = scx * m.geom.src_w, cy = scy * m.geom.src_h;
                double v1x = ax - bx, v1y = ay - by, v2x = cx - bx, v2y = cy - by;
                double l1 = std::sqrt(v1x * v1x + v1y * v1y);
                double l2 = std::sqrt(v2x * v2x + v2y * v2y);
                if (l1 < 1e-6 || l2 < 1e-6) continue;
                double cosang = (v1x * v2x + v1y * v2y) / (l1 * l2);
                if (cosang > 1) cosang = 1;
                if (cosang < -1) cosang = -1;
                double ang = std::acos(cosang) * 180.0 / M_PI;
                std::string raw = "in";
                if ((jt.has_min && ang < jt.min_deg) || (jt.has_max && ang > jt.max_deg))
                    raw = "out";
                auto it2 = deb_.try_emplace(Key{tr.track_id, jt.id}, "in");
                Debounce& d = it2.first->second;
                std::string prev, next;
                if (!d.step(raw, now, cfg_.hold_s, prev, next)) continue;
                if (next == "out") {
                    emit_event(out, "angle_out", tr.track_id,
                               Json{{"joint_id", jt.id},
                                    {"angle_deg", ang},
                                    {"min_deg", jt.has_min ? Json(jt.min_deg) : Json()},
                                    {"max_deg", jt.has_max ? Json(jt.max_deg) : Json()},
                                    {"class_id", tr.det.class_id},
                                    {"score", tr.det.score}});
                } else {  // ("out", "in")
                    emit_event(out, "angle_in", tr.track_id,
                               Json{{"joint_id", jt.id}, {"angle_deg", ang}});
                }
            }
        }
    }

    void on_track_removed(uint32_t track_id, double t_mono_s,
                          std::vector<AnalyzerEvent>& out) override {
        (void)t_mono_s;
        (void)out;
        for (auto it = deb_.begin(); it != deb_.end();) {
            if (it->first.track_id == track_id)
                it = deb_.erase(it);
            else
                ++it;
        }
    }

    Cfg cfg_;
    std::map<Key, Debounce> deb_;
};

}  // namespace

std::unique_ptr<Analyzer> make_pose_angle_analyzer() {
    return std::make_unique<PoseAngleAnalyzer>();
}

}  // namespace vb
