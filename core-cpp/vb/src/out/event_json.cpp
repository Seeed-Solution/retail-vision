// vb.event/1 / vb.frame/1 JSON construction (spec BASE-1 §6.10.2, M1.18).
#include "out/event_json.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "vb/letterbox.h"
#include "vb/types.h"

namespace vb {

namespace {

// §6.10.2 rule 1: floats are rounded to 6 decimals, the same rule as Python
// round(x, 6) in vision_base.apps. %.6g (6 significant digits) only agrees with
// it on [0.1, 1) and diverged on inference_ms.
Json num6(double v) {
    char buf[352];  // %.6f of the largest double fits in 317 bytes
    std::snprintf(buf, sizeof buf, "%.6f", v);
    return Json(std::strtod(buf, nullptr));
}

// §6.10.2 rule 1 ("fields 内层递归应用同一规则"): recurse through objects and
// arrays, rounding floats only. Integers, booleans and strings keep their type
// (rounding a bool or int would change the JSON type and the payload contract).
Json round6_deep(const Json& v) {
    if (v.is_number_float()) return num6(v.get<double>());
    if (v.is_object()) {
        Json o = Json::object();
        for (auto it = v.begin(); it != v.end(); ++it)
            o[it.key()] = round6_deep(it.value());
        return o;
    }
    if (v.is_array()) {
        Json a = Json::array();
        for (const auto& e : v) a.push_back(round6_deep(e));
        return a;
    }
    return v;
}

// §6.10.2 rule 3: source-normalized box/keypoint coordinates are clipped to
// [0, 1]. With letterbox padding a full-canvas model box inverse-maps outside
// the frame (1280x720 -> 640x640 gives y = [-0.389, 1.389]); that is a padding
// artefact, not picture content, and vb-frame.schema.json caps coordinates at
// [0, 1]. Clipping runs before the 6-decimal rounding so exact 0.0/1.0 is
// emitted instead of a signed zero or a 1.0000000001-style boundary value.
double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

// §6.10.2 rule 4: wall_ms uses Python round() semantics (half to even). The
// default FP rounding mode gives nearbyint exactly that; llround rounds half
// away from zero, so 1000.5 became 1001 where Python gives 1000.
int64_t round_even(double v) {
    return static_cast<int64_t>(std::nearbyint(v));
}

// §6.10.2 rule 2: the letterbox inverse transform runs in double, matching
// Python's float64. float32 intermediates diverge by 1 ulp at the 6th-decimal
// rounding boundary (measured 0.441697 vs 0.441698). Same operation order as
// vision_base.letterbox.to_source_norm / box_to_source_norm.
struct InvGeom {
    double src_w, src_h, model_w, model_h, scale, pad_x, pad_y;

    explicit InvGeom(const LetterboxGeom& g)
        : src_w(g.src_w), src_h(g.src_h), model_w(g.model_w), model_h(g.model_h),
          scale(g.scale), pad_x(g.pad_x), pad_y(g.pad_y) {}

    double x(double mx) const { return ((mx * model_w - pad_x) / scale) / src_w; }
    double y(double my) const { return ((my * model_h - pad_y) / scale) / src_h; }
    double w(double mw) const { return mw * model_w / (scale * src_w); }
    double h(double mh) const { return mh * model_h / (scale * src_h); }
};

}  // namespace

Json event_json(const std::string& device_id, const std::string& stream_id,
                const Json& body) {
    Json j;
    j["schema"] = "vb.event/1";
    j["device_id"] = device_id;
    j["stream_id"] = stream_id;
    j["seq"] = body.value("seq", uint64_t(0));
    j["ts_ms"] = round_even(body.value("wall_ms", 0.0));
    j["analyzer"] = body.value("analyzer", std::string());
    j["type"] = body.value("type", std::string());
    j["track_id"] = body.value("track_id", uint32_t(0));
    j["fields"] = round6_deep(body.contains("fields") ? body.at("fields")
                                                       : Json::object());
    return j;
}

Json frame_json(const std::string& device_id, const std::string& stream_id,
                const WireFrameRec& r) {
    Json j;
    j["schema"] = "vb.frame/1";
    j["device_id"] = device_id;
    j["stream_id"] = stream_id;
    j["seq"] = r.seq;
    j["ts_ms"] = round_even(r.wall_ms);
    j["inference_ms"] = num6(r.inference_ms);

    LetterboxGeom g;
    g.src_w = r.src_w;
    g.src_h = r.src_h;
    g.model_w = r.model_w;
    g.model_h = r.model_h;
    g.scale = r.scale;
    g.pad_x = r.pad_x;
    g.pad_y = r.pad_y;
    g.align = static_cast<Align>(r.align);
    const InvGeom ig(g);

    Json dets = Json::array();
    const size_t n = r.dets.size();
    const size_t kn = r.kpt_per_det;
    const size_t an = r.attr_per_det;
    for (size_t i = 0; i < n; ++i) {
        const WireDet& d = r.dets[i];
        const double scx = ig.x(d.cx), scy = ig.y(d.cy);
        const double sw = ig.w(d.w), sh = ig.h(d.h);
        Json dj;
        dj["track_id"] = d.track_id;
        dj["class_id"] = d.class_id;
        dj["score"] = num6(d.score);
        Json box = Json::array();
        box.push_back(num6(clamp01(scx - sw / 2.0)));
        box.push_back(num6(clamp01(scy - sh / 2.0)));
        box.push_back(num6(clamp01(scx + sw / 2.0)));
        box.push_back(num6(clamp01(scy + sh / 2.0)));
        dj["box"] = box;
        if (kn > 0) {
            Json kps = Json::array();
            for (size_t k = 0; k < kn; ++k) {
                const double mx = r.kpts[(i * kn + k) * 3];
                const double my = r.kpts[(i * kn + k) * 3 + 1];
                const double mc = r.kpts[(i * kn + k) * 3 + 2];
                Json kp = Json::array();
                kp.push_back(num6(clamp01(ig.x(mx))));
                kp.push_back(num6(clamp01(ig.y(my))));
                kp.push_back(num6(mc));  // confidence is not a coordinate
                kps.push_back(kp);
            }
            dj["keypoints"] = kps;
        }
        if (an > 0) {
            // Attribute names are not carried on VBR1 (hello attr_names is
            // empty for all built-in analyzers); emit positional names.
            Json attrs = Json::object();
            for (size_t a = 0; a < an; ++a)
                attrs["attr_" + std::to_string(a)] = num6(r.attrs[i * an + a]);
            dj["attrs"] = attrs;
        }
        dets.push_back(dj);
    }
    j["detections"] = dets;
    return j;
}

}  // namespace vb
