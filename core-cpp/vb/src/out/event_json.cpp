// vb.event/1 / vb.frame/1 JSON construction (spec BASE-1 §6.10.2, M1.18).
#include "out/event_json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "vb/letterbox.h"
#include "vb/types.h"

namespace vb {

namespace {

// §6.10.2 consistency rule: floats are emitted with %.6g precision so the
// C++ and Python (round(x, 6)) outputs compare equal.
Json num6(double v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return Json(std::strtod(buf, nullptr));
}

}  // namespace

Json event_json(const std::string& device_id, const std::string& stream_id,
                const Json& body) {
    Json j;
    j["schema"] = "vb.event/1";
    j["device_id"] = device_id;
    j["stream_id"] = stream_id;
    j["seq"] = body.value("seq", uint64_t(0));
    j["ts_ms"] = static_cast<int64_t>(std::llround(body.value("wall_ms", 0.0)));
    j["analyzer"] = body.value("analyzer", std::string());
    j["type"] = body.value("type", std::string());
    j["track_id"] = body.value("track_id", uint32_t(0));
    j["fields"] = body.value("fields", Json::object());
    return j;
}

Json frame_json(const std::string& device_id, const std::string& stream_id,
                const WireFrameRec& r) {
    Json j;
    j["schema"] = "vb.frame/1";
    j["device_id"] = device_id;
    j["stream_id"] = stream_id;
    j["seq"] = r.seq;
    j["ts_ms"] = static_cast<int64_t>(std::llround(r.wall_ms));
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

    Json dets = Json::array();
    const size_t n = r.dets.size();
    const size_t kn = r.kpt_per_det;
    const size_t an = r.attr_per_det;
    for (size_t i = 0; i < n; ++i) {
        const WireDet& d = r.dets[i];
        float scx, scy, sw, sh;
        g.box_to_source_norm(d.cx, d.cy, d.w, d.h, scx, scy, sw, sh);
        Json dj;
        dj["track_id"] = d.track_id;
        dj["class_id"] = d.class_id;
        dj["score"] = num6(d.score);
        Json box = Json::array();
        box.push_back(num6(scx - sw / 2));
        box.push_back(num6(scy - sh / 2));
        box.push_back(num6(scx + sw / 2));
        box.push_back(num6(scy + sh / 2));
        dj["box"] = box;
        if (kn > 0) {
            Json kps = Json::array();
            for (size_t k = 0; k < kn; ++k) {
                float mx = r.kpts[(i * kn + k) * 3];
                float my = r.kpts[(i * kn + k) * 3 + 1];
                float mc = r.kpts[(i * kn + k) * 3 + 2];
                float sx, sy;
                g.to_source_norm(mx, my, sx, sy);
                Json kp = Json::array();
                kp.push_back(num6(sx));
                kp.push_back(num6(sy));
                kp.push_back(num6(mc));
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
