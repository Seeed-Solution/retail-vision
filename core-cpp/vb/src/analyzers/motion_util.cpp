// Shared helpers for the §6.2.5 motion analyzers (spec BASE-1, M1.16/M1.17).
#include <algorithm>

#include "analyzers/motion_util.h"

namespace vb {

void anchor_point(const Track& t, const LetterboxGeom& g, const std::string& anchor,
                  float& x, float& y) {
    float mx = t.det.cx, my = t.det.cy;
    if (anchor == "center") {
        // box centre
    } else {  // bottom_center (default)
        my = t.det.cy + t.det.h * 0.5f;
    }
    g.to_source_norm(mx, my, x, y);
}

bool parse_zones(const Json& j, const char* who, std::vector<ZoneCfg>& out,
                 std::string& err) {
    out.clear();
    auto it = j.find("zones");
    if (it == j.end() || it->is_null()) return true;
    if (!it->is_array() || it->size() > 64) {
        err = std::string(who) + ": zones must be an array (0..64)";
        return false;
    }
    for (const auto& zj : *it) {
        if (!zj.is_object() || !zj.contains("id") || !zj.contains("polygon")) {
            err = std::string(who) + ": each zone needs id and polygon";
            return false;
        }
        ZoneCfg z;
        z.id = zj.at("id").get<std::string>();
        if (z.id.empty() || z.id.size() > 32) {
            err = std::string(who) + ": zone id must be 1..32 chars";
            return false;
        }
        for (const auto& o : out) {
            if (o.id == z.id) {
                err = std::string(who) + ": duplicate zone id " + z.id;
                return false;
            }
        }
        const auto& poly = zj.at("polygon");
        if (!poly.is_array() || poly.size() < 3 || poly.size() > 64) {
            err = std::string(who) + ": polygon must have 3..64 vertices";
            return false;
        }
        for (const auto& p : poly) {
            if (!p.is_array() || p.size() != 2 || !p[0].is_number() ||
                !p[1].is_number() || p[0].get<double>() < 0 ||
                p[0].get<double>() > 1 || p[1].get<double>() < 0 ||
                p[1].get<double>() > 1) {
                err = std::string(who) +
                      ": polygon coords must be in [0,1] (source-normalized)";
                return false;
            }
            z.poly.push_back({p[0].get<float>(), p[1].get<float>()});
        }
        out.push_back(std::move(z));
    }
    return true;
}

bool parse_classes(const Json& j, const char* who, std::vector<int>& out,
                   std::string& err) {
    out.clear();
    auto it = j.find("classes");
    if (it == j.end() || it->is_null()) return true;
    if (!it->is_array()) {
        err = std::string(who) + ": classes must be an array of ints";
        return false;
    }
    for (const auto& cl : *it) {
        if (!cl.is_number_integer()) {
            err = std::string(who) + ": classes must be an array of ints";
            return false;
        }
        out.push_back(cl.get<int>());
    }
    return true;
}

const ZoneCfg* zone_of(const std::vector<ZoneCfg>& zones, float x, float y,
                       std::string& zone_id) {
    zone_id = "";
    for (const auto& z : zones) {
        if (point_in_polygon(x, y, z.poly)) {
            zone_id = z.id;
            return &z;
        }
    }
    return nullptr;
}

bool parse_anchor(const Json& j, const char* who, std::string& out, std::string& err) {
    out = "bottom_center";
    auto it = j.find("anchor");
    if (it == j.end() || it->is_null()) return true;
    if (!it->is_string()) {
        err = std::string(who) + ": anchor must be a string";
        return false;
    }
    out = it->get<std::string>();
    if (out != "bottom_center" && out != "center") {
        err = std::string(who) + ": anchor must be \"bottom_center\" or \"center\"";
        return false;
    }
    return true;
}

}  // namespace vb
