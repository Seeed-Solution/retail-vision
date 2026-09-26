// Shared external-fixture runner for analyzers (spec BASE-1 §6.2.4, M1.6).
// Used by both vb_selftest (`analyzer` subcommand) and test_analyzers.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "vb/analyzer.h"
#include "vb/json.h"
#include "vb/letterbox.h"

namespace vb_fixture {

using vb::Analyzer;
using vb::AnalyzerEvent;
using vb::Json;
using vb::Track;

struct Result {
    bool ok = false;
    std::string fail;
};

static vb::LetterboxGeom geom_from_json(const Json& j) {
    int src_w = j.value("src_w", 1000), src_h = j.value("src_h", 1000);
    int model_w = j.value("model_w", 1000), model_h = j.value("model_h", 1000);
    vb::Align a = j.value("align", std::string("center")) == "center" ? vb::Align::Center
                                                                      : vb::Align::TopLeft;
    return vb::letterbox_fit(src_w, src_h, model_w, model_h, a);
}

struct Emitted {
    uint64_t seq;
    AnalyzerEvent ev;
};

// Feed one fixture frame list through the analyzer and collect the events,
// tagging each with the frame seq it was emitted on.
static bool run_frames(Analyzer& a, const Json& j, const vb::LetterboxGeom& geom,
                       std::vector<Emitted>& out, std::string& fail) {
    std::vector<std::pair<uint64_t, Json>> reconf;  // (before_seq, config)
    for (const auto& r : j.value("reconfigure", Json::array())) {
        reconf.push_back({r.at("before_seq").get<uint64_t>(), r});
    }
    for (const auto& f : j.at("frames")) {
        uint64_t seq = f.at("seq").get<uint64_t>();
        double t = f.value("t", 0.0);
        for (auto& rc : reconf) {
            if (rc.first == seq) {
                std::string err;
                bool ok = a.configure(rc.second.at("config").dump(), err);
                const std::string expect_err =
                    rc.second.value("expect_error", Json()).is_null()
                        ? ""
                        : rc.second.at("expect_error").get<std::string>();
                if (expect_err.empty()) {
                    if (!ok) {
                        fail = "seq " + std::to_string(seq) +
                               ": reconfigure failed unexpectedly: " + err;
                        return false;
                    }
                } else if (ok || err.find(expect_err) == std::string::npos) {
                    fail = "seq " + std::to_string(seq) +
                           ": reconfigure error mismatch: got ok=" + (ok ? "true" : "false") +
                           " err=" + err;
                    return false;
                }
            }
        }
        vb::FrameMeta m;
        m.stream_index = 0;
        m.seq = seq;
        m.wall_ms = t * 1000.0;
        m.t_mono_s = t;
        m.geom = geom;
        std::vector<Track> tracks;
        for (const auto& tj : f.value("tracks", Json::array())) {
            Track tr;
            tr.track_id = tj.at("track_id").get<uint32_t>();
            tr.det.cx = tj.at("cx").get<float>();
            tr.det.cy = tj.at("cy").get<float>();
            tr.det.w = tj.value("w", 0.0f);
            tr.det.h = tj.value("h", 0.0f);
            tr.det.score = tj.value("score", 0.0f);
            tr.det.class_id = tj.value("class_id", 0);
            tr.vx = tj.value("vx", 0.0f);
            tr.vy = tj.value("vy", 0.0f);
            tr.hits = tj.value("hits", 1u);
            tr.misses = tj.value("misses", 0u);
            tr.first_seen_s = tj.value("first_seen", t);
            tr.last_seen_s = tj.value("last_seen", t);
            for (const auto& k : tj.value("kpts", Json::array())) {
                vb::Keypoint kp;
                kp.x = k.at(0).get<float>();
                kp.y = k.at(1).get<float>();
                kp.conf = k.at(2).get<float>();
                tr.kpts.push_back(kp);
            }
            tracks.push_back(std::move(tr));
        }
        std::vector<float> attrs(tracks.size() * a.attr_count(), 0.0f);
        std::vector<AnalyzerEvent> evs;
        a.on_frame(m, tracks, attrs.empty() ? nullptr : attrs.data(), evs);
        for (auto& e : evs) out.push_back({seq, std::move(e)});
        for (uint32_t rid : f.value("removed", Json::array())) {
            std::vector<AnalyzerEvent> revs;
            a.on_track_removed(rid, t, revs);
            for (auto& e : revs) out.push_back({seq, std::move(e)});
        }
    }
    return true;
}

static bool compare_events(const std::vector<Emitted>& got, const Json& expect,
                           std::string& fail) {
    size_t n = expect.size();
    if (got.size() != n) {
        fail = "event count: expected " + std::to_string(n) + ", got " +
               std::to_string(got.size());
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        const auto& e = expect[i];
        const Emitted& g = got[i];
        uint64_t seq = e.at("seq").get<uint64_t>();
        if (g.seq != seq) {
            fail = "event " + std::to_string(i) + ": seq expected " + std::to_string(seq) +
                   ", got " + std::to_string(g.seq);
            return false;
        }
        if (g.ev.type != e.at("type").get<std::string>()) {
            fail = "event " + std::to_string(i) + " (seq " + std::to_string(seq) +
                   "): type expected " + e.at("type").get<std::string>() + ", got " + g.ev.type;
            return false;
        }
        uint32_t tid = e.value("track_id", 0u);
        if (g.ev.track_id != tid) {
            fail = "event " + std::to_string(i) + " (seq " + std::to_string(seq) +
                   "): track_id expected " + std::to_string(tid) + ", got " +
                   std::to_string(g.ev.track_id);
            return false;
        }
        Json gf = Json::parse(g.ev.fields_json);
        const Json& ef = e.at("fields");
        if (!gf.is_object() || gf.size() != ef.size()) {
            fail = "event " + std::to_string(i) + " (seq " + std::to_string(seq) +
                   "): fields key set differs: expected " + ef.dump() + ", got " + gf.dump();
            return false;
        }
        for (auto it = ef.begin(); it != ef.end(); ++it) {
            auto git = gf.find(it.key());
            if (git == gf.end()) {
                fail = "event " + std::to_string(i) + " (seq " + std::to_string(seq) +
                       "): missing field " + it.key();
                return false;
            }
            if (it->is_array() && git->is_array()) {
                if (it->size() != git->size()) {
                    fail = "event " + std::to_string(i) + " (seq " + std::to_string(seq) +
                           "): field " + it.key() + " expected " + it->dump() + ", got " +
                           git->dump();
                    return false;
                }
                for (size_t k = 0; k < it->size(); ++k) {
                    if (!(*git)[k].is_number() || !(*it)[k].is_number() ||
                        std::fabs((*git)[k].get<double>() - (*it)[k].get<double>()) > 1e-6) {
                        fail = "event " + std::to_string(i) + " (seq " + std::to_string(seq) +
                               "): field " + it.key() + " expected " + it->dump() + ", got " +
                               git->dump();
                        return false;
                    }
                }
            } else if (it->is_number() && git->is_number()) {
                if (std::fabs(it->get<double>() - git->get<double>()) > 1e-6) {
                    fail = "event " + std::to_string(i) + " (seq " + std::to_string(seq) +
                           "): field " + it.key() + " expected " + it->dump() + ", got " +
                           git->dump();
                    return false;
                }
            } else if (*git != *it) {
                fail = "event " + std::to_string(i) + " (seq " + std::to_string(seq) +
                       "): field " + it.key() + " expected " + it->dump() + ", got " +
                       git->dump();
                return false;
            }
        }
    }
    return true;
}

// Run a whole §6.2.4 fixture document against a configured-on-demand analyzer.
static Result run_analyzer_fixture(Analyzer& a, const Json& j) {
    Result r;
    const Json cfg = j.value("config", Json::object());
    const Json ece = j.value("expect_configure_error", Json());
    std::string err;
    bool ok = a.configure(cfg.dump(), err);
    if (!ece.is_null()) {
        std::string want = ece.get<std::string>();
        if (ok || err.find(want) == std::string::npos) {
            r.fail = "configure error mismatch: got ok=" + std::string(ok ? "true" : "false") +
                     " err=" + err + " (want substring \"" + want + "\")";
            return r;
        }
        r.ok = true;
        return r;
    }
    if (!ok) {
        r.fail = "configure failed: " + err;
        return r;
    }
    vb::LetterboxGeom geom =
        geom_from_json(j.value("geom", Json::object()));
    std::vector<Emitted> got;
    if (!run_frames(a, j, geom, got, r.fail)) return r;
    if (!compare_events(got, j.value("expect_events", Json::array()), r.fail)) return r;
    r.ok = true;
    return r;
}

}  // namespace vb_fixture
