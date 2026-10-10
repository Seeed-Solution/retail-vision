// Generic tracker implementation (spec BASE-1 §6.2, M1.5).
// Matching logic reproduces retail-vision core-cpp/person_tracker.cpp
// (IoU pass with velocity prediction -> centre-distance fallback for
// recently-lost tracks, older tracks preferred) minus dwell/zone/line/edge.
#include "vb/tracker.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vb {
namespace {

// Retail thresholds are expressed in px/s on a 640x640 canvas; the generic
// tracker works in normalized units per second, so divide by 640.
constexpr float kVelZero = 3.0f / 640.0f;      // below: velocity clamps to 0
constexpr float kVelDwellSpeed = 10.0f / 640.0f;
constexpr float kVelSuddenStartPrev = 5.0f / 640.0f;
constexpr float kVelSuddenStartInst = 50.0f / 640.0f;
constexpr float kVelAlpha = 0.08f;
constexpr float kVelAlphaSudden = 0.6f;
constexpr float kPredictMinSpeed = 1.0f / 640.0f;  // predict only if faster than this
constexpr int kFallbackMaxMisses = 5;              // centre-distance fallback window

float iou(const Detection& a, const Detection& b) {
    float ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
    float ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
    float bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
    float bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

}  // namespace

Tracker::Tracker(TrackerConfig c) : cfg_(c) {}

void Tracker::reset() {
    tracks_.clear();
    alive_.clear();
    next_id_ = 1;
    last_t_ = 0;
    has_t_ = false;
}

void Tracker::update_velocity(Track& tr, float ncx, float ncy, double dt) {
    if (dt <= 0.001) return;
    float dx = ncx - tr.det.cx, dy = ncy - tr.det.cy;
    float ivx = static_cast<float>(dx / dt), ivy = static_cast<float>(dy / dt);
    float inst = std::hypot(ivx, ivy);
    float prev = std::hypot(tr.vx, tr.vy);

    if (inst < kVelZero) {
        tr.vx = tr.vy = 0;
        return;
    }
    bool sudden_stop = prev > kVelDwellSpeed && (prev - inst) / prev > 0.5f;
    bool sudden_start = prev < kVelSuddenStartPrev && inst > kVelSuddenStartInst;
    float alpha = (sudden_stop || sudden_start) ? kVelAlphaSudden : kVelAlpha;
    tr.vx = (1 - alpha) * tr.vx + alpha * ivx;
    tr.vy = (1 - alpha) * tr.vy + alpha * ivy;
}

const std::vector<Track>& Tracker::update(std::vector<Detection>& dets,
                                          const std::vector<Keypoint>& kpts,
                                          double t_mono_s,
                                          std::vector<uint32_t>& removed) {
    removed.clear();
    double dt = has_t_ ? (t_mono_s - last_t_) : 0.0;
    last_t_ = t_mono_s;
    has_t_ = true;

    // Tracks ordered by descending hits (prefer older tracks), stable.
    std::vector<size_t> order(tracks_.size());
    for (size_t i = 0; i < tracks_.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [this](size_t a, size_t b) {
        return tracks_[a].hits > tracks_[b].hits;
    });

    std::vector<char> det_used(dets.size(), 0);
    std::vector<int> track_match(tracks_.size(), -1);  // det index or -1

    auto predicted = [t_mono_s](const Track& tr, float& px, float& py) {
        px = tr.det.cx;
        py = tr.det.cy;
        if (tr.misses > 0 && std::hypot(tr.vx, tr.vy) > kPredictMinSpeed) {
            float dts = static_cast<float>(t_mono_s - tr.last_seen_s);
            px += tr.vx * dts;
            py += tr.vy * dts;
        }
    };

    auto class_ok = [this](const Detection& a, const Detection& b) {
        return !cfg_.class_aware || a.class_id == b.class_id;
    };

    // Pass 1: IoU on velocity-predicted boxes.
    std::vector<size_t> unmatched;
    for (size_t oi : order) {
        Track& tr = tracks_[oi];
        float px, py;
        predicted(tr, px, py);
        Detection pred = tr.det;
        pred.cx = px;
        pred.cy = py;
        float best_iou = cfg_.iou_threshold;
        int best = -1;
        for (size_t d = 0; d < dets.size(); ++d) {
            if (det_used[d]) continue;
            if (!class_ok(tr.det, dets[d])) continue;
            float v = iou(pred, dets[d]);
            if (v > best_iou) {
                best_iou = v;
                best = static_cast<int>(d);
            }
        }
        if (best >= 0) {
            det_used[best] = 1;
            track_match[oi] = best;
        } else {
            unmatched.push_back(oi);
        }
    }

    // Pass 2: centre-distance fallback for recently lost tracks.
    // The reference point depends on cfg_.anchor (§6.2): Center = box
    // centre, BottomCenter = bottom-edge midpoint (ground point). IoU in
    // pass 1 is unaffected by the anchor choice.
    auto anchor_xy = [this](const Detection& d, float& ax, float& ay) {
        ax = d.cx;
        ay = (cfg_.anchor == TrackerConfig::BottomCenter) ? d.cy + d.h / 2 : d.cy;
    };
    for (size_t oi : unmatched) {
        Track& tr = tracks_[oi];
        if (static_cast<int>(tr.misses) > kFallbackMaxMisses) continue;
        float px, py;
        predicted(tr, px, py);
        Detection pred = tr.det;
        pred.cx = px;
        pred.cy = py;
        float pax, pay;
        anchor_xy(pred, pax, pay);
        float best_dist = cfg_.dist_threshold;
        int best = -1;
        for (size_t d = 0; d < dets.size(); ++d) {
            if (det_used[d]) continue;
            if (!class_ok(tr.det, dets[d])) continue;
            float dax, day;
            anchor_xy(dets[d], dax, day);
            float dist = std::hypot(dax - pax, day - pay);
            if (dist < best_dist) {
                best_dist = dist;
                best = static_cast<int>(d);
            }
        }
        if (best >= 0) {
            det_used[best] = 1;
            track_match[oi] = best;
        }
    }

    // Update matched tracks.
    for (size_t i = 0; i < tracks_.size(); ++i) {
        int d = track_match[i];
        if (d < 0) continue;
        Track& tr = tracks_[i];
        Detection nd = dets[d];
        update_velocity(tr, nd.cx, nd.cy, dt);
        tr.det = nd;
        tr.kpts.clear();
        uint32_t off = nd.kpt_offset, cnt = nd.kpt_count;
        if (cnt > 0 && off + cnt <= kpts.size()) {
            tr.kpts.assign(kpts.begin() + off, kpts.begin() + off + cnt);
        }
        tr.hits++;
        tr.misses = 0;
        tr.last_seen_s = t_mono_s;
        dets[d].track_id = tr.track_id;
    }

    // Age unmatched tracks; remove on timeout.
    for (size_t i = 0; i < tracks_.size();) {
        if (track_match[i] >= 0) {
            ++i;
            continue;
        }
        Track& tr = tracks_[i];
        tr.misses++;
        bool by_misses = static_cast<int>(tr.misses) > cfg_.max_misses;
        bool by_time = cfg_.max_lost_s > 0 && (t_mono_s - tr.last_seen_s) > cfg_.max_lost_s;
        if (by_misses || by_time) {
            removed.push_back(tr.track_id);
            tracks_.erase(tracks_.begin() + i);
            track_match.erase(track_match.begin() + i);
        } else {
            ++i;
        }
    }

    // New tracks for unmatched detections.
    for (size_t d = 0; d < dets.size(); ++d) {
        if (det_used[d]) continue;
        Track tr;
        tr.track_id = next_id_++;
        tr.det = dets[d];
        uint32_t off = dets[d].kpt_offset, cnt = dets[d].kpt_count;
        if (cnt > 0 && off + cnt <= kpts.size()) {
            tr.kpts.assign(kpts.begin() + off, kpts.begin() + off + cnt);
        }
        tr.hits = 1;
        tr.misses = 0;
        tr.first_seen_s = tr.last_seen_s = t_mono_s;
        dets[d].track_id = tr.track_id;
        tracks_.push_back(std::move(tr));
    }

    // Alive list: insertion (ascending-id) order, filtered by min_hits.
    alive_.clear();
    for (const auto& tr : tracks_) {
        if (static_cast<int>(tr.hits) >= cfg_.min_hits) alive_.push_back(tr);
    }
    return alive_;
}

}  // namespace vb
