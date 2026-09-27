// ContextPool: N context threads fairly polling per-stream LatestFrame slots
// (spec BASE-1 §5.4 native thread model + §8 M1.7 pool pseudocode).
//
// Guarantees:
//  - a stream's tracker/analyzer chain is held by at most one context thread
//    at a time (busy flag; holder_mu is held for the whole frame);
//  - runtime add/remove: the stream map is mutated under streams_mu_ only;
//    remove waits until the stream is no longer busy before destroying it.
#include <mutex>
#include <algorithm>
#include <cstring>

#include "pool.h"

namespace vb {

bool vbr1_counts_ok(size_t kpt_per_det, size_t attr_per_det, std::string& err) {
    if (kpt_per_det > 255) {
        err = "keypoint count " + std::to_string(kpt_per_det) +
              " exceeds the 255 cap of VBR1";
        return false;
    }
    if (attr_per_det > 255) {
        err = "attribute count " + std::to_string(attr_per_det) +
              " exceeds the 255 cap of VBR1";
        return false;
    }
    return true;
}

void interleave_attrs(const float* scratch, size_t n_tracks, size_t n_attr,
                      size_t total_attrs, size_t attr_off, float* out) {
    for (size_t t = 0; t < n_tracks; ++t) {
        std::memcpy(out + t * total_attrs + attr_off, scratch + t * n_attr,
                    n_attr * sizeof(float));
    }
}

void ContextPool::start() {
    for (auto& ctx : rt_->contexts_) {
        InferenceContext* p = ctx.get();
        threads_.emplace_back([this, p] { worker(p); });
    }
}

void ContextPool::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_)
        if (t.joinable()) t.join();
}

void ContextPool::worker(InferenceContext* ctx) {
    std::vector<std::shared_ptr<StreamState>> snapshot;
    std::vector<std::shared_ptr<StreamState>> batch;
    std::vector<FrameBuf> frames;
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stopping_) return;
        }
        snapshot.clear();
        {
            std::lock_guard<std::mutex> lk(rt_->streams_mu_);
            snapshot.reserve(rt_->streams_.size());
            for (auto& kv : rt_->streams_) snapshot.push_back(kv.second);
        }
        batch.clear();
        frames.clear();
        if (!snapshot.empty()) {
            size_t n = snapshot.size();
            size_t start;
            {
                std::lock_guard<std::mutex> lk(mu_);
                start = cursor_;
            }
            for (size_t k = 0; k < n && batch.size() < static_cast<size_t>(rt_->max_batch_);
                 ++k) {
                const std::shared_ptr<StreamState>& s = snapshot[(start + k) % n];
                // try_lock: one stream that is stuck in a long frame must not
                // hold up the streams after it in the cursor order. A busy
                // (or being-removed) stream is simply skipped this round.
                std::unique_lock<std::mutex> hlk(s->holder_mu, std::try_to_lock);
                if (!hlk.owns_lock() || s->busy || s->removing) continue;
                FrameBuf f;
                if (!s->latest.take(f)) continue;
                s->busy = true;
                batch.push_back(s);
                frames.push_back(std::move(f));
            }
            std::lock_guard<std::mutex> lk(mu_);
            cursor_ += batch.size() + 1;
        }
        if (batch.empty()) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(1));
            continue;
        }
        for (size_t i = 0; i < batch.size(); ++i) {
            // A remove() may have marked this stream after the batch was
            // claimed but before its turn: the removal is waiting on this
            // stream's busy flag, so skipping keeps frames and events of a
            // stopped stream from appearing after the remove reply (and from
            // being attributed to a later stream that reuses the index).
            if (batch[i]->removing) continue;
            process(batch[i], frames[i], ctx);
        }
        // Release the busy set.
        for (auto& s : batch) {
            std::lock_guard<std::mutex> hlk(s->holder_mu);
            s->busy = false;
            s->holder_cv.notify_all();
        }
    }
}

void ContextPool::apply_pending_controls(StreamState& s) {
    std::unique_lock<std::mutex> lk(s.ctl_mu);
    if (s.pending_score.has_value()) {
        s.spec.score_threshold = *s.pending_score;
        s.pending_score.reset();
        s.ctl_cv.notify_all();
    }
    if (s.pending_cfg.has_value() && !s.pending_cfg->done) {
        PendingAnalyzerCfg c = std::move(*s.pending_cfg);
        Analyzer* a = nullptr;
        for (size_t i = 0; i < s.analyzers.size(); ++i)
            if (s.analyzer_names[i] == c.name) a = s.analyzers[i].get();
        if (!a) {
            c.ok = false;
            c.err = "no analyzer " + c.name + " on stream " + std::to_string(s.index);
        } else {
            c.ok = a->configure(c.json, c.err);
        }
        c.done = true;
        s.pending_cfg = std::move(c);
        s.ctl_cv.notify_all();
    }
}

void ContextPool::process(const std::shared_ptr<StreamState>& s, FrameBuf& f,
                          InferenceContext* ctx) {
    // holder_mu held for the whole frame: together with the busy flag this
    // lets the control thread safely touch the chain between frames.
    std::unique_lock<std::mutex> hold(s->holder_mu);
    apply_pending_controls(*s);

    double infer_start_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    DetectionResult res;
    std::string err;
    const FrameBuf* const one_frame[1] = {&f};
    int rc = ctx->infer(one_frame, 1, s->spec.score_threshold, 0.45f, &res, err);
    if (rc != 0) {
        s->metrics.record_state("error", err);
        return;  // frame dropped; the source keeps producing
    }

    // §6.12: dev-mode raw tensor passthrough (VBT1), no-op unless enabled.
    rt_->maybe_send_dev_tensors(*s, f, res, ctx);

    // §6.6/§6.2: with tracker.enabled=false no tracking runs; this frame's
    // detections are wrapped as track_id=0, hits=1, misses=0 tracks so the
    // analyzer chain still sees them. Analyzers that need a track lifecycle
    // were rejected at add().
    std::vector<Track> untracked;
    std::vector<uint32_t> removed;
    const std::vector<Track>* alive_ptr = nullptr;
    if (s->track_enabled) {
        alive_ptr = &s->tracker.update(res.dets, res.kpts, f.t_mono_s, removed);
    } else {
        untracked.reserve(res.dets.size());
        for (const auto& d : res.dets) {
            Track t;
            t.track_id = 0;
            t.det = d;
            t.det.track_id = 0;
            if (d.kpt_count > 0 && d.kpt_offset + d.kpt_count <= res.kpts.size()) {
                t.kpts.assign(res.kpts.begin() + d.kpt_offset,
                              res.kpts.begin() + d.kpt_offset + d.kpt_count);
            }
            t.hits = 1;
            t.misses = 0;
            untracked.push_back(std::move(t));
        }
        alive_ptr = &untracked;
    }
    const std::vector<Track>& alive = *alive_ptr;
    const size_t n_tracks = alive.size();

    // Analyzer chain (per-track float attributes + events). The VBR1 attribute
    // block is [track][attribute] with the summed span as its stride (§6.3),
    // while each analyzer writes its own contiguous per-track block: give every
    // analyzer a private buffer and interleave afterwards, otherwise two
    // analyzers' values land in each other's slots.
    size_t total_attrs = 0;
    for (auto& a : s->analyzers) total_attrs += a->attr_count();
    std::vector<float> attrs(n_tracks * total_attrs, 0.0f);
    std::vector<float> scratch;
    FrameMeta meta;
    meta.stream_index = s->index;
    meta.seq = f.seq;
    meta.wall_ms = f.wall_ms;
    meta.t_mono_s = f.t_mono_s;
    meta.geom = res.geom;
    std::vector<AnalyzerEvent> events;
    size_t attr_off = 0;
    for (auto& a : s->analyzers) {
        const size_t n_attr = a->attr_count();
        if (n_attr == 0) {
            a->on_frame(meta, alive, nullptr, events);
        } else {
            scratch.assign(n_tracks * n_attr, 0.0f);
            a->on_frame(meta, alive, scratch.data(), events);
            interleave_attrs(scratch.data(), n_tracks, n_attr, total_attrs, attr_off,
                             attrs.data());
        }
        attr_off += n_attr;
        emit_events(*s, meta, events, a->name());
        events.clear();
    }
    for (uint32_t id : removed) {
        for (auto& a : s->analyzers) {
            a->on_track_removed(id, f.t_mono_s, events);
            emit_events(*s, meta, events, a->name());
            events.clear();
        }
    }

    std::string enc_err;
    if (!encode_frame_record(*s, f, res, alive, attrs, total_attrs, enc_err)) {
        // §6.3: kpt_per_det/attr_per_det are u8 fields. A count above 255
        // cannot be encoded truthfully, and sending the full payload with the
        // clamped count would desynchronise the receiver on the next record,
        // so the frame is dropped and the reason is reported through the
        // stream state instead.
        s->metrics.record_state("error", enc_err);
    }

    // Snapshot ring (host RGB copy of the newest frames + track boxes).
    if (s->ring_cap > 0 && f.host && f.fmt == PixFmt::RGB888) {
        SnapshotRingEntry e;
        e.seq = f.seq;
        e.w = f.w;
        e.h = f.h;
        e.pixels.assign(f.host, f.host + static_cast<size_t>(f.h) * f.stride);
        for (const auto& t : alive) {
            float scx, scy, sw, sh;
            res.geom.box_to_source_norm(t.det.cx, t.det.cy, t.det.w, t.det.h,
                                        scx, scy, sw, sh);
            SnapBox b;
            b.track_id = t.track_id;
            b.x0 = (scx - sw / 2) * f.w;
            b.y0 = (scy - sh / 2) * f.h;
            b.x1 = (scx + sw / 2) * f.w;
            b.y1 = (scy + sh / 2) * f.h;
            e.boxes.push_back(b);
        }
        s->ring.push_back(std::move(e));
        while (s->ring.size() > static_cast<size_t>(s->ring_cap)) s->ring.pop_front();
    }

    float inference_ms = res.preprocess_ms + res.inference_ms + res.postprocess_ms;
    float queue_delay_ms = static_cast<float>((infer_start_s - f.t_mono_s) * 1000.0);
    s->metrics.record_frame(inference_ms, queue_delay_ms, f.wall_ms);
    uint64_t dropped = s->latest.dropped();
    if (dropped > s->drop_baseline) {
        s->metrics.add_dropped(dropped - s->drop_baseline);
        s->drop_baseline = dropped;
    }
}

void ContextPool::emit_events(StreamState& s, const FrameMeta& m,
                              const std::vector<AnalyzerEvent>& events,
                              const char* analyzer_name) {
    for (auto& ev : events) {
        Json j;
        j["stream_index"] = s.index;
        j["seq"] = m.seq;
        j["wall_ms"] = m.wall_ms;
        j["analyzer"] = analyzer_name;
        j["type"] = ev.type;
        j["track_id"] = ev.track_id;
        try {
            j["fields"] = json_parse(ev.fields_json);
        } catch (...) {
            j["fields"] = ev.fields_json;
        }
        std::vector<uint8_t> rec;
        wire_encode_json_record("VBE1", json_dump(j), rec);
        rt_->writer_.push_event(std::move(rec));
    }
}

bool ContextPool::encode_frame_record(StreamState& s, const FrameBuf& f,
                                      const DetectionResult& res,
                                      const std::vector<Track>& alive,
                                      const std::vector<float>& attrs,
                                      size_t attr_total, std::string& err) {
    // Per-detection counts are u8 in VBR1; refuse rather than emit a record
    // whose declared and actual lengths disagree (B: record-boundary break).
    size_t kpt_n = 0;
    for (auto& t : alive) {
        if (t.kpts.empty()) continue;
        if (kpt_n == 0) kpt_n = t.kpts.size();
        else if (t.kpts.size() != kpt_n) kpt_n = static_cast<size_t>(-1);
    }
    if (kpt_n == static_cast<size_t>(-1)) kpt_n = 0;  // mixed per-track counts: send none
    if (!vbr1_counts_ok(kpt_n, attr_total, err)) return false;

    WireFrameRec r;
    r.stream_index = s.index;
    r.seq = f.seq;
    r.wall_ms = f.wall_ms;
    r.src_w = f.w;
    r.src_h = f.h;
    r.model_w = res.geom.model_w;
    r.model_h = res.geom.model_h;
    r.scale = res.geom.scale;
    r.pad_x = res.geom.pad_x;
    r.pad_y = res.geom.pad_y;
    r.align = static_cast<uint8_t>(res.geom.align);
    r.inference_ms = res.preprocess_ms + res.inference_ms + res.postprocess_ms;
    r.queue_delay_ms = static_cast<float>(
        (std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count() -
         f.t_mono_s) * 1000.0);
    r.kpt_per_det = static_cast<uint8_t>(kpt_n);
    for (auto& t : alive) {
        WireDet wd;
        wd.cx = t.det.cx;
        wd.cy = t.det.cy;
        wd.w = t.det.w;
        wd.h = t.det.h;
        wd.score = t.det.score;
        wd.class_id = t.det.class_id;
        wd.track_id = t.track_id;
        r.dets.push_back(wd);
        if (kpt_n > 0 && t.kpts.size() == kpt_n) {
            for (const auto& k : t.kpts) {
                r.kpts.push_back(k.x);
                r.kpts.push_back(k.y);
                r.kpts.push_back(k.conf);
            }
        }
    }
    r.attr_per_det = static_cast<uint8_t>(attr_total);
    r.attrs = attrs;
    std::vector<uint8_t> rec;
    // The encoder refuses a record whose vectors disagree with the declared
    // per-detection counts; report why instead of dropping the frame silently.
    if (!wire_encode_vbr1(r, rec, &err)) return false;
    rt_->writer_.push_frame(std::move(rec));
    if (rt_->on_frame_rec) rt_->on_frame_rec(r);
    return true;
}

}  // namespace vb
