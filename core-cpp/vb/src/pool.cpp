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

#include "pool.h"

namespace vb {

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
                std::lock_guard<std::mutex> hlk(s->holder_mu);
                if (s->busy || s->removing) continue;
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
        for (size_t i = 0; i < batch.size(); ++i) process(batch[i], frames[i], ctx);
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

    std::vector<uint32_t> removed;
    const std::vector<Track>& alive =
        s->tracker.update(res.dets, res.kpts, f.t_mono_s, removed);

    // Analyzer chain (per-track float attributes + events).
    size_t total_attrs = 0;
    for (auto& a : s->analyzers) total_attrs += a->attr_count();
    std::vector<float> attrs(alive.size() * total_attrs, 0.0f);
    FrameMeta meta;
    meta.stream_index = s->index;
    meta.seq = f.seq;
    meta.wall_ms = f.wall_ms;
    meta.t_mono_s = f.t_mono_s;
    meta.geom = res.geom;
    std::vector<AnalyzerEvent> events;
    size_t attr_off = 0;
    for (auto& a : s->analyzers) {
        float* dst = total_attrs ? attrs.data() + attr_off : nullptr;
        a->on_frame(meta, alive, dst, events);
        attr_off += a->attr_count();
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

    encode_frame_record(*s, f, res, alive, attrs, total_attrs);

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

void ContextPool::encode_frame_record(StreamState& s, const FrameBuf& f,
                                      const DetectionResult& res,
                                      const std::vector<Track>& alive,
                                      const std::vector<float>& attrs,
                                      size_t attr_total) {
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
    size_t kpt_n = 0;
    for (auto& t : alive) {
        if (t.kpts.empty()) continue;
        if (kpt_n == 0) kpt_n = t.kpts.size();
        else if (t.kpts.size() != kpt_n) kpt_n = static_cast<size_t>(-1);
    }
    if (kpt_n == static_cast<size_t>(-1)) kpt_n = 0;
    r.kpt_per_det = static_cast<uint8_t>(std::min<size_t>(255, kpt_n));
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
    r.attr_per_det = static_cast<uint8_t>(std::min<size_t>(255, attr_total));
    r.attrs = attrs;
    std::vector<uint8_t> rec;
    wire_encode_vbr1(r, rec);
    rt_->writer_.push_frame(std::move(rec));
    if (rt_->on_frame_rec) rt_->on_frame_rec(r);
}

}  // namespace vb
