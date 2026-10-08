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
#include <limits>

#include "pool.h"
#include "vb/rate_crop.h"

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
        if (c.name == "stage2" && s.stage2) {
            try { c.ok = rt_->configure_stage2(s, json_parse(c.json), c.err); }
            catch (const std::exception& e) { c.ok = false; c.err = e.what(); }
        } else if (!a) {
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
    // Host ROI inference uses local pixels, but all downstream coordinates
    // remain in the original source geometry. Apply this correction exactly
    // once before stage2/tracker/analyzers consume the result.
    if (f.crop_x0 != 0 || f.crop_y0 != 0) {
        try {
            CropRectPx crop{f.crop_x0, f.crop_y0, f.w, f.h};
            res.geom = crop_geom(f.full_w, f.full_h, crop, res.geom.model_w,
                                 res.geom.model_h, res.geom.align);
        } catch (const std::exception& e) {
            s->metrics.record_state("error", e.what());
            return;
        }
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

    // Stage2 is the only pixel analyzer. It runs after tracking and before
    // the ordinary analyzer chain; a failed read leaves reads empty while the
    // detection/analyzer path continues.
    std::vector<Stage2Read> stage2_reads;
    if (s->stage2) {
        std::vector<Detection> candidates;
        candidates.reserve(alive.size());
        for (const auto& t : alive) if (t.misses == 0) {
            Detection d = t.det;
            // Decoder coordinates are on the model canvas. Stage2 requests
            // are source-normalized, including letterbox reversal.
            float cx, cy, w, h;
            res.geom.box_to_source_norm(d.cx, d.cy, d.w, d.h, cx, cy, w, h);
            d.cx = cx; d.cy = cy; d.w = w; d.h = h; d.track_id = t.track_id;
            candidates.push_back(d);
        }
        std::vector<CropReq> crops = stage2_select_crops(candidates, f.t_mono_s,
                                                          s->stage2_filter,
                                                          s->stage2_tracks);
        std::vector<TensorView> outs(crops.size());
        std::string s2err;
        double stage2_start = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        bool stage2_failed = false;
        if (!crops.empty() && s->stage2->infer_crops(f, crops.data(), crops.size(), outs.data(), s2err) == 0) {
            stage2_reads.reserve(crops.size());
            for (size_t i = 0; i < crops.size(); ++i) {
                float mean = -1, minc = -1;
                std::string txt = ctc_greedy(outs[i], s->stage2_layout, s->stage2_charset,
                                              &mean, &minc);
                if (mean < 0 || minc < 0) { stage2_failed = true; break; }
                Stage2Read r; r.track_id = crops[i].track_id; r.text = std::move(txt);
                r.mean_conf = mean; r.min_char_conf = minc; r.seq = f.seq;
                r.bbox[0] = crops[i].x0; r.bbox[1] = crops[i].y0;
                r.bbox[2] = crops[i].x1; r.bbox[3] = crops[i].y1;
                stage2_reads.push_back(std::move(r));
            }
        } else if (!crops.empty()) {
            stage2_failed = true;
        }
        if (!crops.empty()) {
            double stage2_end = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            s->metrics.record_stage2(static_cast<float>((stage2_end - stage2_start) * 1000), stage2_failed);
            if (stage2_failed) stage2_reads.clear();
        }
    }

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
    meta.reads = s->stage2 ? &stage2_reads : nullptr;
    std::vector<AnalyzerEvent> events;
    uint64_t dedup_before = 0;
    for (auto& a : s->analyzers) dedup_before += a->text_vote_dedup_count();
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
        if (s->stage2) s->stage2_tracks.erase(id);
        for (auto& a : s->analyzers) {
            a->on_track_removed(id, f.t_mono_s, events);
            emit_events(*s, meta, events, a->name());
            events.clear();
        }
    }
    uint64_t dedup_after = 0;
    for (auto& a : s->analyzers) dedup_after += a->text_vote_dedup_count();
    if (dedup_after > dedup_before) s->metrics.add_text_vote_dedup(dedup_after - dedup_before);

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
    if (s->ring_cap > 0) {
        SnapshotRingEntry e;
        e.seq = f.seq;
        bool snapshot_ok = false;
        if (f.host && (f.fmt == PixFmt::RGB888 || f.fmt == PixFmt::BGR888)) {
            e.w = f.full_w > 0 ? f.full_w : f.w;
            e.h = f.full_h > 0 ? f.full_h : f.h;
            const uint8_t* src = f.full_host ? f.full_host : f.host;
            const int src_stride = f.full_stride > 0 ? f.full_stride : f.stride;
            const size_t ew = e.w > 0 ? static_cast<size_t>(e.w) : 0;
            const size_t row_bytes = ew <= std::numeric_limits<size_t>::max() / 3 ? ew * 3 : 0;
            size_t total_bytes = 0;
            const bool size_ok = host_rgb_layout_bytes(e.w, e.h, src_stride, total_bytes);
            if (size_ok) {
                e.pixels.resize(total_bytes);
                for (int y = 0; y < e.h; ++y) {
                    const uint8_t* row = src + static_cast<size_t>(y) * static_cast<size_t>(src_stride);
                    uint8_t* dst = e.pixels.data() + static_cast<size_t>(y) * row_bytes;
                    if (f.fmt == PixFmt::RGB888) std::memcpy(dst, row, row_bytes);
                    else for (int x = 0; x < e.w; ++x) {
                        const size_t off = static_cast<size_t>(x) * 3;
                        dst[off + 0] = row[off + 2];
                        dst[off + 1] = row[off + 1];
                        dst[off + 2] = row[off + 0];
                    }
                }
                snapshot_ok = true;
            }
        } else {
            std::string snapshot_err;
            const int source_w = f.full_w > 0 ? f.full_w : f.w;
            const int source_h = f.full_h > 0 ? f.full_h : f.h;
            size_t expected_bytes = 0;
            // Check the full-source RGB budget before the adapter allocates.
            const bool layout_ok = source_w > 0 &&
                static_cast<size_t>(source_w) <= kMaxHostRgbBytes / 3u &&
                host_rgb_layout_bytes(source_w, source_h, source_w * 3, expected_bytes);
            if (layout_ok)
                snapshot_ok = ctx->copy_snapshot_rgb(f, e.pixels, e.w, e.h, snapshot_err);
            if (snapshot_ok && e.w == source_w && e.h == source_h &&
                e.pixels.size() == expected_bytes) {
                // valid optional conversion
            } else {
                snapshot_ok = false;
                e.pixels.clear();
            }
        }
        if (!snapshot_ok) {
            // Snapshot conversion is optional; do not affect inference or
            // frame accounting when a backend has no conversion capability.
        } else for (const auto& t : alive) {
            float scx, scy, sw, sh;
            res.geom.box_to_source_norm(t.det.cx, t.det.cy, t.det.w, t.det.h,
                                        scx, scy, sw, sh);
            SnapBox b;
            b.track_id = t.track_id;
            b.x0 = (scx - sw / 2) * e.w;
            b.y0 = (scy - sh / 2) * e.h;
            b.x1 = (scx + sw / 2) * e.w;
            b.y1 = (scy + sh / 2) * e.h;
            e.boxes.push_back(b);
        }
        if (snapshot_ok) {
            s->ring.push_back(std::move(e));
            while (s->ring.size() > static_cast<size_t>(s->ring_cap)) s->ring.pop_front();
        }
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
    r.src_w = f.full_w > 0 ? f.full_w : f.w;
    r.src_h = f.full_h > 0 ? f.full_h : f.h;
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
