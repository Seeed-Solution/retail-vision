// Generic GStreamer frame source (spec BASE-1 §8 M1.9, VB_WITH_GST only).
//
// Two shapes, decided by classify_source_url():
//   Uri         — uridecodebin with the url set through its "uri" property;
//                 the url never reaches the launch-line parser.
//   DevPipeline — a verbatim launch string, dev mode only (§6.12).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include "sources/gst_source.h"

namespace vb {
namespace {

double now_ms_epoch() {
    using namespace std::chrono;
    return duration<double, std::milli>(system_clock::now().time_since_epoch()).count();
}

double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// uridecodebin exposes its decoded pads dynamically; link the first video pad
// into the static chain. expose-all-streams=FALSE + caps=video/x-raw already
// restrict what is exposed, so this is a guard rather than a filter.
void on_decode_pad_added(GstElement* /*decode*/, GstPad* pad, gpointer user_data) {
    GstElement* next = GST_ELEMENT(user_data);
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);
    bool video = false;
    if (caps && gst_caps_get_size(caps) > 0) {
        const char* name = gst_structure_get_name(gst_caps_get_structure(caps, 0));
        video = name != nullptr && g_str_has_prefix(name, "video/");
    }
    if (caps) gst_caps_unref(caps);
    if (!video) return;
    GstPad* sinkpad = gst_element_get_static_pad(next, "sink");
    if (!sinkpad) return;
    if (!gst_pad_is_linked(sinkpad)) gst_pad_link(pad, sinkpad);
    gst_object_unref(sinkpad);
}

class GstSource : public FrameSource {
public:
    explicit GstSource(StreamSpec spec) : spec_(std::move(spec)) {}
    ~GstSource() override { close(); }

    // Builds the pipeline without starting it, so make_gst_source() can fail
    // add-time for a rejected url or an unavailable element without touching
    // the network.
    bool build(std::string& err) {
        std::lock_guard<std::mutex> lk(mu_);
        return build_locked(err);
    }

    bool open(std::string& err) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (playing_) return true;
        if (!pipeline_ && !build_locked(err)) return false;
        if (!start_locked(err)) {
            // The add reply carries this, but neither --parity nor --standalone
            // prints replies, and "add failed" alone does not say why the
            // pipeline never reached PLAYING. Deduped: the runtime retries
            // open() until its add deadline.
            if (err != last_open_err_) {
                last_open_err_ = err;
                std::fprintf(stderr, "gst source: open failed: %s\n", err.c_str());
            }
            close_locked();
            return false;
        }
        next_seq_ = 0;
        eos_ = false;
        return true;
    }

    int read(FrameBuf& out, int timeout_ms) override {
        GstElement* sink = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!pipeline_ || !sink_) return -1;
            // Hold a reference across the pull: close() may unref sink_ from
            // another thread while we are blocked in try_pull_sample().
            sink = GST_ELEMENT(gst_object_ref(sink_));
        }
        GstSample* sample = gst_app_sink_try_pull_sample(
            GST_APP_SINK(sink), (timeout_ms > 0 ? timeout_ms : 1000) * GST_MSECOND);
        if (!sample) {
            // try_pull_sample also returns NULL *immediately* (not after the
            // timeout) once the appsink is EOS with its queue drained. The bus
            // message can lag that by a little, and a tight caller loop would
            // otherwise spin through its whole budget returning "no frame yet"
            // without ever letting the bus thread post EOS. Ask the sink
            // directly so stream end is reported the moment it is true.
            const bool sink_eos = gst_app_sink_is_eos(GST_APP_SINK(sink));
            gst_object_unref(sink);
            std::lock_guard<std::mutex> lk(mu_);
            if (!pipeline_) return -1;
            if (eos_) return -1;
            GstBus* bus = gst_element_get_bus(pipeline_);
            int rc = sink_eos ? -1 : 0;
            if (bus) {
                GstMessage* msg;
                while ((msg = gst_bus_pop(bus)) != nullptr) {
                    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS ||
                        GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                        rc = -1;
                        eos_ = true;
                    }
                    gst_message_unref(msg);
                }
                gst_object_unref(bus);
            }
            return rc;
        }

        GstCaps* caps = gst_sample_get_caps(sample);
        GstVideoInfo info;
        if (!caps || !gst_video_info_from_caps(&info, caps)) {
            gst_sample_unref(sample);
            gst_object_unref(sink);
            return -1;
        }
        GstBuffer* buf = gst_sample_get_buffer(sample);
        GstVideoFrame frame;
        if (!buf || !gst_video_frame_map(&frame, &info, buf, GST_MAP_READ)) {
            gst_sample_unref(sample);
            gst_object_unref(sink);
            return -1;
        }
        const int w = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
        const int h = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
        const int src_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
        // The caps bound the negotiated size; re-check it here so a misbehaving
        // element (or a caps-less buffer) cannot size an allocation, and use a
        // checked multiply instead of w * h * 3.
        bool size_ok = w > 0 && h > 0 && w <= kGstMaxDim && h <= kGstMaxDim &&
                       src_stride >= w * 3;
        size_t row = 0, need = 0;
        if (size_ok) {
            row = static_cast<size_t>(w) * 3;
            size_ok = row != 0 && static_cast<size_t>(h) <= kGstMaxFrameBytes / row;
            if (size_ok) need = row * static_cast<size_t>(h);
        }
        if (!size_ok || need > kGstMaxFrameBytes) {
            gst_video_frame_unmap(&frame);
            gst_sample_unref(sample);
            gst_object_unref(sink);
            return -1;
        }
        // Copy rows into an owned tightly-packed RGB buffer (CPU path: plain
        // loops are fine, M1.9). Release the mapped frame immediately.
        auto pixels = std::make_shared<std::vector<uint8_t>>(need);
        for (int y = 0; y < h; ++y) {
            std::memcpy(pixels->data() + static_cast<size_t>(y) * row,
                        reinterpret_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0)) +
                            static_cast<size_t>(y) * static_cast<size_t>(src_stride),
                        row);
        }
        gst_video_frame_unmap(&frame);
        gst_sample_unref(sample);
        gst_object_unref(sink);

        uint64_t seq;
        double t;
        {
            std::lock_guard<std::mutex> lk(mu_);
            seq = next_seq_++;
        }
        t = now_s();
        out.stream_index = 0;
        out.seq = seq;
        out.wall_ms = now_ms_epoch();
        out.t_mono_s = t;
        out.w = w;
        out.h = h;
        out.stride = w * 3;
        out.fmt = PixFmt::RGB888;
        out.mem = Mem::Host;
        out.host = pixels->data();
        out.full_host = out.host;
        out.full_stride = out.stride;
        out.full_w = w;
        out.full_h = h;
        out.hold = pixels;
        return 1;
    }

    void close() override {
        std::lock_guard<std::mutex> lk(mu_);
        close_locked();
    }
    const char* decode_path() const override { return "sw"; }

private:
    bool build_locked(std::string& err) {
        if (pipeline_) return true;
        const SourceUrlKind kind = classify_source_url(spec_.url, err);
        if (kind == SourceUrlKind::Rejected) return false;
        const bool ok = (kind == SourceUrlKind::Uri) ? build_uri_locked(err)
                                                     : build_parse_locked(err);
        if (!ok) close_locked();
        return ok;
    }

    // url -> element property. The launch line is never built from the url.
    bool build_uri_locked(std::string& err) {
        const char* factories[6] = {"uridecodebin", "capsfilter", "videoconvert",
                                    "videoscale", "capsfilter", "appsink"};
        GstElement* els[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        for (int i = 0; i < 6; ++i) {
            els[i] = gst_element_factory_make(factories[i], nullptr);
            if (!els[i]) {
                err = std::string("GStreamer element unavailable: ") + factories[i];
                for (GstElement* e : els)
                    if (e) gst_object_unref(e);
                return false;
            }
        }
        GstElement* decode = els[0];
        GstElement* srccaps = els[1];
        GstElement* conv = els[2];
        GstElement* scale = els[3];
        GstElement* outcaps = els[4];
        GstElement* sink = els[5];

        g_object_set(decode, "uri", spec_.url.c_str(), nullptr);
        // Bounded decode: one stream, bounded network buffer. What keeps the
        // output to video is the `srccaps` capsfilter below, not uridecodebin's
        // own (deprecated) `caps` property: setting `caps` to a bare
        // video/x-raw makes decodebin prune every decoder and fail with
        // "Your GStreamer installation is missing a plug-in" -- measured on the
        // CPU base, which then could not decode H.264 at all.
        g_object_set(decode, "expose-all-streams", FALSE, nullptr);
        g_object_set(decode, "buffer-size", 2 * 1024 * 1024, nullptr);
        // Source-side cap: refuse to decode past kGstMaxDim (a stream that
        // large fails negotiation instead of allocating an unbounded frame).
        GstCaps* in_caps = gst_caps_new_simple(
            "video/x-raw", "width", GST_TYPE_INT_RANGE, 1, kGstMaxDim, "height",
            GST_TYPE_INT_RANGE, 1, kGstMaxDim, nullptr);
        g_object_set(srccaps, "caps", in_caps, nullptr);
        gst_caps_unref(in_caps);
        GstCaps* out_caps = gst_caps_new_simple(
            "video/x-raw", "format", G_TYPE_STRING, "RGB", "width",
            GST_TYPE_INT_RANGE, 1, kGstMaxDim, "height", GST_TYPE_INT_RANGE, 1,
            kGstMaxDim, nullptr);
        g_object_set(outcaps, "caps", out_caps, nullptr);
        gst_caps_unref(out_caps);
        g_object_set(sink, "max-buffers", 1u, "drop", TRUE, "sync", FALSE,
                     "emit-signals", FALSE, nullptr);

        pipeline_ = gst_pipeline_new("vb-source");
        if (!pipeline_) {
            err = "gst_pipeline_new failed";
            for (GstElement* e : els) gst_object_unref(e);
            return false;
        }
        gst_bin_add_many(GST_BIN(pipeline_), decode, srccaps, conv, scale, outcaps,
                         sink, nullptr);
        const bool linked = gst_element_link(srccaps, conv) &&
                            gst_element_link(conv, scale) &&
                            gst_element_link(scale, outcaps) &&
                            gst_element_link(outcaps, sink);
        if (!linked) {
            err = "failed to link the frame source pipeline";
            close_locked();
            return false;
        }
        g_signal_connect(decode, "pad-added", G_CALLBACK(on_decode_pad_added), srccaps);
        sink_ = GST_ELEMENT(gst_object_ref(sink));
        return true;
    }

    // Dev/test launch string (§6.12 gated; classify_source_url already refused
    // it in production).
    bool build_parse_locked(std::string& err) {
        const std::string desc =
            spec_.url +
            " ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=[1," +
            std::to_string(kGstMaxDim) + "],height=[1," +
            std::to_string(kGstMaxDim) +
            "] ! appsink name=sink max-buffers=1 drop=true sync=false";
        GError* gerr = nullptr;
        pipeline_ = gst_parse_launch(desc.c_str(), &gerr);
        if (!pipeline_) {
            err = gerr && gerr->message ? gerr->message : "gst_parse_launch failed";
            if (gerr) g_error_free(gerr);
            return false;
        }
        if (gerr) g_error_free(gerr);
        sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        if (!sink_ || !GST_IS_APP_SINK(sink_)) {
            err = "pipeline has no appsink";
            close_locked();
            return false;
        }
        gst_app_sink_set_emit_signals(GST_APP_SINK(sink_), FALSE);
        return true;
    }

    // Starting a source is not a success until it runs: set_state() reports
    // ASYNC for any source that has to connect or preroll, so wait for the
    // state change (or an error) within kGstOpenTimeoutS and only then claim
    // the source is open (§6.3: the add reply waits for the source).
    bool start_locked(std::string& err) {
        const GstStateChangeReturn sr =
            gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (sr == GST_STATE_CHANGE_FAILURE) {
            err = bus_error_locked("pipeline failed to go PLAYING");
            return false;
        }
        GstState state = GST_STATE_VOID_PENDING;
        const GstStateChangeReturn wr = gst_element_get_state(
            pipeline_, &state, nullptr,
            static_cast<GstClockTime>(kGstOpenTimeoutS * GST_SECOND));
        if (wr == GST_STATE_CHANGE_FAILURE) {
            err = bus_error_locked("pipeline failed to go PLAYING");
            return false;
        }
        if (state != GST_STATE_PLAYING) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "source did not reach PLAYING within %.1f s",
                          kGstOpenTimeoutS);
            err = std::string(buf) + " (" + bus_error_locked("no error reported") + ")";
            return false;
        }
        playing_ = true;
        return true;
    }

    // Best-effort message from the bus (the last ERROR, if any) so a failed
    // open reports why the source did not come up.
    std::string bus_error_locked(const char* dflt) const {
        std::string out = dflt ? dflt : "";
        GstBus* bus = pipeline_ ? gst_element_get_bus(pipeline_) : nullptr;
        if (!bus) return out;
        GstMessage* msg;
        while ((msg = gst_bus_pop(bus)) != nullptr) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* gerr = nullptr;
                gchar* dbg = nullptr;
                gst_message_parse_error(msg, &gerr, &dbg);
                if (gerr && gerr->message) out = gerr->message;
                if (gerr) g_error_free(gerr);
                g_free(dbg);
            }
            gst_message_unref(msg);
        }
        gst_object_unref(bus);
        return out;
    }

    void close_locked() {
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
        if (sink_) {
            gst_object_unref(sink_);
            sink_ = nullptr;
        }
        playing_ = false;
    }

    StreamSpec spec_;
    GstElement* pipeline_ = nullptr;
    GstElement* sink_ = nullptr;
    std::mutex mu_;
    uint64_t next_seq_ = 0;
    bool eos_ = false;
    bool playing_ = false;
    std::string last_open_err_;  // dedupes the open-failure stderr line
};

struct GstInit {
    GstInit() { gst_init(nullptr, nullptr); }
};

}  // namespace

std::unique_ptr<FrameSource> make_gst_source(const StreamSpec& s, std::string& err) {
    static GstInit init;  // idempotent
    auto src = std::make_unique<GstSource>(s);
    // Validate url policy and pipeline construction eagerly so a bad url
    // surfaces at add-time semantics without connecting (open() is called by
    // the runtime's source thread and performs the bounded start).
    if (!src->build(err)) return nullptr;
    return src;
}

}  // namespace vb
