// Rockchip MPP RTSP frame source (spec BASE-1 §M2.1).
//
// rtspsrc ! rtph264depay ! h264parse ! mppvideodec ! video/x-raw(memory:DMABuf),
// format=NV12 ! appsink   (H.265 uses rtph265depay ! h265parse)
//
// Pipeline shape, element properties and the single-DMA-BUF NV12 plane
// assumptions follow fall platforms/rknn/video_source.py @ 5af128d
// (`GStreamerMPP.start` / `_native_frame`, output_format "dma_nv12", license
// Apache-2.0). The launch string is not built from the stream url: the url is
// set as the rtspsrc `location` property, as in src/sources/gst_source.cpp.
//
// What this source does NOT do: it never maps the buffer, never converts a
// pixel and never falls back to software decoding. A frame is a borrowed
// DMABUF fd plus the GstSample that keeps it alive (FrameBuf::hold).
#include <cstring>
#include <mutex>

#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include "mpp_source.h"
#include "vb/json.h"

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

// The MPP decoder element and its custom format enum (GstMppVideoDecFormat):
// 23 is NV12, which the device's gst-inspect-1.0 reports explicitly.
constexpr int kMppFormatNv12 = 23;

// NV12 source dimensions here are bounded by the same limit the generic source
// uses, so a misbehaving decoder cannot size an allocation.
constexpr int kMppMaxDim = 8192;

struct CodecElements {
    const char* depay;
    const char* parse;
    const char* encoding_name;
};

// rtspsrc exposes its pads dynamically, so the video pad has to be matched by
// encoding-name at runtime. The wanted name and the element to link into travel
// through this context, one per pipeline.
struct PadLinkCtx {
    GstElement* next = nullptr;
    const char* encoding_name = "";
};

void on_rtsp_pad_added(GstElement* /*src*/, GstPad* pad, gpointer user_data) {
    auto* ctx = static_cast<PadLinkCtx*>(user_data);
    if (!ctx || !ctx->next) return;
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);
    bool wanted = false;
    if (caps && gst_caps_get_size(caps) > 0) {
        const GstStructure* st = gst_caps_get_structure(caps, 0);
        const char* media = gst_structure_get_string(st, "media");
        const char* encoding = gst_structure_get_string(st, "encoding-name");
        wanted = media && encoding && std::strcmp(media, "video") == 0 &&
                 std::strcmp(encoding, ctx->encoding_name) == 0;
    }
    if (caps) gst_caps_unref(caps);
    if (!wanted) return;
    GstPad* sinkpad = gst_element_get_static_pad(ctx->next, "sink");
    if (!sinkpad) return;
    if (!gst_pad_is_linked(sinkpad)) gst_pad_link(pad, sinkpad);
    gst_object_unref(sinkpad);
}

// GstRTSPLowerTrans values, spelled out so this file does not need the
// gstreamer-rtsp-1.0 pkg-config module for two constants.
constexpr guint kRtspLowerTransUdp = 1u << 0;
constexpr guint kRtspLowerTransTcp = 1u << 2;

bool codec_elements(const StreamSpec& spec, CodecElements& out, std::string& err) {
    std::string codec = "h264";
    if (!spec.options_json.empty() && spec.options_json != "{}") {
        try {
            Json j = json_parse(spec.options_json);
            auto it = j.find("codec");
            if (it != j.end() && it->is_string()) codec = it->get<std::string>();
        } catch (const std::exception& e) {
            err = std::string("stream options: ") + e.what();
            return false;
        }
    }
    for (char& c : codec)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (codec == "h264" || codec == "avc") {
        out = {"rtph264depay", "h264parse", "H264"};
        return true;
    }
    if (codec == "h265" || codec == "hevc") {
        out = {"rtph265depay", "h265parse", "H265"};
        return true;
    }
    err = "rknn source: unsupported options.codec '" + codec + "' (h264|h265)";
    return false;
}

class MppSource : public FrameSource {
public:
    explicit MppSource(StreamSpec spec) : spec_(std::move(spec)) {}
    ~MppSource() override { close(); }

    // Builds the pipeline without starting it (same contract as
    // make_gst_source: a bad url or a missing element fails at add() time).
    bool build(std::string& err) {
        std::lock_guard<std::mutex> lk(mu_);
        return build_locked(err);
    }

    bool open(std::string& err) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (playing_) return true;
        if (!pipeline_ && !build_locked(err)) return false;
        if (!start_locked(err)) {
            close_locked();
            return false;
        }
        next_seq_ = 0;
        eos_ = false;
        failed_.clear();
        return true;
    }

    int read(FrameBuf& out, int timeout_ms) override {
        GstElement* sink = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!pipeline_ || !sink_) return -1;
            if (!failed_.empty()) return -1;  // sticky: never resume on a soft path
            // Hold a reference across the pull: close() may unref sink_ from
            // another thread while we are blocked in try_pull_sample().
            sink = GST_ELEMENT(gst_object_ref(sink_));
        }
        GstSample* sample = gst_app_sink_try_pull_sample(
            GST_APP_SINK(sink), (timeout_ms > 0 ? timeout_ms : 1000) * GST_MSECOND);
        if (!sample) {
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
        GstBuffer* buf = gst_sample_get_buffer(sample);
        std::string why;
        if (!caps || !buf || !gst_video_info_from_caps(&info, caps)) {
            why = "sample has no usable video caps";
        } else if (!gst_caps_features_contains(gst_caps_get_features(caps, 0),
                                               GST_CAPS_FEATURE_MEMORY_DMABUF)) {
            // The whole point of this source is a device buffer; a plain
            // video/x-raw buffer would mean the pipeline dropped to CPU memory.
            why = "decoded buffer is not memory:DMABuf";
        } else if (GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_NV12) {
            why = "decoded buffer is not NV12";
        }
        if (why.empty()) {
            const int w = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
            const int h = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
            const int y_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
            const gsize y_off = GST_VIDEO_INFO_PLANE_OFFSET(&info, 0);
            const gsize uv_off = GST_VIDEO_INFO_PLANE_OFFSET(&info, 1);
            if (w <= 0 || h <= 0 || w > kMppMaxDim || h > kMppMaxDim || (w % 2) || (h % 2)) {
                why = "invalid NV12 geometry from mppvideodec";
            } else if (y_stride < w) {
                why = "invalid NV12 Y stride from mppvideodec";
            } else if (y_off != 0 || uv_off != static_cast<gsize>(y_stride) * h) {
                // RGA wraps one fd with no plane offset; NV12 is expressible
                // only when the UV plane follows the Y plane contiguously.
                why = "NV12 planes are not contiguous (Y off=" + std::to_string(y_off) +
                      " UV off=" + std::to_string(uv_off) + " stride=" +
                      std::to_string(y_stride) + " h=" + std::to_string(h) + ")";
            }
        }
        int fd = -1;
        if (why.empty()) {
            if (gst_buffer_n_memory(buf) != 1) {
                why = "dma_nv12 supports exactly one GstMemory";
            } else {
                GstMemory* mem = gst_buffer_peek_memory(buf, 0);
                if (!mem || !gst_is_dmabuf_memory(mem)) {
                    why = "GstMemory is not DMA-BUF";
                } else {
                    fd = gst_dmabuf_memory_get_fd(mem);
                    if (fd < 0) why = "gst_dmabuf_memory_get_fd returned a negative fd";
                }
            }
        }
        if (!why.empty()) {
            gst_sample_unref(sample);
            gst_object_unref(sink);
            std::lock_guard<std::mutex> lk(mu_);
            // Sticky: a source that cannot deliver device buffers must not keep
            // running as if it were on the intended path.
            failed_ = why;
            eos_ = true;
            return -1;
        }

        const int w = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
        const int h = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
        const int y_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);

        uint64_t seq;
        {
            std::lock_guard<std::mutex> lk(mu_);
            seq = next_seq_++;
        }
        out.stream_index = 0;
        out.seq = seq;
        out.wall_ms = now_ms_epoch();
        out.t_mono_s = now_s();
        out.w = w;
        out.h = h;
        out.stride = y_stride;
        out.fmt = PixFmt::NV12;
        out.mem = Mem::DmaBuf;
        out.host = nullptr;
        out.dmabuf_fd = fd;
        out.full_w = w;
        out.full_h = h;
        // The fd belongs to the GstBuffer allocator and may be recycled as soon
        // as the sample dies, so the sample is what the frame holds.
        out.hold = std::shared_ptr<GstSample>(sample, [](GstSample* s) {
            gst_sample_unref(s);
        });
        gst_object_unref(sink);
        return 1;
    }

    void close() override {
        std::lock_guard<std::mutex> lk(mu_);
        close_locked();
    }

    const char* decode_path() const override { return "mpp"; }

    // Last hard failure (empty when none): surfaced by the backend so a refused
    // buffer is reported instead of looking like a stalled stream.
    std::string failure() const {
        std::lock_guard<std::mutex> lk(mu_);
        return failed_;
    }

private:
    bool build_locked(std::string& err) {
        if (pipeline_) return true;
        CodecElements codec{};
        if (!codec_elements(spec_, codec, err)) return false;
        if (spec_.url.rfind("rtsp://", 0) != 0 && spec_.url.rfind("rtsps://", 0) != 0) {
            err = "rknn source: only rtsp:// urls are supported (got '" +
                  spec_.url + "')";
            return false;
        }
        const char* factories[6] = {"rtspsrc", codec.depay, codec.parse,
                                    "mppvideodec", "capsfilter", "appsink"};
        GstElement* els[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        for (int i = 0; i < 6; ++i) {
            els[i] = gst_element_factory_make(factories[i], nullptr);
            if (!els[i]) {
                err = std::string("GStreamer element unavailable: ") + factories[i] +
                      (i == 3 ? " (no RK MPP plugin: hardware decode is required,"
                                " there is no software fallback)"
                              : "");
                for (GstElement* e : els)
                    if (e) gst_object_unref(e);
                return false;
            }
        }
        GstElement* src = els[0];
        GstElement* decoder = els[3];
        GstElement* capsfilter = els[4];
        GstElement* sink = els[5];

        g_object_set(src, "location", spec_.url.c_str(), nullptr);
        g_object_set(src, "latency", 100, nullptr);
        g_object_set(src, "drop-on-latency", TRUE, nullptr);
        // tcp keeps control and media on the one connection the fleet's LAN
        // test source tolerates; udp is opt-in through options.transport.
        bool tcp = true;
        if (spec_.transport == "udp") tcp = false;
        if (!spec_.options_json.empty() && spec_.options_json != "{}") {
            try {
                Json j = json_parse(spec_.options_json);
                auto it = j.find("transport");
                if (it != j.end() && it->is_string())
                    tcp = it->get<std::string>() != "udp";
            } catch (...) {
            }
        }
        // rtspsrc's default is "try UDP, fall back to TCP", which makes the
        // negotiated transport depend on the network; the measured LAN source
        // is reached over TCP, so both branches are set explicitly.
        g_object_set(src, "protocols", tcp ? kRtspLowerTransTcp : kRtspLowerTransUdp,
                     nullptr);

        g_object_set(decoder, "fast-mode", TRUE, nullptr);
        // Custom element enum: 23 = NV12. Setting it as an int is what
        // PyGObject did for the same property upstream.
        g_object_set(decoder, "format", kMppFormatNv12, nullptr);
        // Without the DMA-BUF feature the element hands out CPU-mapped buffers,
        // which is exactly the path this source must not take.
        if (!g_object_class_find_property(G_OBJECT_GET_CLASS(decoder), "dma-feature")) {
            err = "mppvideodec has no dma-feature property: this build cannot "
                  "produce DMA-BUF NV12";
            for (GstElement* e : els)
                if (e) gst_object_unref(e);
            return false;
        }
        g_object_set(decoder, "dma-feature", TRUE, nullptr);

        GstCaps* caps = gst_caps_new_simple(
            "video/x-raw", "format", G_TYPE_STRING, "NV12", "width",
            GST_TYPE_INT_RANGE, 1, kMppMaxDim, "height", GST_TYPE_INT_RANGE, 1,
            kMppMaxDim, nullptr);
        gst_caps_set_features(caps, 0,
                              gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_DMABUF,
                                                    nullptr));
        g_object_set(capsfilter, "caps", caps, nullptr);
        gst_caps_unref(caps);

        g_object_set(sink, "max-buffers", 3u, "drop", TRUE, "sync", FALSE,
                     "emit-signals", FALSE, nullptr);

        pipeline_ = gst_pipeline_new("vb-mpp-source");
        if (!pipeline_) {
            err = "gst_pipeline_new failed";
            for (GstElement* e : els)
                if (e) gst_object_unref(e);
            return false;
        }
        // Every element has to be in the bin before linking: gst_element_link()
        // refuses two elements that do not share an ancestor, so leaving the
        // depayloader/parser out of the bin makes the whole link fail.
        gst_bin_add_many(GST_BIN(pipeline_), src, els[1], els[2], decoder, capsfilter,
                         sink, nullptr);
        pad_ctx_.next = els[1];
        pad_ctx_.encoding_name = codec.encoding_name;
        g_signal_connect(src, "pad-added", G_CALLBACK(on_rtsp_pad_added), &pad_ctx_);
        bool linked = gst_element_link(els[1], els[2]) &&
                      gst_element_link(els[2], decoder) &&
                      gst_element_link(decoder, capsfilter) &&
                      gst_element_link(capsfilter, sink);
        if (!linked) {
            err = "failed to link the MPP frame source pipeline";
            close_locked();
            return false;
        }
        sink_ = GST_ELEMENT(gst_object_ref(sink));
        return true;
    }

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
            static_cast<GstClockTime>(kOpenTimeoutS * GST_SECOND));
        if (wr == GST_STATE_CHANGE_FAILURE) {
            err = bus_error_locked("pipeline failed to go PLAYING");
            return false;
        }
        if (state != GST_STATE_PLAYING) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "source did not reach PLAYING within %.1f s", kOpenTimeoutS);
            err = std::string(buf) + " (" + bus_error_locked("no error reported") + ")";
            return false;
        }
        playing_ = true;
        return true;
    }

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

    static constexpr double kOpenTimeoutS = 5.0;

    StreamSpec spec_;
    PadLinkCtx pad_ctx_{};
    std::string failed_;
    GstElement* pipeline_ = nullptr;
    GstElement* sink_ = nullptr;
    mutable std::mutex mu_;
    uint64_t next_seq_ = 0;
    bool eos_ = false;
    bool playing_ = false;
};

struct GstInit {
    GstInit() { gst_init(nullptr, nullptr); }
};

}  // namespace

std::unique_ptr<FrameSource> make_mpp_source(const StreamSpec& s, std::string& err) {
    static GstInit init;  // idempotent
    auto src = std::make_unique<MppSource>(s);
    if (!src->build(err)) {
        // The add reply carries this error to the control session, but neither
        // --parity nor --standalone prints replies, and "add failed" alone does
        // not say why the hardware pipeline was refused.
        std::fprintf(stderr, "rknn source: %s (%s)\n",
                     err.empty() ? "pipeline refused" : err.c_str(), s.url.c_str());
        return nullptr;
    }
    return src;
}

}  // namespace vb
