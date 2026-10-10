// Hailo frame source: fall's GStreamer pipeline as a §6.1 FrameSource
// (spec BASE-1 §M2.3). Pipeline shape from fall main.cpp pipelinePrefix /
// sharedPipeline (origin/main@eb72e1e); read loop modelled on the M1.9
// generic source (try_pull_sample, EOS-vs-timeout disambiguation, row-wise
// copy into a tightly packed RGB buffer).
#include "hailo_source.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "runtime_config.h"
#include "sources/gst_source.h"

namespace vb {
namespace hailo_source {

bool validate_source_url(const std::string& url, std::string& err) {
    if (url.rfind("test://", 0) == 0) return true;
    const bool uri = url.rfind("rtsp://", 0) == 0 || url.rfind("rtsps://", 0) == 0 ||
                     url.rfind("file://", 0) == 0 || url.rfind("http://", 0) == 0 ||
                     url.rfind("https://", 0) == 0;
    if (!uri) {
        err = "hailo source: url must start with rtsp://, rtsps://, file://, "
              "http(s):// or test://";
        return false;
    }
    // Composed into a gst_parse_launch string: reject launch-syntax and
    // whitespace characters. (fall composes the same string without this
    // check; the base refuses instead of trusting the config source.)
    for (char c : url) {
        if (c == '!' || c == '"' || c == '\'' || c == '`' || c == ';' ||
            c == '\\' || c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            err = "hailo source: url contains a character the launch-string "
                  "composition forbids (! \" ' ` ; \\ whitespace)";
            return false;
        }
    }
    return true;
}

std::string pipeline_prefix(const std::string& url, const RtspSettings& rtsp) {
    std::ostringstream q;
    if (url.rfind("test://", 0) == 0)
        q << "videotestsrc is-live=true pattern=ball ! video/x-raw,framerate=30/1 ";
    else if (url.rfind("file://", 0) == 0)
        q << "uridecodebin uri=\"" << url << "\" ! videorate ! video/x-raw,framerate=15/1 ";
    else
        q << "rtspsrc location=\"" << url << "\" latency=" << rtsp.latency_ms
          << " protocols=tcp" << (rtsp.drop_on_latency ? " drop-on-latency=true" : "")
          << " ! " << vb::hailo_config::rtspDepayChain(rtsp.codec) << " ";
    q << "! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=640 ";
    return q.str();
}

}  // namespace hailo_source

namespace {

double now_ms_epoch() {
    using namespace std::chrono;
    return duration<double, std::milli>(system_clock::now().time_since_epoch()).count();
}

double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

class HailoSource : public FrameSource {
public:
    HailoSource(StreamSpec spec, hailo_source::RtspSettings rtsp)
        : spec_(std::move(spec)), rtsp_(std::move(rtsp)) {}
    ~HailoSource() override { close(); }

    bool open(std::string& err) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (playing_) return true;
        if (!pipeline_ && !build_locked(err)) return false;
        if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) ==
            GST_STATE_CHANGE_FAILURE) {
            err = "hailo source: pipeline failed to start";
            close_locked();
            return false;
        }
        playing_ = true;
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
            // try_pull_sample returns NULL immediately once the appsink is EOS
            // with its queue drained; ask the sink directly so stream end is
            // reported the moment it is true (M1.9 review finding).
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
                        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                            GError* e = nullptr;
                            gchar* dbg = nullptr;
                            gst_message_parse_error(msg, &e, &dbg);
                            std::fprintf(stderr, "hailo source: pipeline error: %s (%s)\n",
                                         e ? e->message : "?", dbg ? dbg : "");
                            if (e) g_error_free(e);
                            g_free(dbg);
                        }
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
        // The model canvas is fixed by the caps (640x640 RGB); rows are still
        // copied individually because videoconvert may pad the row stride.
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
        out.stride = static_cast<int32_t>(row);
        out.fmt = PixFmt::RGB888;
        out.mem = Mem::Host;
        out.host = pixels->data();
        out.full_w = w;
        out.full_h = h;
        out.hold = pixels;
        return 1;
    }

    void close() override {
        std::lock_guard<std::mutex> lk(mu_);
        close_locked();
    }

    const char* decode_path() const override {
        // h265 pins the stateless V4L2 decoder + GL convert/scale (Pi 5 has
        // HEVC hardware only); everything else decodes in software.
        return rtsp_.codec == "h265" ? "hw" : "sw";
    }

private:
    bool build_locked(std::string& err) {
        if (pipeline_) return true;
        if (!hailo_source::validate_source_url(spec_.url, err)) return false;
        try {
            hailo_source::RtspSettings probe = rtsp_;  // validate() throws
            probe.validate();
        } catch (const std::exception& e) {
            err = std::string("hailo source: ") + e.what();
            return false;
        }
        if (rtsp_.codec == "h265") {
            // fall main.cpp: pin the GL backend for gldownload before the
            // pipeline is created; do not overwrite caller-set values.
            ::setenv("GST_GL_PLATFORM", "egl", 0);
            ::setenv("GST_GL_WINDOW", "surfaceless", 0);
            ::setenv("GST_GL_API", "gles2", 0);
        }
        const std::string desc = hailo_source::pipeline_prefix(spec_.url, rtsp_) +
                                 "! appsink name=sink max-buffers=2 drop=true "
                                 "sync=false emit-signals=true";
        GError* e = nullptr;
        pipeline_ = gst_parse_launch(desc.c_str(), &e);
        if (!pipeline_) {
            err = std::string("hailo source: pipeline creation failed: ") +
                  (e ? e->message : "unknown error");
            if (e) g_error_free(e);
            return false;
        }
        sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        if (!sink_) {
            err = "hailo source: appsink missing from the pipeline";
            close_locked();
            return false;
        }
        return true;
    }

    void close_locked() {
        if (sink_) {
            gst_object_unref(sink_);
            sink_ = nullptr;
        }
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
        playing_ = false;
    }

    StreamSpec spec_;
    hailo_source::RtspSettings rtsp_;
    std::mutex mu_;
    GstElement* pipeline_ = nullptr;
    GstElement* sink_ = nullptr;
    bool playing_ = false;
    bool eos_ = false;
    uint64_t next_seq_ = 0;
};

struct GstInit {
    GstInit() { gst_init(nullptr, nullptr); }
};

}  // namespace

std::unique_ptr<FrameSource> make_hailo_source(const StreamSpec& s,
                                               const hailo_source::RtspSettings& rtsp,
                                               std::string& err) {
    static GstInit init;  // idempotent
    err.clear();
    auto src = std::make_unique<HailoSource>(s, rtsp);
    // Fail add-time for a rejected url or an unavailable element without
    // touching the network (same contract as the generic source).
    if (!src->open(err)) return nullptr;
    return src;
}

}  // namespace vb
