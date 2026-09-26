// Generic GStreamer frame source (spec BASE-1 §8 M1.9, VB_WITH_GST only).
#include <chrono>
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

class GstSource : public FrameSource {
public:
    explicit GstSource(StreamSpec spec) : spec_(std::move(spec)) {}
    ~GstSource() override { close(); }

    bool open(std::string& err) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (pipeline_) return true;
        GError* gerr = nullptr;
        std::string src_part;
        if (spec_.url.find("://") != std::string::npos) {
            // Escape double quotes so the launch line stays well-formed.
            std::string uri = spec_.url;
            std::string escaped;
            for (char c : uri) {
                if (c == '"') escaped += "\\\"";
                else escaped += c;
            }
            src_part = "uridecodebin uri=\"" + escaped + "\"";
        } else {
            src_part = spec_.url;  // dev/test pipeline head (e.g. videotestsrc)
        }
        const std::string desc =
            src_part +
            " ! videoconvert ! videoscale ! video/x-raw,format=RGB"
            " ! appsink name=sink max-buffers=1 drop=true sync=false";
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
        switch (gst_element_set_state(pipeline_, GST_STATE_PLAYING)) {
            case GST_STATE_CHANGE_FAILURE:
                err = "pipeline failed to go PLAYING";
                close_locked();
                return false;
            default:
                break;  // ASYNC/NO_PREROLL: the first pull waits up to timeout
        }
        next_seq_ = 0;
        eos_ = false;
        return true;
    }

    int read(FrameBuf& out, int timeout_ms) override {
        GstSample* sample = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!pipeline_ || !sink_) return -1;
        }
        sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink_),
                                              (timeout_ms > 0 ? timeout_ms : 1000) * GST_MSECOND);
        if (!sample) {
            std::lock_guard<std::mutex> lk(mu_);
            if (!pipeline_) return -1;
            if (eos_) return -1;
            GstBus* bus = gst_element_get_bus(pipeline_);
            int rc = 0;
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
            return -1;
        }
        GstBuffer* buf = gst_sample_get_buffer(sample);
        GstVideoFrame frame;
        if (!buf || !gst_video_frame_map(&frame, &info, buf, GST_MAP_READ)) {
            gst_sample_unref(sample);
            return -1;
        }
        int w = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
        int h = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
        // Copy rows into an owned tightly-packed RGB buffer (CPU path: plain
        // loops are fine, M1.9). Release the mapped frame immediately.
        auto pixels = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(w) * h * 3);
        for (int y = 0; y < h; ++y) {
            std::memcpy(pixels->data() + static_cast<size_t>(y) * w * 3,
                        reinterpret_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0)) +
                             static_cast<size_t>(y) * GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0),
                        static_cast<size_t>(w) * 3);
        }
        gst_video_frame_unmap(&frame);
        gst_sample_unref(sample);

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
    }

    StreamSpec spec_;
    GstElement* pipeline_ = nullptr;
    GstElement* sink_ = nullptr;
    std::mutex mu_;
    uint64_t next_seq_ = 0;
    bool eos_ = false;
};

struct GstInit {
    GstInit() { gst_init(nullptr, nullptr); }
};

}  // namespace

std::unique_ptr<FrameSource> make_gst_source(const StreamSpec& s, std::string& err) {
    static GstInit init;  // idempotent
    if (s.url.empty()) {
        err = "empty url";
        return nullptr;
    }
    auto src = std::make_unique<GstSource>(s);
    // open() is called by the runtime source thread; validate eagerly here so
    // bad urls surface at add-time semantics without pulling a frame.
    if (!src->open(err)) return nullptr;
    // Reopen cleanly on the runtime's own open() call.
    src->close();
    return src;
}

}  // namespace vb
