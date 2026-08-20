// Retail people-flow detection on Raspberry Pi 5 + Hailo-8.
//
// Hot path is native: GStreamer pulls RTSP, hailonet runs the detection HEF on
// the NPU, this process reads the on-chip NMS output, tracks people, runs the
// dwell state machine and publishes one batched VisionPayload per cycle.
// No Torch, no Ultralytics, no ONNX Runtime, no Python anywhere in the loop.
//
// The tracker, dwell state machine and payload construction are ports of the
// C++ reference in sscma-example-sg200x/solutions/retail-vision; only the frame
// source and the inference backend differ.

#include <gst/gst.h>
#include <gst/hailo/tensor_meta.hpp>

#ifdef HAVE_MOSQUITTO
#include <mosquitto.h>
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "detector.h"
#include "mqtt_payload.h"
#include "person_tracker.h"
#include "zone_metrics.h"

using Clock = std::chrono::steady_clock;

namespace {

std::string env(const char* n, const char* d = "") {
    const char* v = std::getenv(n);
    return (v && *v) ? std::string(v) : std::string(d);
}
float envF(const char* n, float d) {
    const std::string v = env(n);
    return v.empty() ? d : std::stof(v);
}
int envI(const char* n, int d) {
    const std::string v = env(n);
    return v.empty() ? d : std::stoi(v);
}

double now() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
uint64_t epochMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

struct App;

struct Stream {
    std::string id;
    std::string url;
    std::string topic;
    GstElement* pipeline = nullptr;
    App* app = nullptr;

    retail_vision::PersonTracker tracker;
    retail_vision::ZoneMetrics zone;

    // Source resolution, learned from the caps event upstream of the scaler.
    std::atomic<int> src_width{0};
    std::atomic<int> src_height{0};

    uint32_t frame_id = 0;
    uint64_t frames_total = 0;
    uint64_t frames_since_publish = 0;
    uint64_t messages = 0;
    double started = 0.0;
    double last_publish = 0.0;
    double latency_sum_since_publish = 0.0;
    std::atomic<double> input_at{0.0};
    bool logged_tensors = false;
};

struct App {
#ifdef HAVE_MOSQUITTO
    mosquitto* mqtt = nullptr;
#endif
    std::string status_topic;
    float score = 0.4f;
    int person_class = 0;
    double publish_interval = 1.0;
    int model_w = 640, model_h = 640;
    std::vector<std::unique_ptr<Stream>> streams;
    GMainLoop* loop = nullptr;
};

App* g_app = nullptr;

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t end = s.find(sep, start);
        if (end == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

std::string replaceAll(std::string s, const std::string& a, const std::string& b) {
    size_t p;
    while ((p = s.find(a)) != std::string::npos) s.replace(p, a.size(), b);
    return s;
}

void publishStatus(App& app, const char* state) {
#ifdef HAVE_MOSQUITTO
    if (!app.mqtt) return;
    // Retained so a dashboard subscribing after the fact still learns the state,
    // and matching the LWT payload so an unclean exit reads the same as a clean
    // one from the consumer's side.
    mosquitto_publish(app.mqtt, nullptr, app.status_topic.c_str(),
                      static_cast<int>(std::strlen(state)), state, 1, true);
#else
    (void)app;
    (void)state;
#endif
}

void onSignal(int) {
    if (g_app && g_app->loop) g_main_loop_quit(g_app->loop);
}

// Records the true source resolution so cx_pct / cy_pct and the letterbox
// correction refer to the camera frame rather than the 640x640 model square.
GstPadProbeReturn capsProbe(GstPad*, GstPadProbeInfo* info, gpointer data) {
    GstEvent* ev = GST_PAD_PROBE_INFO_EVENT(info);
    if (!ev || GST_EVENT_TYPE(ev) != GST_EVENT_CAPS) return GST_PAD_PROBE_OK;
    GstCaps* caps = nullptr;
    gst_event_parse_caps(ev, &caps);
    if (!caps || gst_caps_get_size(caps) == 0) return GST_PAD_PROBE_OK;
    GstStructure* st = gst_caps_get_structure(caps, 0);
    gint w = 0, h = 0;
    if (gst_structure_get_int(st, "width", &w) && gst_structure_get_int(st, "height", &h)) {
        auto& s = *static_cast<Stream*>(data);
        s.src_width = w;
        s.src_height = h;
    }
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn preProbe(GstPad*, GstPadProbeInfo*, gpointer data) {
    static_cast<Stream*>(data)->input_at = now();
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn outProbe(GstPad*, GstPadProbeInfo* info, gpointer data) {
    auto& s = *static_cast<Stream*>(data);
    App& app = *s.app;
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    std::vector<retail_vision::RawTensor> tensors;
    struct Mapping {
        GstBuffer* b;
        GstMapInfo m;
    };
    std::vector<Mapping> maps;

    gpointer state = nullptr;
    GstMeta* meta = nullptr;
    while ((meta = gst_buffer_iterate_meta_filtered(buf, &state, GST_PARENT_BUFFER_META_API_TYPE))) {
        auto* p = reinterpret_cast<GstParentBufferMeta*>(meta);
        GstMapInfo mi{};
        if (!gst_buffer_map(p->buffer, &mi, GST_MAP_READ)) continue;
        auto* tm = GST_TENSOR_META_GET(p->buffer);
        if (tm) {
            maps.push_back({p->buffer, mi});
            tensors.push_back({mi.data, mi.size, tm->info});
        } else {
            gst_buffer_unmap(p->buffer, &mi);
        }
    }

    if (!s.logged_tensors && !tensors.empty()) {
        s.logged_tensors = true;
        std::cout << "stream " << s.id << " hailo outputs: "
                  << retail_vision::describeTensors(tensors) << std::endl;
        if (!retail_vision::hasNmsOutput(tensors)) {
            std::cerr << "stream " << s.id
                      << ": HEF has no on-chip NMS output; this build only decodes "
                         "NMS-postprocess HEFs" << std::endl;
        }
    }

    const double t_infer = now();
    auto dets = retail_vision::decodeHailoNms(tensors, app.score, app.person_class);
    for (auto& m : maps) gst_buffer_unmap(m.b, &m.m);

    const double t = now();
    const float t_sec = static_cast<float>(t - s.started);

    auto persons = s.tracker.update(dets, t_sec);
    s.zone.update(s.tracker.getStateCounts(), s.tracker.getEntryCount(),
                  s.tracker.getExitCount(), t_sec);

    ++s.frame_id;
    ++s.frames_total;
    ++s.frames_since_publish;
    s.latency_sum_since_publish += (t_infer - s.input_at.load()) * 1000.0;

    // One message per publish cycle. The tracker still runs at frame rate so
    // velocity and dwell timing keep their resolution; only the reporting is
    // decimated.
    if (t - s.last_publish < app.publish_interval) return GST_PAD_PROBE_OK;

    const double elapsed = t - s.last_publish;
    const float fps = elapsed > 0 ? static_cast<float>(s.frames_since_publish / elapsed) : 0.f;
    const float mean_latency =
        s.frames_since_publish ? static_cast<float>(s.latency_sum_since_publish / s.frames_since_publish) : 0.f;

    int fw = s.src_width.load();
    int fh = s.src_height.load();
    if (fw <= 0 || fh <= 0) {
        fw = app.model_w;
        fh = app.model_h;
    }

    const std::string json = retail_vision::buildVisionJson(
        epochMs(), s.frame_id, fps, mean_latency, s.zone.getSnapshot(), persons,
        fw, fh, app.model_w, app.model_h);

#ifdef HAVE_MOSQUITTO
    if (app.mqtt) {
        mosquitto_publish(app.mqtt, nullptr, s.topic.c_str(),
                          static_cast<int>(json.size()), json.data(), 0, false);
    } else {
        std::cout << json << std::endl;
    }
#else
    std::cout << json << std::endl;
#endif

    ++s.messages;
    s.last_publish = t;
    s.frames_since_publish = 0;
    s.latency_sum_since_publish = 0.0;
    return GST_PAD_PROBE_OK;
}

gboolean busCb(GstBus*, GstMessage* m, gpointer data) {
    auto* s = static_cast<Stream*>(data);
    if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_ERROR) {
        GError* e = nullptr;
        gchar* d = nullptr;
        gst_message_parse_error(m, &e, &d);
        std::cerr << "pipeline " << s->id << " error: " << e->message << std::endl;
        g_error_free(e);
        g_free(d);
        g_main_loop_quit(s->app->loop);
    } else if (GST_MESSAGE_TYPE(m) == GST_MESSAGE_EOS) {
        std::cerr << "pipeline " << s->id << ": end of stream" << std::endl;
        g_main_loop_quit(s->app->loop);
    }
    return TRUE;
}

gboolean finish(gpointer data) {
    auto* a = static_cast<App*>(data);
    for (auto& s : a->streams) {
        const double sec = now() - s->started;
        std::cout << "BENCHMARK stream=" << s->id << " frames=" << s->frames_total
                  << " seconds=" << sec << " fps=" << (sec > 0 ? s->frames_total / sec : 0)
                  << " messages=" << s->messages
                  << " msg_per_sec=" << (sec > 0 ? s->messages / sec : 0) << std::endl;
    }
    g_main_loop_quit(a->loop);
    return G_SOURCE_REMOVE;
}

void applySpatialConfig(Stream& s) {
    // COUNT_ZONE: "x1,y1;x2,y2;..." normalized polygon, foot-point membership.
    const std::string zone = env("COUNT_ZONE");
    if (!zone.empty()) {
        std::vector<retail_vision::geom::Point> poly;
        for (const auto& pt : split(zone, ';')) {
            auto xy = split(pt, ',');
            if (xy.size() == 2) poly.push_back({std::stof(xy[0]), std::stof(xy[1])});
        }
        s.tracker.setCountZone(poly);
    }
    // ENTRY_LINE: "ax,ay,bx,by[,ab_in]" normalized directed segment.
    const std::string line = env("ENTRY_LINE");
    if (!line.empty()) {
        auto v = split(line, ',');
        if (v.size() >= 4) {
            const bool ab_in = (v.size() < 5) || (v[4] == "1" || v[4] == "true");
            s.tracker.setEntryLine({std::stof(v[0]), std::stof(v[1])},
                                   {std::stof(v[2]), std::stof(v[3])}, ab_in);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    gst_init(&argc, &argv);
#ifdef HAVE_MOSQUITTO
    mosquitto_lib_init();
#endif

    App app;
    g_app = &app;

    const std::string installation = env("INSTALLATION_ID", "store-01");
    const std::string topic_tpl =
        env("MQTT_TOPIC", (installation + "/retail-vision/results/{camera_id}").c_str());
    app.status_topic = env("MQTT_STATUS_TOPIC", (installation + "/retail-vision/status").c_str());
    app.score = envF("SCORE_THRESHOLD", 0.4f);
    app.person_class = envI("PERSON_CLASS_ID", 0);
    const float hz = envF("PUBLISH_HZ", 1.0f);
    app.publish_interval = hz > 0 ? 1.0 / hz : 1.0;

#ifdef HAVE_MOSQUITTO
    app.mqtt = mosquitto_new(nullptr, true, nullptr);
    if (app.mqtt) {
        const std::string user = env("MQTT_USERNAME"), pass = env("MQTT_PASSWORD");
        if (!user.empty()) mosquitto_username_pw_set(app.mqtt, user.c_str(), pass.c_str());
        // LWT before connect: the broker publishes "offline" retained if this
        // process dies without a clean disconnect.
        mosquitto_will_set(app.mqtt, app.status_topic.c_str(), 7, "offline", 1, true);
        if (mosquitto_connect(app.mqtt, env("MQTT_HOST", "127.0.0.1").c_str(),
                              envI("MQTT_PORT", 1883), 30) == MOSQ_ERR_SUCCESS) {
            mosquitto_loop_start(app.mqtt);
            publishStatus(app, "online");
        } else {
            std::cerr << "mqtt connect failed; falling back to stdout" << std::endl;
            mosquitto_destroy(app.mqtt);
            app.mqtt = nullptr;
        }
    }
#endif

    const std::string list = env("STREAMS", "cam-01|test://");
    for (const auto& item : split(list, ';')) {
        if (item.empty()) continue;
        const size_t sep = item.find('|');
        if (sep == std::string::npos) continue;
        auto s = std::make_unique<Stream>();
        s->id = item.substr(0, sep);
        s->url = item.substr(sep + 1);
        s->topic = replaceAll(topic_tpl, "{camera_id}", s->id);
        s->app = &app;

        retail_vision::TrackerConfig cfg;
        cfg.frame_width = app.model_w;
        cfg.frame_height = app.model_h;
        cfg.dwell_speed_threshold = envF("DWELL_SPEED_THRESHOLD", cfg.dwell_speed_threshold);
        cfg.dwell_min_duration = envF("DWELL_MIN_DURATION", cfg.dwell_min_duration);
        cfg.dwell_assistance_threshold =
            envF("DWELL_ASSISTANCE_THRESHOLD", cfg.dwell_assistance_threshold);
        cfg.iou_threshold = envF("IOU_THRESHOLD", cfg.iou_threshold);
        s->tracker.setConfig(cfg);

        s->zone.setWindowDuration(envF("ZONE_WINDOW_SEC", 60.0f));
        auto* zone_ptr = &s->zone;
        s->tracker.setTrackRemovedCallback(
            [zone_ptr](const retail_vision::TrackRecord& r) { zone_ptr->onTrackRemoved(r); });
        applySpatialConfig(*s);

        app.streams.push_back(std::move(s));
    }

    if (app.streams.empty()) {
        std::cerr << "STREAMS must contain camera-id|url entries" << std::endl;
        return 2;
    }

    const std::string hef = env("HEF_PATH", "/models/yolov8n.hef");
    const int latency = envI("RTSP_LATENCY_MS", 100);
    app.loop = g_main_loop_new(nullptr, FALSE);
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    for (auto& sp : app.streams) {
        // A live camera must never block the pipeline, so its sink is unsynced.
        // A file replays as fast as the NPU can drain it (measured: 234 fps),
        // which is the right way to benchmark throughput but the wrong way to
        // observe behaviour: at 15x speed people jump between frames, IoU
        // matching fails and the tracker churns ids instead of dwelling. So a
        // file source is paced to its own timestamps unless told otherwise.
        const bool is_file = sp->url.rfind("file://", 0) == 0;
        const std::string sink_sync = env("SINK_SYNC", is_file ? "true" : "false");
        // The queue policy has to agree with the sink policy. A live source must
        // never block, so its queue leaks; but a leaky queue in front of a
        // *blocking* sink discards nearly every frame, which showed up as 1 Hz
        // publishes separated by 10 s gaps. A paced file source therefore gets a
        // short non-leaking queue and simply runs at the media's frame rate.
        const std::string queue_cfg = is_file
            ? "queue max-size-buffers=8"
            : "queue max-size-buffers=2 leaky=downstream";

        std::ostringstream q;
        if (sp->url.rfind("test://", 0) == 0) {
            q << "videotestsrc is-live=true pattern=ball ! video/x-raw,framerate=30/1 ";
        } else if (sp->url.rfind("file://", 0) == 0) {
            q << "uridecodebin uri=\"" << sp->url << "\" ! videorate ! video/x-raw,framerate=15/1 ";
        } else {
            // h264parse lives in gstreamer1.0-plugins-bad; without it
            // gst_parse_launch fails to link the RTSP branch entirely.
            q << "rtspsrc location=\"" << sp->url << "\" latency=" << latency
              << " protocols=tcp ! rtph264depay ! h264parse ! decodebin ";
        }
        q << "! videoconvert ! identity name=raw ! videoscale add-borders=true "
             "! video/x-raw,format=RGB,width="
          << app.model_w << ",height=" << app.model_h
          << ",pixel-aspect-ratio=1/1 "
             "! "
          << queue_cfg
          << " ! identity name=pre "
             "! hailonet name=net hef-path=\""
          << hef
          << "\" scheduling-algorithm=1 vdevice-group-id=retail-shared ! fakesink sync="
          << sink_sync;

        GError* e = nullptr;
        sp->pipeline = gst_parse_launch(q.str().c_str(), &e);
        if (!sp->pipeline) {
            std::cerr << "pipeline build failed: " << (e ? e->message : "unknown") << std::endl;
            return 3;
        }

        auto* raw = gst_bin_get_by_name(GST_BIN(sp->pipeline), "raw");
        auto* pre = gst_bin_get_by_name(GST_BIN(sp->pipeline), "pre");
        auto* net = gst_bin_get_by_name(GST_BIN(sp->pipeline), "net");
        auto* rawpad = gst_element_get_static_pad(raw, "sink");
        auto* prepad = gst_element_get_static_pad(pre, "src");
        auto* netpad = gst_element_get_static_pad(net, "src");
        gst_pad_add_probe(rawpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, capsProbe, sp.get(), nullptr);
        gst_pad_add_probe(prepad, GST_PAD_PROBE_TYPE_BUFFER, preProbe, sp.get(), nullptr);
        gst_pad_add_probe(netpad, GST_PAD_PROBE_TYPE_BUFFER, outProbe, sp.get(), nullptr);
        gst_object_unref(rawpad);
        gst_object_unref(prepad);
        gst_object_unref(netpad);
        gst_object_unref(raw);
        gst_object_unref(pre);
        gst_object_unref(net);

        auto* bus = gst_element_get_bus(sp->pipeline);
        gst_bus_add_watch(bus, busCb, sp.get());
        gst_object_unref(bus);

        sp->started = now();
        sp->last_publish = sp->started;
        gst_element_set_state(sp->pipeline, GST_STATE_PLAYING);
        std::cout << "stream " << sp->id << " -> topic " << sp->topic << " (" << sp->url << ")"
                  << std::endl;
    }

    const int seconds = envI("BENCHMARK_SECONDS", 0);
    if (seconds > 0) g_timeout_add_seconds(seconds, finish, &app);
    g_main_loop_run(app.loop);

    for (auto& sp : app.streams) {
        gst_element_set_state(sp->pipeline, GST_STATE_NULL);
        gst_object_unref(sp->pipeline);
    }

#ifdef HAVE_MOSQUITTO
    if (app.mqtt) {
        publishStatus(app, "offline");
        // Disconnect first, then join: mosquitto_disconnect lets the network
        // thread flush the retained "offline" before the loop is torn down.
        mosquitto_disconnect(app.mqtt);
        mosquitto_loop_stop(app.mqtt, false);
        mosquitto_destroy(app.mqtt);
    }
    mosquitto_lib_cleanup();
#endif
    g_main_loop_unref(app.loop);
    return 0;
}
