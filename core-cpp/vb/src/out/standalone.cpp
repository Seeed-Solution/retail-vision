// Standalone mode: single-process vb-runtime with a JSONL or MQTT output
// adapter instead of the IPC socket (spec BASE-1 §6.10, M1.18/M1.19).
//
// The Writer's output (VBR1/VBE1 wire records) is consumed here and turned
// into vb.event/1 / vb.frame/1 records; status records are produced by a
// local timer thread from the runtime metrics. Logs go to stderr only.
#include "out/standalone.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "out/event_json.h"
#include "out/jsonl_out.h"
#include "out/mqtt_lite.h"
#include "pool.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/runtime.h"

namespace vb {

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

double mono_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

// Little-endian readers for VBR1 bodies.
uint32_t rd_u32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24; }
uint64_t rd_u64(const uint8_t* p) {
    return rd_u32(p) | uint64_t(rd_u32(p + 4)) << 32;
}

bool decode_vbr1(const uint8_t* p, size_t len, WireFrameRec& r) {
    if (len < 64) return false;
    auto rd_f32 = [&](size_t off) {
        uint32_t b = rd_u32(p + off);
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    };
    r = WireFrameRec{};
    r.stream_index = rd_u32(p);
    r.seq = rd_u64(p + 4);
    std::memcpy(&r.wall_ms, p + 12, 8);  // f64 LE
    r.src_w = int32_t(rd_u32(p + 20));
    r.src_h = int32_t(rd_u32(p + 24));
    r.model_w = int32_t(rd_u32(p + 28));
    r.model_h = int32_t(rd_u32(p + 32));
    r.scale = rd_f32(36);
    r.pad_x = rd_f32(40);
    r.pad_y = rd_f32(44);
    r.align = p[48];
    r.kpt_per_det = p[49];
    uint16_t n_det = uint16_t(rd_u32(p + 50) & 0xffff);
    r.inference_ms = rd_f32(52);
    r.queue_delay_ms = rd_f32(56);
    r.attr_per_det = p[60];
    size_t off = 64;
    if (off + size_t(n_det) * 28 > len) return false;
    for (uint16_t i = 0; i < n_det; ++i) {
        WireDet d;
        d.cx = rd_f32(off);
        d.cy = rd_f32(off + 4);
        d.w = rd_f32(off + 8);
        d.h = rd_f32(off + 12);
        d.score = rd_f32(off + 16);
        d.class_id = int32_t(rd_u32(p + off + 20));
        d.track_id = rd_u32(p + off + 24);
        r.dets.push_back(d);
        off += 28;
    }
    size_t nk = size_t(n_det) * r.kpt_per_det * 3;
    size_t na = size_t(n_det) * r.attr_per_det;
    if (off + (nk + na) * 4 > len) return false;
    for (size_t i = 0; i < nk; ++i) r.kpts.push_back(rd_f32(off + i * 4));
    off += nk * 4;
    for (size_t i = 0; i < na; ++i) r.attrs.push_back(rd_f32(off + i * 4));
    return true;
}

struct StandaloneApp {
    const StandaloneOpts& opts;
    std::string device_id;
    std::map<uint32_t, std::string> stream_ids;      // index -> stream_id
    std::map<uint32_t, uint64_t> frame_counts;       // --frame-every counters
    JsonlOut jsonl;
    std::unique_ptr<MqttLite> mqtt;

    void on_record(const uint8_t* data, size_t len) {
        if (len < 8) return;
        char magic[5] = {char(data[0]), char(data[1]), char(data[2]), char(data[3]), 0};
        const uint8_t* body = data + 8;
        size_t body_len = len - 8;
        std::string sid;
        auto id_of = [&](uint32_t index) {
            auto it = stream_ids.find(index);
            return it == stream_ids.end() ? std::string() : it->second;
        };
        if (std::strcmp(magic, "VBE1") == 0) {
            try {
                Json body_j = json_parse(std::string(
                    reinterpret_cast<const char*>(body), body_len));
                std::string sid2 = id_of(body_j.value("stream_index", uint32_t(0)));
                Json j = event_json(device_id, sid2, body_j);
                emit("events/" + sid2, json_dump(j), 1, false);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "standalone: bad VBE1: %s\n", e.what());
            }
        } else if (std::strcmp(magic, "VBR1") == 0 && opts.frame_every > 0) {
            WireFrameRec r;
            if (!decode_vbr1(body, body_len, r)) {
                std::fprintf(stderr, "standalone: bad VBR1 record\n");
                return;
            }
            sid = id_of(r.stream_index);
            uint64_t& n = frame_counts[r.stream_index];
            if (++n % uint64_t(opts.frame_every) != 0) return;
            Json j = frame_json(device_id, sid, r);
            emit("frames/" + sid, json_dump(j), 0, false);
        }
        // VBC1/VBS1 (replies, snapshots) are not part of standalone output;
        // VBT1 cannot reach this sink because run_standalone refuses --dev.
    }

    void emit(const std::string& topic_suffix, const std::string& line, int qos,
              bool retain) {
        if (opts.output == "mqtt" && mqtt) {
            mqtt->publish(topic_suffix, line, qos, retain);
        } else {
            jsonl.write(line);
        }
    }
};

}  // namespace

int run_standalone(const StandaloneOpts& opts) {
    // §6.12 dev mode exists to hand raw tensors (VBT1) to a Python consumer.
    // standalone has no tensor output adapter, so a VBT1 record arriving here
    // was silently dropped while the operator believed `--dev` produced
    // tensors. Refuse the combination up front instead of ignoring it.
    if (opts.dev) {
        std::fprintf(stderr,
                     "standalone: --dev is not supported (VBT1 tensor output "
                     "requires --ipc-fd)\n");
        return 2;
    }
    std::string err;
    Json cfg;
    try {
        std::ifstream f(opts.config_path);
        if (!f) {
            std::fprintf(stderr, "cannot open config: %s\n", opts.config_path.c_str());
            return 1;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        cfg = json_parse(ss.str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "config: %s\n", e.what());
        return 1;
    }
    std::string schema = cfg.value("schema", std::string());
    if (schema != "vb.config/1") {
        std::fprintf(stderr, "config: schema must be \"vb.config/1\"\n");
        return 1;
    }
    std::string device_id = cfg.value("device_id", std::string());
    const Json& streams = cfg.contains("streams") && cfg.at("streams").is_array()
                              ? cfg.at("streams") : Json::array();
    if (streams.empty()) {
        std::fprintf(stderr, "standalone: config has no streams\n");
        return 1;
    }

    RuntimeConfig rcfg = RuntimeConfig::from_json(cfg, err, opts.dev);
    if (!err.empty()) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    double status_interval = 10.0;
    MqttConfig mcfg;
    if (opts.output == "mqtt") {
        const Json* m = nullptr;
        if (cfg.contains("mqtt") && cfg.at("mqtt").is_object()) m = &cfg.at("mqtt");
        if (!m) {
            std::fprintf(stderr, "standalone: --output mqtt requires an mqtt config section\n");
            return 1;
        }
        mcfg.host = m->value("host", std::string());
        mcfg.port = uint16_t(m->value("port", 1883));
        mcfg.client_id = m->value("client_id", std::string());
        mcfg.username = m->value("username", std::string());
        mcfg.password = m->value("password", std::string());
        mcfg.topic_root = m->value("topic_root", std::string());
        mcfg.keepalive_s = m->value("keepalive_s", 30.0);
        mcfg.tls = m->value("tls", false);
        mcfg.ca_file = m->value("ca_file", std::string());
        mcfg.cert_file = m->value("cert_file", std::string());
        mcfg.key_file = m->value("key_file", std::string());
        status_interval = m->value("status_interval_s", 10.0);
    }
    if (opts.status_every > 0) status_interval = opts.status_every;
    mcfg.reconnect_min_s = 1.0;
    mcfg.reconnect_max_s = 60.0;

    auto backend = create_backend(rcfg.backend_name, rcfg.backend_json, err);
    if (!backend) {
        std::fprintf(stderr, "backend: %s\n", err.c_str());
        return 1;
    }

    StandaloneApp app{opts, device_id, {}, {}, {}, nullptr};
    if (opts.output == "mqtt") {
        app.mqtt = std::make_unique<MqttLite>(mcfg);
        if (!app.mqtt->start(err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    int64_t session_ms = int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    std::string session_id = std::to_string(session_ms);

    Writer writer;
    Runtime rt(std::move(backend), rcfg, writer);
    rt.on_frame_rec = nullptr;
    if (!rt.start(err)) {
        std::fprintf(stderr, "start: %s\n", err.c_str());
        return 1;
    }

    // Add config streams in order (§6.10.1); analyzers come from
    // streams[i].options.analyzers. A failed add is logged and skipped.
    size_t added = 0;
    for (size_t i = 0; i < streams.size(); ++i) {
        const Json& sj = streams[i];
        Json add;
        add["op"] = "add";
        add["req"] = "standalone-" + std::to_string(i);
        Json st;
        st["index"] = static_cast<uint32_t>(i);
        st["id"] = sj.value("stream_id", std::string("stream-") + std::to_string(i));
        st["url"] = sj.at("url").get<std::string>();
        st["transport"] = sj.value("transport", std::string("tcp"));
        if (sj.contains("score_threshold")) st["score_threshold"] = sj["score_threshold"];
        st["options"] = sj.value("options", Json::object());
        Json analyzers = Json::array();
        if (st["options"].contains("analyzers") && st["options"]["analyzers"].is_array())
            analyzers = st["options"]["analyzers"];
        add["stream"] = st;
        add["analyzers"] = analyzers;
        app.stream_ids[static_cast<uint32_t>(i)] = st["id"].get<std::string>();
        std::string add_err;
        // Capture the reply to detect failure (standalone has no control
        // socket; replay through handle_line and watch the ack).
        struct AddWatch {
            std::string req;
            bool ok = false;
            bool seen = false;
        } watch{add["req"].get<std::string>(), false, false};
        rt.on_reply_record = [&watch](const Json& j) {
            if (j.value("req", std::string()) == watch.req) {
                watch.seen = true;
                watch.ok = j.value("ok", false);
            }
        };
        rt.handle_line(add);
        rt.on_reply_record = nullptr;
        if (!watch.seen || !watch.ok) {
            std::fprintf(stderr, "standalone: add stream %s failed\n",
                         st["id"].get<std::string>().c_str());
            app.stream_ids.erase(static_cast<uint32_t>(i));
        } else {
            ++added;
        }
    }
    if (added == 0) {
        rt.stop();
        std::fprintf(stderr, "standalone: no stream could be added\n");
        return 1;
    }

    auto status_json = [&](bool online) {
        Json j;
        j["schema"] = "vb.status/1";
        j["timestamp"] = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        j["session_id"] = session_id;
        j["device_id"] = app.device_id;
        j["online"] = online;
        j["shards"] = 1;
        j["uptime_s"] = (std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch()).count() -
                         session_ms) / 1000.0;
        Json ss = Json::array();
        for (auto& [index, id] : app.stream_ids) {
            auto s = rt.stream(index);
            if (!s) continue;
            Json e = s->metrics.to_json(index);
            e["stream_id"] = id;
            e["fallback_active"] = false;
            e.erase("stream_index");
            ss.push_back(e);
        }
        j["streams"] = ss;
        if (app.mqtt && app.mqtt->dropped() > 0)
            j["mqtt_dropped"] = app.mqtt->dropped();
        return j;
    };
    auto emit_status = [&](bool online) {
        Json j = status_json(online);
        app.emit("status", json_dump(j), 1, true);
    };

    // Writer output -> standalone records (replaces the IPC socket sink).
    writer.start([&](const uint8_t* data, size_t len) {
        app.on_record(data, len);
    });

    double next_status = mono_s() + status_interval;
    while (g_stop == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (mono_s() >= next_status) {
            emit_status(true);
            next_status = mono_s() + status_interval;
        }
    }

    Json final_status = status_json(false);  // snapshot before streams stop
    rt.stop();
    writer.stop();
    // Last record: online:false (§6.10.1).
    app.emit("status", json_dump(final_status), 1, true);
    if (app.mqtt) {
        app.mqtt->stop();  // retained online:false + DISCONNECT
    }
    return 0;
}

}  // namespace vb
