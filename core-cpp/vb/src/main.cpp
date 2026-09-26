// vb-runtime executable (spec BASE-1 §8 M1.8).
//
//   vb-runtime --ipc-fd N | --listen PATH  --config PATH [--parity DIR]
//              [--frames N] [--version]
//
// Parity mode: adds the streams listed in the config file (shard runtime
// configs carry none, §6.9 rule 4), collects N frames and writes one JSON
// line per frame to <DIR>/parity.jsonl, then exits 0 (§7 parity runs).
#include <mutex>
#include <fcntl.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#include "pool.h"
#include "out/standalone.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/runtime.h"

namespace {

using vb::Json;
using vb::Runtime;
using vb::RuntimeConfig;
using vb::Writer;

int make_listen_socket(const char* path) {
    ::unlink(path);
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(fd, 1) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

int run_parity(const RuntimeConfig& cfg, const char* dir, int frames) {
    Writer writer;
    std::string err;
    auto backend = vb::create_backend(cfg.backend_name, cfg.backend_json, err);
    if (!backend) {
        std::fprintf(stderr, "backend: %s\n", err.c_str());
        return 1;
    }
    Runtime rt(std::move(backend), cfg, writer);
    std::vector<vb::WireFrameRec> collected;
    std::mutex mu;
    rt.on_frame_rec = [&](const vb::WireFrameRec& r) {
        std::lock_guard<std::mutex> lk(mu);
        if (collected.size() < static_cast<size_t>(frames)) collected.push_back(r);
    };
    if (!rt.start(err)) {
        std::fprintf(stderr, "start: %s\n", err.c_str());
        return 1;
    }
    uint32_t next_index = 0;
    for (const auto& sj : rt.config_streams) {
        Json add;
        add["op"] = "add";
        add["req"] = "parity-" + std::to_string(next_index);
        Json st;
        st["index"] = next_index++;
        st["id"] = sj.value("stream_id", std::string("parity"));
        st["url"] = sj.at("url").get<std::string>();
        st["transport"] = sj.value("transport", std::string("tcp"));
        if (sj.contains("score_threshold")) st["score_threshold"] = sj["score_threshold"];
        st["options"] = sj.value("options", Json::object());
        add["stream"] = st;
        add["analyzers"] = Json::array();
        rt.handle_line(add);
    }
    if (rt.config_streams.empty()) {
        Json add;
        add["op"] = "add";
        add["req"] = "parity-0";
        Json st;
        st["index"] = 0;
        st["id"] = "parity";
        st["url"] = "synthetic://";
        st["transport"] = "tcp";
        st["options"] = Json::object();
        add["stream"] = st;
        add["analyzers"] = Json::array();
        rt.handle_line(add);
    }
    double deadline = 30.0;
    double t0 = std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(mu);
            if (collected.size() >= static_cast<size_t>(frames)) break;
        }
        double now = std::chrono::duration<double>(
                         std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now - t0 > deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    rt.stop();
    std::string path = std::string(dir) + "/parity.jsonl";
    FILE* out = std::fopen(path.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return 1;
    }
    for (const auto& r : collected) {
        Json j;
        j["stream_index"] = r.stream_index;
        j["seq"] = r.seq;
        j["wall_ms"] = r.wall_ms;
        j["inference_ms"] = r.inference_ms;
        Json dets = Json::array();
        for (const auto& d : r.dets) {
            Json d2;
            d2["cx"] = d.cx;
            d2["cy"] = d.cy;
            d2["w"] = d.w;
            d2["h"] = d.h;
            d2["score"] = d.score;
            d2["class_id"] = d.class_id;
            d2["track_id"] = d.track_id;
            dets.push_back(d2);
        }
        j["detections"] = dets;
        std::fprintf(out, "%s\n", vb::json_dump(j).c_str());
    }
    std::fclose(out);
    std::fprintf(stderr, "parity: %zu frames -> %s\n", collected.size(), path.c_str());
    return collected.empty() ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
#ifdef __linux__
    // §6.9 rule 3: die with the Python shard that spawned us.
    ::prctl(PR_SET_PDEATHSIG, SIGTERM);
    // No getppid()==1 check: in a container the legitimate parent can be
    // PID 1, which made vb-runtime exit silently (M1.14 review 2026-09-26).
#endif
    const char* listen_path = nullptr;
    const char* config_path = nullptr;
    const char* parity_dir = nullptr;
    const char* output = nullptr;
    int ipc_fd = -1;
    int frames = 50;
    int frame_every = 0, status_every = -1;
    bool standalone = false;
    bool dev = false;  // §6.12 dev-mode raw tensor passthrough
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (a == "--version") {
#ifdef VB_WITH_TLS
            std::printf("%s tls:on\n", vb::runtime_version());
#else
            std::printf("%s tls:off\n", vb::runtime_version());
#endif
            return 0;
        } else if (a == "--listen") {
            listen_path = next();
        } else if (a == "--standalone") {
            standalone = true;
        } else if (a == "--dev") {
            dev = true;
        } else if (a == "--output") {
            output = next();
        } else if (a == "--frame-every") {
            frame_every = std::atoi(next());
        } else if (a == "--status-every") {
            status_every = std::atoi(next());
        } else if (a == "--config") {
            config_path = next();
        } else if (a == "--parity") {
            parity_dir = next();
        } else if (a == "--ipc-fd") {
            ipc_fd = std::atoi(next());
        } else if (a == "--frames") {
            frames = std::atoi(next());
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }
    if (argc == 1) {
        std::fprintf(stderr,
                     "usage: vb-runtime --ipc-fd N | --listen PATH --config PATH "
                     "[--parity DIR] [--frames N] [--dev] [--version]\n"
                     "       vb-runtime --standalone --config PATH "
                     "--output jsonl|mqtt [--frame-every N] [--status-every S] [--dev]\n");
        return 2;
    }
    // §6.12: production base images set VB_PRODUCTION=1; --dev is refused.
    if (dev && std::getenv("VB_PRODUCTION") != nullptr) {
        std::fprintf(stderr, "dev mode disabled in production image\n");
        return 2;
    }
    if (standalone && (listen_path || ipc_fd >= 0 || parity_dir)) {
        std::fprintf(stderr,
                     "--standalone cannot be combined with --ipc-fd/--listen/--parity\n");
        return 2;
    }
    if (standalone) {
        vb::StandaloneOpts so;
        so.config_path = config_path ? config_path : "";
        if (output) so.output = output;
        if (so.output != "jsonl" && so.output != "mqtt") {
            std::fprintf(stderr, "--output must be jsonl or mqtt\n");
            return 2;
        }
        so.frame_every = frame_every;
        so.status_every = status_every;
        so.dev = dev;
        if (!config_path) {
            std::fprintf(stderr, "--config is required\n");
            return 2;
        }
        return vb::run_standalone(so);
    }
    if (!config_path) {
        std::fprintf(stderr, "--config is required\n");
        return 2;
    }
    std::string err;
    RuntimeConfig cfg = RuntimeConfig::load(config_path, err, dev);
    if (!err.empty()) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (parity_dir) return run_parity(cfg, parity_dir, frames);

    auto backend = vb::create_backend(cfg.backend_name, cfg.backend_json, err);
    if (!backend) {
        std::fprintf(stderr, "backend: %s\n", err.c_str());
        return 1;
    }

    int listen_fd = -1, conn_fd = -1;
    if (listen_path) {
        listen_fd = make_listen_socket(listen_path);
        if (listen_fd < 0) {
            std::fprintf(stderr, "cannot listen on %s\n", listen_path);
            return 1;
        }
    } else if (ipc_fd >= 0) {
        conn_fd = ipc_fd;
    } else {
        std::fprintf(stderr, "one of --ipc-fd or --listen is required\n");
        return 2;
    }

    Writer writer;
    Runtime rt(std::move(backend), cfg, writer);
    // Config-listed streams (parity convenience; shard configs have none).
    {
        std::ifstream f(config_path);
        if (f) {
            std::stringstream ss;
            ss << f.rdbuf();
            try {
                auto j = vb::json_parse(ss.str());
                auto it = j.find("streams");
                if (it != j.end() && it->is_array())
                    for (const auto& s : *it) rt.config_streams.push_back(s);
            } catch (...) {
            }
        }
    }
    writer.start([&conn_fd](const uint8_t* data, size_t len) {
        if (conn_fd >= 0) vb::write_all_fd(conn_fd, data, len);
    });

    if (!rt.start(err)) {
        std::fprintf(stderr, "start: %s\n", err.c_str());
        return 1;
    }

    if (listen_fd >= 0) {
        conn_fd = ::accept(listen_fd, nullptr, nullptr);
        ::close(listen_fd);
        if (conn_fd < 0) {
            std::fprintf(stderr, "accept failed\n");
            return 1;
        }
    }
    rt.emit_hello();
    vb::serve_fd(conn_fd, rt);
    rt.stop();
    writer.stop();
    if (conn_fd >= 0 && listen_path) ::close(conn_fd);
    ::unlink(listen_path ? listen_path : "");
    return 0;
}
