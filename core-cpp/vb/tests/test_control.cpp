// Control protocol end-to-end over a socketpair (spec BASE-1 §8 M1.8):
// hello, add ok-reply + stream_state, VBR1 frames, stats, set_threshold,
// configure_analyzer (unknown analyzer error), snapshot (VBS1 record when
// built with JPEG, "snapshot unsupported" reply otherwise), remove, stop
// (exit within 1 s).
#include <mutex>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <thread>

#include "check.h"
#include "vb/backend.h"
#include "vb/json.h"
#include "vb/runtime.h"

using namespace vb;

namespace {

struct Peer {
    int fd;  // our end

    bool send_line(const std::string& s) {
        std::string line = s + "\n";
        const char* p = line.data();
        size_t left = line.size();
        while (left) {
            ssize_t n = ::send(fd, p, left, 0);
            if (n <= 0) return false;
            p += n;
            left -= static_cast<size_t>(n);
        }
        return true;
    }

    // Reads one wire record; returns false on EOF.
    bool read_rec(std::string& magic, std::vector<uint8_t>& body) {
        uint8_t hdr[8];
        if (!read_exact(hdr, 8)) return false;
        uint32_t len;
        std::memcpy(&len, hdr + 4, 4);
        body.resize(len);
        if (len && !read_exact(body.data(), len)) return false;
        magic.assign(hdr, hdr + 4);
        return true;
    }

    bool read_exact(uint8_t* buf, size_t n) {
        size_t got = 0;
        while (got < n) {
            ssize_t r = ::recv(fd, buf + got, n - got, 0);
            if (r <= 0) return false;
            got += static_cast<size_t>(r);
        }
        return true;
    }

    // Buffered records already read off the socket but not yet matched.
    std::vector<std::pair<std::string, Json>> pending;

    // Waits for a VBC1 record with op==want_op whose dump contains `expected`
    // (when given). Scans already-buffered records first, so ordering between
    // replies and stream_state records never loses a record.
    Json wait_control(const char* want_op, const char* key, const char* expected,
                      int timeout_ms) {
        auto match = [&](const Json& j) {
            if (!j.is_object() || !j.contains("op")) return false;
            if (j.at("op").get<std::string>() != want_op) return false;
            if (key && expected && j.dump().find(expected) == std::string::npos)
                return false;
            return true;
        };
        for (size_t i = 0; i < pending.size(); ++i) {
            if (match(pending[i].second)) {
                Json j = pending[i].second;
                pending.erase(pending.begin() + i);
                return j;
            }
        }
        int waited = 0;
        while (waited < timeout_ms) {
            std::string magic;
            std::vector<uint8_t> body;
            if (!read_rec(magic, body)) return Json();
            if (magic == "VBC1") {
                Json j = json_parse(std::string(body.begin(), body.end()));
                if (match(j)) return j;
                pending.emplace_back(magic, j);
            } else {
                pending.emplace_back(magic, Json());
            }
            waited += 5;
        }
        return Json();
    }
};

}  // namespace

int main() {
    int sv[2];
    CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    Peer peer{};
    peer.fd = sv[0];
    struct timeval tv{10, 0};
    ::setsockopt(sv[0], SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    Writer writer;
    std::string ignore;
    RuntimeConfig cfg = RuntimeConfig::from_json(
        Json{{"backend", {{"name", "synthetic"}, {"max_batch", 2}}},
             {"contexts_per_worker", 1},
             {"open_timeout_s", 3.0},
             {"reconnect_delay_s", 1.0}},
        ignore);
    std::string err;
    auto backend = create_backend(cfg.backend_name, cfg.backend_json, err);
    CHECK(backend != nullptr);
    Runtime rt(std::move(backend), cfg, writer);
    int conn = sv[1];
    writer.start([conn](const uint8_t* data, size_t len) {
        size_t off = 0;
        while (off < len) {
            ssize_t n = ::send(conn, data + off, len - off, 0);
            if (n <= 0) return;
            off += static_cast<size_t>(n);
        }
    });
    CHECK(rt.start(err));

    std::atomic<bool> server_done{false};
    std::thread server([&] {
        rt.emit_hello();
        serve_fd(sv[1], rt);
        server_done = true;
    });

    // hello
    Json hello = peer.wait_control("hello", "runtime_version", "0.1.0", 2000);
    CHECK(!hello.is_null());
    CHECK(hello.at("abi").get<int>() == 1);
    CHECK(hello.at("backend").get<std::string>() == "synthetic");
    CHECK(hello.at("caps").at("max_batch").get<int>() == 2);
    CHECK(hello.contains("pid"));
    CHECK(hello.contains("model_sha256"));
    CHECK(hello.at("stage2_ready").get<bool>() == false);

    // add
    CHECK(peer.send_line(
        R"({"op":"add","req":"r-1","stream":{"index":3,"id":"cam-04","url":"synthetic://?w=64&h=48&fps=60","transport":"tcp","score_threshold":0.35,"options":{}},"analyzers":[]})"));
    Json reply = peer.wait_control("reply", "req", "\"r-1\"", 3000);
    CHECK(!reply.is_null());
    CHECK(reply.at("ok").get<bool>() == true);
    CHECK(reply.at("applied").at("stream_index").get<int>() == 3);

    // stream_state starting -> running + >= 10 VBR1 frames
    Json st = peer.wait_control("stream_state", "state", "running", 2000);
    CHECK(!st.is_null());
    CHECK(st.at("stream_index").get<int>() == 3);
    int frames = 0;
    for (int i = 0; i < 10; ++i) {
        std::string magic;
        std::vector<uint8_t> body;
        CHECK(peer.read_rec(magic, body));
        if (magic == "VBC1") {
            peer.pending.emplace_back(
                magic, json_parse(std::string(body.begin(), body.end())));
        } else if (magic == "VBR1") {
            ++frames;
            uint32_t idx;
            std::memcpy(&idx, body.data(), 4);
            CHECK(idx == 3);
        }
    }
    CHECK(frames == 10);

    // stats once a second
    Json stats = peer.wait_control("stats", "rss_kb", "rss_kb", 2500);
    CHECK(!stats.is_null());
    CHECK(stats.contains("cpu_s"));
    CHECK(stats.at("streams").size() == 1);
    CHECK(stats.at("streams")[0].at("state").get<std::string>() == "running");
    CHECK(stats.at("streams")[0].at("decode").get<std::string>() == "synthetic");

    // set_threshold: reply carries the applied value
    CHECK(peer.send_line(R"({"op":"set_threshold","req":"r-2","stream_index":3,"value":0.5})"));
    Json thr = peer.wait_control("reply", "req", "\"r-2\"", 3000);
    CHECK(!thr.is_null());
    CHECK(thr.at("ok").get<bool>() == true);
    CHECK_NEAR(thr.at("applied").at("value").get<double>(), 0.5, 1e-6);

    // configure_analyzer: no analyzer of that name
    CHECK(peer.send_line(
        R"({"op":"configure_analyzer","req":"r-3","stream_index":3,"name":"line_cross","config":{}})"));
    Json ca = peer.wait_control("reply", "req", "\"r-3\"", 3000);
    CHECK(!ca.is_null());
    CHECK(ca.at("ok").get<bool>() == false);
    CHECK(ca.at("error").get<std::string>() ==
          "no analyzer line_cross on stream 3");

    // snapshot: the branch depends on whether JPEG encoding was compiled in
    // (VB_WITH_JPEG). Without it the op is refused with a control reply; with
    // it the newest ring frame is encoded and pushed as an out-of-band VBS1
    // record on the data channel, and no VBC1 reply is sent at all — so the
    // test must follow the branch this build actually compiles (§6.10.3).
    CHECK(peer.send_line(
        R"({"op":"snapshot","req":"r-4","stream_index":3,"seq":0,"track_id":0,"crop":true,"max_side":320})"));
#if defined(VB_HAVE_JPEG)
    bool got_vbs1 = false;
    for (;;) {
        std::string magic;
        std::vector<uint8_t> body;
        if (!peer.read_rec(magic, body)) break;
        if (magic == "VBC1") {
            Json j = json_parse(std::string(body.begin(), body.end()));
            // VBS1 replaces the reply: a control reply for r-4 would mean the
            // snapshot silently failed instead of being encoded.
            CHECK(j.dump().find("\"r-4\"") == std::string::npos);
            peer.pending.emplace_back(magic, j);
        } else if (magic == "VBS1") {
            got_vbs1 = true;
            uint32_t meta_len = 0;
            CHECK(body.size() >= 4);
            std::memcpy(&meta_len, body.data(), 4);
            CHECK(body.size() >= 4u + meta_len);
            Json meta = json_parse(std::string(body.begin() + 4,
                                               body.begin() + 4 + meta_len));
            CHECK(meta.at("req").get<std::string>() == "r-4");
            CHECK(meta.at("mime").get<std::string>() == "image/jpeg");
            CHECK(meta.at("seq").get<uint64_t>() > 0);  // a real ring frame
            CHECK(meta.at("w").get<int>() > 0);
            CHECK(meta.at("h").get<int>() > 0);
            // Payload is a JPEG (SOI marker), not an empty buffer.
            const size_t payload = body.size() - 4 - meta_len;
            CHECK(payload > 2);
            CHECK(body[4 + meta_len] == 0xFF && body[5 + meta_len] == 0xD8);
            break;
        }
        // VBR1 frames simply keep streaming while we look for the record.
    }
    CHECK(got_vbs1);
#else
    Json snap = peer.wait_control("reply", "req", "\"r-4\"", 3000);
    CHECK(!snap.is_null());
    CHECK(snap.at("ok").get<bool>() == false);
    CHECK(snap.at("error").get<std::string>() == "snapshot unsupported");
#endif

    // remove -> ok + stopped
    CHECK(peer.send_line(R"({"op":"remove","req":"r-5","stream_index":3})"));
    Json rm = peer.wait_control("reply", "req", "\"r-5\"", 3000);
    CHECK(!rm.is_null());
    CHECK(rm.at("ok").get<bool>() == true);
    Json stopped = peer.wait_control("stream_state", "state", "stopped", 2000);
    CHECK(!stopped.is_null());

    // bad json -> reply ok:false, connection stays
    CHECK(peer.send_line(R"({"op": not json})"));
    Json bad = peer.wait_control("reply", "error", "bad json", 2000);
    CHECK(!bad.is_null());
    CHECK(bad.at("ok").get<bool>() == false);

    // stop -> reply then server returns within 1 s
    auto t0 = std::chrono::steady_clock::now();
    CHECK(peer.send_line(R"({"op":"stop","req":"r-9"})"));
    Json stop_reply = peer.wait_control("reply", "req", "\"r-9\"", 2000);
    CHECK(!stop_reply.is_null());
    CHECK(stop_reply.at("ok").get<bool>() == true);
    server.join();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    CHECK(dt < 2000);
    (void)server_done;

    rt.stop();
    writer.stop();
    ::close(sv[0]);
    ::close(sv[1]);
    std::printf("test_control: all OK (stop took %lld ms)\n", (long long)dt);
    return 0;
}
