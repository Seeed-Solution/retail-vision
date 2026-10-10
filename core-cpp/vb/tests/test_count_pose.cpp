// M1.17 tests: count_threshold / pose_angle external fixtures + the
// synthetic-backend add rejection for pose_angle (caps.keypoints = 0, spec
// BASE-1 §6.2.5) and the min_keypoints() base default.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "analyzer_fixture.h"
#include "check.h"
#include "vb/analyzer.h"
#include "vb/json.h"
#include "vb/runtime.h"

using namespace vb;

namespace {

const char* kCountFixtures[] = {
    "count_basic.json",      // max_count=2: 3 people for 2 s -> count_over
    "count_under.json",      // min_count=1: empty scene -> count_under
    "count_zones.json",      // two zones judged independently
    "count_coast.json",      // misses>0 coasting tracks not counted
    "count_notracker.json",  // track_id=0 wrapped detections usable
};
const char* kPoseFixtures[] = {
    "pose_angle_basic.json",    // collinear 180 deg, right angle 90 deg
    "pose_angle_pixels.json",   // 1920x1080: pixel-space angle differs from naive
    "pose_angle_lowconf.json",  // low-confidence keypoints skipped, no state change
    "pose_angle_boundary.json", // max_deg=160 boundary
};

void run_fixtures(const char* analyzer, const char* const* fixtures, size_t n,
                  const std::string& dir) {
    for (size_t i = 0; i < n; ++i) {
        std::string err;
        auto a = create_analyzer(analyzer, err);
        CHECK(a != nullptr);
        auto j = json_parse(read_file(dir + "/" + fixtures[i]));
        auto r = vb_fixture::run_analyzer_fixture(*a, j);
        if (!r.ok) {
            std::fprintf(stderr, "%s: FAIL: %s\n", fixtures[i], r.fail.c_str());
            std::exit(1);
        }
        std::printf("  %s: OK\n", fixtures[i]);
    }
}

// Minimal control-protocol peer (same idea as test_control.cpp).
struct Peer {
    int fd;
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
    bool read_exact(uint8_t* buf, size_t n) {
        size_t got = 0;
        while (got < n) {
            ssize_t r = ::recv(fd, buf + got, n - got, 0);
            if (r <= 0) return false;
            got += static_cast<size_t>(r);
        }
        return true;
    }
    std::vector<std::pair<std::string, Json>> pending;
    Json wait_reply(const char* req, int timeout_ms) {
        auto match = [&](const Json& j) {
            return j.is_object() && j.contains("op") &&
                   j.at("op").get<std::string>() == "reply" &&
                   j.dump().find(req) != std::string::npos;
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
            uint8_t hdr[8];
            if (!read_exact(hdr, 8)) return Json();
            uint32_t len;
            std::memcpy(&len, hdr + 4, 4);
            std::vector<uint8_t> body(len);
            if (len && !read_exact(body.data(), len)) return Json();
            std::string magic(hdr, hdr + 4);
            waited += 5;
            if (magic == "VBC1") {
                Json j = json_parse(std::string(body.begin(), body.end()));
                if (match(j)) return j;
                pending.emplace_back(magic, j);
            } else {
                pending.emplace_back(magic, Json());
            }
        }
        return Json();
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_count_pose <fixtures-dir>\n");
        return 2;
    }
    const std::string dir = argv[1];

    // min_keypoints(): default 0 on the existing analyzers.
    {
        std::string err;
        auto z = create_analyzer("zone", err);
        CHECK(z != nullptr && z->min_keypoints() == 0);
        auto p = create_analyzer("pose_angle", err);
        CHECK(p != nullptr && p->min_keypoints() == 0);  // empty joints config
        CHECK(p->configure("{\"joints\":[{\"id\":\"k\",\"a\":11,\"b\":13,\"c\":15,"
                           "\"max_deg\":160}]}", err));
        CHECK(p->min_keypoints() == 16);
    }

    run_fixtures("count_threshold", kCountFixtures,
                 sizeof kCountFixtures / sizeof(*kCountFixtures), dir);
    run_fixtures("pose_angle", kPoseFixtures,
                 sizeof kPoseFixtures / sizeof(*kPoseFixtures), dir);

    // Synthetic backend has caps.keypoints = 0: adding pose_angle (with
    // joints) must reply ok:false with "requires keypoints" (§6.2.5).
    {
        int sv[2];
        CHECK(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
        Peer peer{};
        peer.fd = sv[0];
        struct timeval tv{10, 0};
        ::setsockopt(sv[0], SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        Writer writer;
        std::string ignore;
        RuntimeConfig cfg = RuntimeConfig::from_json(
            Json{{"backend", {{"name", "synthetic"}, {"max_batch", 2}}}}, ignore);
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

        std::thread server([&] {
            rt.emit_hello();
            serve_fd(sv[1], rt);
        });

        CHECK(peer.send_line(
            R"({"op":"add","req":"r-1","stream":{"index":0,"id":"c","url":"synthetic://?fps=5","transport":"tcp","score_threshold":0.35,"options":{}},"analyzers":[{"name":"pose_angle","config":{"joints":[{"id":"knee","a":11,"b":13,"c":15,"max_deg":160}]}}]})"));
        Json reply = peer.wait_reply("r-1", 5000);
        CHECK(!reply.is_null());
        CHECK(reply.at("ok").get<bool>() == false);
        CHECK(reply.at("error").get<std::string>().find("requires keypoints") !=
              std::string::npos);

        CHECK(peer.send_line(R"({"op":"stop","req":"r-9"})"));
        Json stop = peer.wait_reply("r-9", 2000);
        CHECK(!stop.is_null() && stop.at("ok").get<bool>() == true);
        server.join();
        rt.stop();
        writer.stop();
        ::close(sv[0]);
        ::close(sv[1]);
    }

    std::printf("test_count_pose: all OK\n");
    return 0;
}
