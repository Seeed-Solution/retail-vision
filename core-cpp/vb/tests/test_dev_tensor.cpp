// Dev-mode raw tensor passthrough VBT1 (spec BASE-1 §6.12, M1.23a).
//
//   test_dev_tensor <fixtures_dir> <path-to-vb-runtime>
//
// Covers: VBT1 encoding byte-identical to wire_tensor_case1.bin; rate
// limiting (15 fps source, dev.max_fps=1, 5 s -> 5±1 records); second add
// rejected with "dev mode allows 1 stream"; VB_PRODUCTION=1 + --dev exits 2;
// dev.raw_tensors=true without --dev fails at startup; 16 MiB record cap.
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "check.h"
#include "dev_tensor.h"
#include "vb/runtime.h"
#include "wire_case_util.h"

using namespace vb;

namespace {

// Count VBT1 records in a captured record stream.
int count_vbt1(const std::vector<uint8_t>& bytes) {
    int n = 0;
    size_t off = 0;
    while (off + 8 <= bytes.size()) {
        uint32_t body_len = 0;
        for (int i = 0; i < 4; ++i)
            body_len |= static_cast<uint32_t>(bytes[off + 4 + i]) << (8 * i);
        if (std::memcmp(&bytes[off], "VBT1", 4) == 0) ++n;
        off += 8 + body_len;
    }
    CHECK(off == bytes.size());  // record stream must parse exactly
    return n;
}

// Run vb-runtime with args, capture stderr, return exit code.
int run_cli(const std::string& bin, const std::vector<std::string>& args,
            bool set_vb_production, std::string& err_out) {
    int pipefd[2];
    CHECK(::pipe(pipefd) == 0);
    pid_t pid = ::fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);
        if (set_vb_production) ::setenv("VB_PRODUCTION", "1", 1);
        else ::unsetenv("VB_PRODUCTION");
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(bin.c_str()));
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        ::execv(bin.c_str(), argv.data());
        _exit(127);
    }
    ::close(pipefd[1]);
    err_out.clear();
    char buf[4096];
    ssize_t n;
    while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0)
        err_out.append(buf, static_cast<size_t>(n));
    ::close(pipefd[0]);
    int status = 0;
    CHECK(::waitpid(pid, &status, 0) == pid);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    CHECK(argc >= 3);
    const std::string fixtures = argv[1];
    const std::string runtime_bin = argv[2];
    std::string err;

    // ---- 1. VBT1 encoding is byte-identical to the fixture ----
    {
        auto j = json_parse(read_file(fixtures + "/wire_tensor_case1.json"));
        CHECK_STREQ(j.at("magic").get<std::string>(), "VBT1");
        DevTensorFrame tf = dev_tensor_frame_from_case_json(j);
        std::vector<uint8_t> enc;
        CHECK(wire_encode_vbt1(tf, enc, err));
        std::string bin = read_file(fixtures + "/wire_tensor_case1.bin");
        CHECK(enc.size() == bin.size());
        CHECK(std::memcmp(enc.data(), bin.data(), enc.size()) == 0);
        std::printf("fixture encode: %zu bytes, byte-identical\n", enc.size());
    }

    // ---- 2. 16 MiB cap: oversized record is refused ----
    {
        DevTensorFrame tf;
        DevTensor t;
        t.name = "big";
        t.dims = {1};
        t.data.resize(kDevTensorMaxRecord);  // 16 MiB of payload
        tf.tensors.push_back(std::move(t));
        std::vector<uint8_t> enc;
        CHECK(!wire_encode_vbt1(tf, enc, err));
        CHECK_STREQ(err, "dev tensor record exceeds 16 MiB");
        CHECK(enc.empty());
    }

    // ---- 3. Rate limiter ----
    {
        DevRateLimiter lim(1.0);
        CHECK(lim.try_send(100.0));        // first frame passes
        CHECK(!lim.try_send(100.5));
        CHECK(!lim.try_send(100.99));
        CHECK(lim.try_send(101.0));
        DevRateLimiter lim2(2.0);
        CHECK(lim2.try_send(0.0));
        CHECK(!lim2.try_send(0.4));
        CHECK(lim2.try_send(0.5));
    }

    // ---- 4. dev config validation ----
    {
        DevConfig c;
        CHECK(dev_config_from_json(json_parse(R"({})"), false, c, err));
        CHECK(!c.raw_tensors && c.max_fps == 1.0 && c.max_streams == 1);

        CHECK(!dev_config_from_json(
            json_parse(R"({"dev":{"raw_tensors":true}})"), false, c, err));
        CHECK_STREQ(err, "dev.raw_tensors is not allowed in production configs");
        CHECK(dev_config_from_json(json_parse(R"({"dev":{"raw_tensors":true}})"),
                                   true, c, err));
        CHECK(c.raw_tensors);

        CHECK(!dev_config_from_json(json_parse(R"({"dev":{"max_fps":3}})"),
                                    false, c, err));
        CHECK_STREQ(err, "dev.max_fps must be in (0, 2]");
        CHECK(!dev_config_from_json(json_parse(R"({"dev":{"max_fps":0}})"),
                                    false, c, err));
        CHECK(!dev_config_from_json(json_parse(R"({"dev":{"max_streams":2}})"),
                                    false, c, err));
        CHECK_STREQ(err, "dev.max_streams must be 1");
        CHECK(dev_config_from_json(json_parse(R"({"dev":{"max_fps":2,"max_streams":1}})"),
                                   false, c, err));
    }

    // ---- 5. Runtime: rate-limited VBT1 + dev.max_streams enforcement ----
    {
        Json cfgj = json_parse(R"({
            "backend": {"name": "synthetic", "decoder": {"type": "raw"},
                        "model_w": 64, "model_h": 64, "boxes": 2},
            "dev": {"raw_tensors": true, "max_fps": 1.0, "max_streams": 1},
            "status_interval_s": 0.2
        })");
        err.clear();
        RuntimeConfig cfg = RuntimeConfig::from_json(cfgj, err, /*allow_dev=*/true);
        CHECK(err.empty());
        auto backend = create_backend("synthetic", cfg.backend_json, err);
        CHECK(backend != nullptr);

        Writer writer;
        Runtime rt(std::move(backend), cfg, writer);
        std::vector<uint8_t> bytes;
        std::mutex mu;
        std::vector<Json> replies;
        rt.on_reply_record = [&](const Json& j) {
            std::lock_guard<std::mutex> lk(mu);
            replies.push_back(j);
        };
        writer.start([&](const uint8_t* d, size_t n) {
            std::lock_guard<std::mutex> lk(mu);
            bytes.insert(bytes.end(), d, d + n);
        });
        CHECK(rt.start(err));

        auto add = [&](uint32_t index, const char* req) {
            Json a;
            a["op"] = "add";
            a["req"] = req;
            Json st;
            st["index"] = index;
            st["id"] = "s" + std::to_string(index);
            st["url"] = "synthetic://?w=64&h=48&fps=15&boxes=2";
            st["transport"] = "tcp";
            st["options"] = Json::object();
            a["stream"] = st;
            a["analyzers"] = Json::array();
            rt.handle_line(a);
        };
        add(0, "a1");
        add(1, "a2");  // must be rejected: dev.max_streams == 1

        std::this_thread::sleep_for(std::chrono::seconds(5));
        rt.stop();
        writer.stop();

        bool a1_ok = false, a2_rejected = false;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (const auto& r : replies) {
                if (r.value("op", std::string()) != "reply") continue;
                if (r.value("req", std::string()) == "a1" &&
                    r.value("ok", false))
                    a1_ok = true;
                if (r.value("req", std::string()) == "a2" &&
                    !r.value("ok", true) &&
                    r.value("error", std::string()) == "dev mode allows 1 stream")
                    a2_rejected = true;
            }
        }
        CHECK(a1_ok);
        CHECK(a2_rejected);

        std::lock_guard<std::mutex> lk(mu);
        int n = count_vbt1(bytes);
        std::printf("runtime: %d VBT1 records in 5 s (expect 5±1)\n", n);
        CHECK(n >= 4 && n <= 6);
    }

    // ---- 6. CLI: VB_PRODUCTION=1 refuses --dev with exit code 2 ----
    {
        std::string out;
        int rc = run_cli(runtime_bin, {"--dev", "--standalone", "--config", "x"},
                         true, out);
        CHECK(rc == 2);
        CHECK(out.find("dev mode disabled in production image") !=
              std::string::npos);
    }

    // ---- 7. CLI: dev.raw_tensors=true without --dev fails startup ----
    {
        char path[] = "/tmp/vb_dev_cfg_XXXXXX";
        int fd = ::mkstemp(path);
        CHECK(fd >= 0);
        const char* body =
            "{\"backend\":{\"name\":\"synthetic\",\"decoder\":{\"type\":\"raw\"}},"
            "\"dev\":{\"raw_tensors\":true}}";
        CHECK(::write(fd, body, std::strlen(body)) ==
              static_cast<ssize_t>(std::strlen(body)));
        ::close(fd);
        std::string out;
        int rc = run_cli(runtime_bin,
                         {"--listen", "/tmp/vb_dev_test.sock", "--config", path},
                         false, out);
        ::unlink(path);
        CHECK(rc == 1);
        CHECK(out.find("dev.raw_tensors is not allowed in production configs") !=
              std::string::npos);
    }

    std::printf("dev_tensor: OK\n");
    return 0;
}
