#include "check.h"
#include "snapshot.h"
#include "vb/backend.h"
#include "vb/decoder.h"
#include "vb/runtime.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace vb;
struct Gate {
    std::mutex mu;
    std::condition_variable cv;
    bool entered = false, release = false;
    std::atomic<int> runs{0};
    std::atomic<bool> destroyed{false};
};
class ImageContext final : public Stage2Context {
    std::shared_ptr<Gate> gate_;
    std::vector<float> logits_{0, 0, 10, 10};
public:
    explicit ImageContext(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {}
    ~ImageContext() override { gate_->destroyed = true; }
    int infer_crops(const FrameBuf&, const CropReq*, size_t, TensorView*, std::string&) override { return -1; }
    int infer_rgb(const uint8_t*, int, int, int, TensorView* out, std::string&) override {
        int run = ++gate_->runs;
        if (run > 1) {
            std::unique_lock<std::mutex> lk(gate_->mu);
            gate_->entered = true;
            gate_->cv.notify_all();
            if (!gate_->cv.wait_for(lk, std::chrono::seconds(3), [&] { return gate_->release; })) return -1;
        }
        out->data = logits_.data(); out->count = 4; out->dims = {2, 2}; return 0;
    }
};
class ImageBackend final : public Backend {
    std::unique_ptr<Backend> delegate_;
    std::shared_ptr<Gate> gate_;
public:
    explicit ImageBackend(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {
        std::string err; delegate_ = create_backend("synthetic", "{}", err); CHECK(delegate_);
    }
    const char* name() const override { return "test-image"; }
    Caps caps() const override { return delegate_->caps(); }
    std::pair<int,int> model_hw() const override { return delegate_->model_hw(); }
    std::string model_sha256() const override { return "test"; }
    std::unique_ptr<FrameSource> create_source(const StreamSpec& s, std::string& e) override { return delegate_->create_source(s,e); }
    std::unique_ptr<InferenceContext> create_context(int i, std::string& e) override { return delegate_->create_context(i,e); }
    std::unique_ptr<Stage2Context> create_stage2(const Stage2Spec&, std::string&) override { return std::make_unique<ImageContext>(gate_); }
};
std::string base64(const std::vector<uint8_t>& bytes) {
    static const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        unsigned n = bytes[i] << 16;
        if (i + 1 < bytes.size()) n |= bytes[i+1] << 8;
        if (i + 2 < bytes.size()) n |= bytes[i+2];
        result += chars[n >> 18]; result += chars[(n >> 12) & 63];
        result += i+1 < bytes.size() ? chars[(n >> 6) & 63] : '=';
        result += i+2 < bytes.size() ? chars[n & 63] : '=';
    }
    return result;
}
int main() {
#ifndef VB_HAVE_JPEG
    return 77;
#else
    SnapshotRingEntry pixels; pixels.w = 2; pixels.h = 2; pixels.pixels.assign(12, 100);
    std::vector<uint8_t> jpeg; int w,h; std::string err;
    CHECK(snapshot_encode_jpeg(pixels, 0, false, 640, jpeg, w, h, err));
    auto gate = std::make_shared<Gate>();
    WriterConfig wc; wc.event_backpressure_limit = 1; wc.stop_flush_ms = 50;
    Writer writer(wc);
    // A peer that never drains: stop must unblock both producers and sink.
    writer.start([&](const uint8_t*, size_t) {
        while (!writer.flush_expired()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    RuntimeConfig config;
    config.stage2_json = Json{{"model_path","test"},{"input_hw",{2,2}},
                              {"charset",{"","A"}}}.dump();
    Runtime rt(std::make_unique<ImageBackend>(gate), config, writer);
    CHECK(rt.start(err)); CHECK(gate->runs == 1);
    auto request = [&](int id) { return Json{{"op","infer_image"},{"req",std::to_string(id)}, {"jpeg_b64",base64(jpeg)}}; };
    rt.handle_line(request(0));
    { std::unique_lock<std::mutex> lk(gate->mu);
      CHECK(gate->cv.wait_for(lk,std::chrono::seconds(1),[&] { return gate->entered; })); }
    for (int i = 1; i <= 16; ++i) rt.handle_line(request(i));
    std::atomic<int> full{0};
    rt.on_reply_record = [&](const Json& j) {
        if (j.value("error",std::string()) == "infer_image queue full") ++full;
    };
    // Flood past the queue cap so reply producers park behind the stalled sink.
    std::thread producer([&] { for (int i = 17; i < 25; ++i) rt.handle_line(request(i)); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!full && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    CHECK(full > 0); CHECK(!gate->destroyed);
    auto began = std::chrono::steady_clock::now();
    std::thread stop([&] { rt.stop(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(!gate->destroyed);
    { std::lock_guard<std::mutex> lk(gate->mu); gate->release = true; }
    gate->cv.notify_all();
    producer.join(); stop.join(); writer.stop();
    CHECK(std::chrono::steady_clock::now() - began < std::chrono::seconds(1));
    CHECK(gate->runs == 2); CHECK(gate->destroyed);
    return 0;
#endif
}
