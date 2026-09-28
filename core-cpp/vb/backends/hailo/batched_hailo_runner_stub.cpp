// HailoRT batched inference session, VB_HAILO_STUB build (spec BASE-1 §M2.3).
//
// Same interface as batched_hailo_runner.cpp, no HailoRT: infer() returns the
// tensors installed through vb::hailo_stub::set_outputs() (or zero-filled
// buffers with the 9 standard pose-output infos) for every frame slot. This
// is what the offline ctest run decodes; it never touches a device.
#include "batched_hailo_runner.h"

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace vb {
namespace {

constexpr size_t kStubInputBytes = 640U * 640U * 3U;

HailoVStreamInfo pose_info(int32_t side, int32_t features, uint8_t fmt) {
    HailoVStreamInfo info;
    info.features = features;
    info.height = side;
    info.width = side;
    info.format_type = fmt;
    // UINT16 kpt outputs are dequantised with the HEF's own scale/zero point;
    // the default mirror (scale 1, zp 0) matches the float passthrough the
    // tests hand-compute against.
    return info;
}

std::vector<HailoVStreamInfo> default_pose_infos() {
    std::vector<HailoVStreamInfo> infos;
    for (int32_t side : {20, 40, 80}) {
        infos.push_back(pose_info(side, 64, kHailoFormatUint8));
        infos.push_back(pose_info(side, 1, kHailoFormatUint8));
        infos.push_back(pose_info(side, 51, kHailoFormatUint16));
    }
    return infos;
}

size_t info_bytes(const HailoVStreamInfo& info) {
    const size_t elem = info.format_type == kHailoFormatUint16 ? 2 : 1;
    return static_cast<size_t>(info.features) *
           static_cast<size_t>(info.height) * static_cast<size_t>(info.width) *
           elem;
}

struct StubState {
    std::mutex mu;
    std::vector<HailoVStreamInfo> infos = default_pose_infos();
    std::vector<std::vector<uint8_t>> bytes;  // parallel to infos; absent/short = zeros
};

StubState& stub_state() {
    static StubState state;
    return state;
}

class StubImpl {
public:
    explicit StubImpl(int batch_size) : batch_size_(batch_size) {
        if (batch_size_ != 1 && batch_size_ != 4 && batch_size_ != 8)
            throw std::invalid_argument("batch size must be 1, 4, or 8");
    }

    void infer(const std::vector<BatchFrame>& frames, size_t n,
               std::vector<std::vector<RawTensor>>& out) {
        if (n == 0 || n > static_cast<size_t>(batch_size_))
            throw std::runtime_error("invalid frame batch size");
        for (size_t i = 0; i < n; ++i) {
            if (frames[i].rgb.size() != kStubInputBytes)
                throw std::runtime_error("batch frame is not RGB 640x640");
        }
        std::lock_guard<std::mutex> lk(stub_state().mu);
        out.clear();
        out.reserve(n);
        for (size_t slot = 0; slot < n; ++slot) {
            std::vector<RawTensor> tensors;
            tensors.reserve(stub_state().infos.size());
            for (size_t t = 0; t < stub_state().infos.size(); ++t) {
                const auto& info = stub_state().infos[t];
                std::vector<uint8_t>& storage = slot_bytes(slot, t, info);
                if (t < stub_state().bytes.size() &&
                    stub_state().bytes[t].size() == info_bytes(info)) {
                    std::memcpy(storage.data(), stub_state().bytes[t].data(),
                                storage.size());
                } else {
                    std::fill(storage.begin(), storage.end(), 0);
                }
                tensors.push_back({storage.data(), info});
            }
            out.push_back(std::move(tensors));
        }
    }

private:
    // Per-slot copies so decoding slot k cannot race the next infer()
    // overwriting the same bytes. The real runner's RawTensors have the same
    // valid-until-next-infer lifetime rule; the stub mirrors it.
    std::vector<uint8_t>& slot_bytes(size_t slot, size_t t,
                                     const HailoVStreamInfo& info) {
        auto key = std::make_pair(slot, t);
        auto found = slot_bytes_.find(key);
        if (found == slot_bytes_.end() ||
            found->second.size() != info_bytes(info)) {
            found = slot_bytes_
                        .emplace(key, std::vector<uint8_t>(info_bytes(info), 0))
                        .first;
        }
        return found->second;
    }

    int batch_size_;
    std::map<std::pair<size_t, size_t>, std::vector<uint8_t>> slot_bytes_;
};

}  // namespace

namespace hailo_stub {

void set_outputs(std::vector<HailoVStreamInfo> infos,
                 std::vector<std::vector<uint8_t>> bytes) {
    std::lock_guard<std::mutex> lk(stub_state().mu);
    stub_state().infos = std::move(infos);
    stub_state().bytes = std::move(bytes);
}

void clear_outputs() {
    std::lock_guard<std::mutex> lk(stub_state().mu);
    stub_state().infos = default_pose_infos();
    stub_state().bytes.clear();
}

}  // namespace hailo_stub

class BatchedHailoRunner::Impl {
 public:
    Impl(const std::string& /*hef_path*/, int batch_size, FrameBatcher& batcher,
         HailoResultHandler on_result, HailoErrorHandler /*on_error*/)
        : impl_(batch_size), batcher_(batcher), on_result_(std::move(on_result)) {}

    void start() {
        if (worker_.joinable()) throw std::logic_error("runner already started");
        stopping_ = false;
        worker_ = std::thread([this] { run(); });
    }

    void stop() {
        stopping_ = true;
        batcher_.stop(true);
        if (worker_.joinable()) worker_.join();
    }

    void infer(const std::vector<BatchFrame>& frames, size_t n,
               std::vector<std::vector<RawTensor>>& out) {
        impl_.infer(frames, n, out);
    }

private:
    void run() noexcept {
        std::vector<BatchFrame> frames;
        while (!stopping_ && batcher_.take(frames)) {
            std::vector<std::vector<RawTensor>> tensors;
            impl_.infer(frames, frames.size(), tensors);
            for (size_t i = 0; i < frames.size(); ++i)
                on_result_(std::move(frames[i]), tensors[i]);
        }
    }

    StubImpl impl_;
    FrameBatcher& batcher_;
    HailoResultHandler on_result_;
    std::thread worker_;
    std::atomic<bool> stopping_{false};
};

HefContextInfo BatchedHailoRunner::inspectHef(const std::string& /*hef_path*/) {
    return {1, true};  // single multi-context network group, as the pose HEF
}

BatchedHailoRunner::BatchedHailoRunner(
    const std::string& hef_path, int batch_size, FrameBatcher& batcher,
    HailoResultHandler on_result, HailoErrorHandler on_error)
    : impl_(std::make_unique<Impl>(hef_path, batch_size, batcher,
                                  std::move(on_result), std::move(on_error))),
      batch_size_(batch_size) {}
BatchedHailoRunner::~BatchedHailoRunner() = default;
void BatchedHailoRunner::start() { impl_->start(); }
void BatchedHailoRunner::stop() { impl_->stop(); }
int BatchedHailoRunner::batch_size() const { return batch_size_; }

int BatchedHailoRunner::infer(const std::vector<BatchFrame>& frames, size_t n,
                              std::vector<std::vector<RawTensor>>& out,
                              std::string& err) {
    try {
        impl_->infer(frames, n, out);
        err.clear();
        return 0;
    } catch (const std::exception& e) {
        err = e.what();
        return -1;
    }
}

}  // namespace vb
