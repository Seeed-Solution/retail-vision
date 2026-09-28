// HailoRT batched inference session, real build (spec BASE-1 §M2.3).
//
// Copied from fall-detection platforms/rpi-hailo/src/batched_hailo_runner.cpp
// (origin/main@eb72e1e, Apache-2.0; see NOTICE). Namespaced to vb::; the
// HailoRT session (VDevice, configured HEF, DMA-mapped binding slots,
// zero-padded partial batches, 9-output pose validation) is unchanged. Two
// §6.1 alignments: infer() exposes the batch synchronously for
// InferenceContext::infer, and RawTensor carries a vb-owned HailoVStreamInfo
// mirror so the stub build shares this header.
//
// HailoRT is never shipped: the library comes from the host
// (/usr/lib/libhailort.so, found by find_package(HailoRT)) and is recorded as
// DT_NEEDED (spec §M2.3 mirror rules; spec §M2 "镜像与锁定" rule 3).
#include "batched_hailo_runner.h"

#include <hailo/hailort.hpp>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace vb {
namespace {
using namespace std::chrono_literals;

std::runtime_error failure(const char *operation, hailo_status status) {
  std::ostringstream out;
  out << operation << " failed with Hailo status " << status;
  return std::runtime_error(out.str());
}

void requireStatus(const char *operation, hailo_status status) {
  if (status != HAILO_SUCCESS) throw failure(operation, status);
}

template <typename T>
T takeExpected(const char *operation, hailort::Expected<T> &&expected) {
  if (!expected) throw failure(operation, expected.status());
  return expected.release();
}

class MappedBuffer {
 public:
  explicit MappedBuffer(size_t logical_size) : logical_size_(logical_size) {
    if (logical_size_ == 0) throw std::invalid_argument("zero-length Hailo buffer");
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    const size_t page_size = static_cast<size_t>(page);
    mapping_size_ = ((logical_size_ + page_size - 1) / page_size) * page_size;
    address_ = mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address_ == MAP_FAILED) {
      address_ = nullptr;
      throw std::bad_alloc();
    }
  }
  ~MappedBuffer() {
    if (address_) munmap(address_, mapping_size_);
  }
  MappedBuffer(const MappedBuffer &) = delete;
  MappedBuffer &operator=(const MappedBuffer &) = delete;
  MappedBuffer(MappedBuffer &&other) noexcept
      : address_(other.address_), logical_size_(other.logical_size_),
        mapping_size_(other.mapping_size_) {
    other.address_ = nullptr;
  }
  MappedBuffer &operator=(MappedBuffer &&) = delete;

  uint8_t *data() { return static_cast<uint8_t *>(address_); }
  hailort::MemoryView view() { return {address_, logical_size_}; }

 private:
  void *address_ = nullptr;
  size_t logical_size_ = 0;
  size_t mapping_size_ = 0;
};

struct OutputSpec {
  std::string name;
  size_t bytes = 0;
  hailo_vstream_info_t info{};
};

// vb::HailoVStreamInfo mirror of the fields this backend consumes.
HailoVStreamInfo mirror_info(const hailo_vstream_info_t &info) {
  HailoVStreamInfo out;
  out.features = static_cast<int32_t>(info.shape.features);
  out.height = static_cast<int32_t>(info.shape.height);
  out.width = static_cast<int32_t>(info.shape.width);
  out.format_type = static_cast<uint8_t>(info.format.type);
  out.qp_scale = info.quant_info.qp_scale;
  out.qp_zp = static_cast<int32_t>(info.quant_info.qp_zp);
  return out;
}

std::vector<OutputSpec> validateOutputs(const hailort::InferModel &model) {
  auto infos_expected = model.hef().get_output_vstream_infos();
  if (!infos_expected) throw failure("get_output_vstream_infos", infos_expected.status());
  const auto infos = infos_expected.release();
  if (infos.size() != 9 || model.outputs().size() != 9)
    throw std::runtime_error("pose HEF must expose exactly 9 outputs");

  std::map<std::string, hailo_vstream_info_t> by_name;
  for (const auto &info : infos) by_name.emplace(info.name, info);
  std::map<std::pair<int, int>, int> expected;
  for (int side : {20, 40, 80}) {
    expected[{side, 64}] = HAILO_FORMAT_TYPE_UINT8;
    expected[{side, 1}] = HAILO_FORMAT_TYPE_UINT8;
    expected[{side, 51}] = HAILO_FORMAT_TYPE_UINT16;
  }

  std::vector<OutputSpec> outputs;
  outputs.reserve(9);
  for (const auto &stream : model.outputs()) {
    auto found = by_name.find(stream.name());
    if (found == by_name.end())
      throw std::runtime_error("missing vstream info for output " + stream.name());
    const auto &info = found->second;
    const auto key = std::make_pair(static_cast<int>(info.shape.height),
                                    static_cast<int>(info.shape.features));
    auto wanted = expected.find(key);
    if (info.shape.width != info.shape.height || wanted == expected.end() ||
        info.format.type != wanted->second)
      throw std::runtime_error("unexpected pose output shape or type: " + stream.name());
    const size_t element_bytes =
        info.format.type == HAILO_FORMAT_TYPE_UINT16 ? 2U : 1U;
    const size_t expected_bytes = static_cast<size_t>(info.shape.height) *
                                  info.shape.width * info.shape.features *
                                  element_bytes;
    if (stream.get_frame_size() != expected_bytes)
      throw std::runtime_error("unexpected pose output byte size: " + stream.name());
    expected.erase(wanted);
    outputs.push_back({stream.name(), stream.get_frame_size(), info});
  }
  if (!expected.empty()) throw std::runtime_error("pose HEF output set is incomplete");
  return outputs;
}
}  // namespace

class BatchedHailoRunner::Impl {
 public:
  Impl(const std::string &hef_path, int batch_size, FrameBatcher &batcher,
       HailoResultHandler on_result, HailoErrorHandler on_error)
      : batch_size_(batch_size), batcher_(batcher),
        on_result_(std::move(on_result)), on_error_(std::move(on_error)) {
    if (batch_size_ != 1 && batch_size_ != 4 && batch_size_ != 8)
      throw std::invalid_argument("batch size must be 1, 4, or 8");

    const auto context = BatchedHailoRunner::inspectHef(hef_path);
    if (context.network_groups != 1)
      throw std::runtime_error("shared runner requires exactly one HEF network group");

    hailo_vdevice_params_t params{};
    requireStatus("hailo_init_vdevice_params", hailo_init_vdevice_params(&params));
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_NONE;
    vdevice_ = takeExpected("VDevice::create", hailort::VDevice::create(params));
    infer_model_ = takeExpected("create_infer_model",
                                vdevice_->create_infer_model(hef_path));
    if (infer_model_->inputs().size() != 1)
      throw std::runtime_error("shared runner requires exactly one input");
    const auto &input = infer_model_->inputs().front();
    const auto shape = input.shape();
    if (shape.height != 640 || shape.width != 640 || shape.features != 3 ||
        input.format().type != HAILO_FORMAT_TYPE_UINT8 ||
        input.get_frame_size() != 640U * 640U * 3U)
      throw std::runtime_error("shared runner requires RGB UINT8 640x640 input");
    input_name_ = input.name();
    input_size_ = input.get_frame_size();
    outputs_ = validateOutputs(*infer_model_);
    infer_model_->set_batch_size(static_cast<uint16_t>(batch_size_));
    configured_ = takeExpected("InferModel::configure", infer_model_->configure());
    requireStatus("ConfiguredInferModel::activate", configured_.activate());
    allocateSlots();
  }

  ~Impl() { stop(); }

  void start() {
    if (worker_.joinable()) throw std::logic_error("runner already started");
    stopping_ = false;
    worker_ = std::thread([this] { run(); });
  }

  void stop() {
    stopping_ = true;
    batcher_.stop(true);
    // HailoRT 4.21 run_async(vector<Bindings>) and shutdown share internal
    // pipeline locking. Let the single worker finish (or hit its checked
    // timeout) before shutting the configured model down.
    if (worker_.joinable()) worker_.join();
    if (!shutdown_called_.exchange(true)) {
      const auto status = configured_.shutdown();
      if (status != HAILO_SUCCESS && on_error_)
        on_error_(failure("ConfiguredInferModel::shutdown", status).what());
    }
  }

  // §6.1 alignment: one synchronous batch. Slots beyond n are zero-filled,
  // exactly like fall's infer(). The returned RawTensors point at the slot
  // buffers and stay valid until the next infer() call.
  void infer(const std::vector<BatchFrame> &frames, size_t n,
             std::vector<std::vector<RawTensor>> &out) {
    if (n == 0 || n > static_cast<size_t>(batch_size_))
      throw std::runtime_error("invalid frame batch size");

    for (size_t i = 0; i < n; ++i) {
      const auto &frame = frames[i];
      if (frame.rgb.size() != input_size_)
        throw std::runtime_error("batch frame is not RGB 640x640");
      std::memcpy(input_buffers_[i].data(), frame.rgb.data(), input_size_);
    }
    for (size_t i = n; i < static_cast<size_t>(batch_size_); ++i)
      std::memset(input_buffers_[i].data(), 0, input_size_);

    requireStatus("wait_for_async_ready",
                  configured_.wait_for_async_ready(30s,
                      static_cast<uint32_t>(batch_size_)));
    completion_status_ = HAILO_UNINITIALIZED;
    auto job_expected = configured_.run_async(
        bindings_,
        [this](const hailort::AsyncInferCompletionInfo &info) {
          completion_status_ = info.status;
        });
    if (!job_expected) throw failure("run_async", job_expected.status());
    auto job = job_expected.release();
    requireStatus("AsyncInferJob::wait", job.wait(30s));
    requireStatus("async completion", completion_status_.load());

    out.clear();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back(raw_tensors_[i]);
  }

 private:
  void run() noexcept {
    try {
      std::vector<BatchFrame> frames;
      while (!stopping_ && batcher_.take(frames)) {
        std::vector<std::vector<RawTensor>> tensors;
        infer(frames, frames.size(), tensors);
        for (size_t i = 0; i < frames.size(); ++i)
          on_result_(std::move(frames[i]), tensors[i]);
      }
    } catch (const std::exception &error) {
      if (!stopping_ && on_error_) on_error_(error.what());
      stopping_ = true;
      batcher_.stop(true);
    }
  }

  void allocateSlots() {
    input_buffers_.reserve(batch_size_);
    output_buffers_.reserve(batch_size_);
    input_mappings_.reserve(batch_size_);
    output_mappings_.reserve(static_cast<size_t>(batch_size_) * outputs_.size());
    bindings_.reserve(batch_size_);
    raw_tensors_.resize(batch_size_);

    for (int slot = 0; slot < batch_size_; ++slot) {
      input_buffers_.emplace_back(input_size_);
      input_mappings_.push_back(takeExpected(
          "map input buffer",
          hailort::DmaMappedBuffer::create(
              *vdevice_, input_buffers_.back().data(), input_size_,
              HAILO_DMA_BUFFER_DIRECTION_H2D)));
      auto binding = takeExpected("create_bindings", configured_.create_bindings());
      auto input_binding =
          takeExpected("Bindings::input", binding.input(input_name_));
      requireStatus("input set_buffer",
                    input_binding.set_buffer(input_buffers_.back().view()));

      output_buffers_.emplace_back();
      auto &buffers = output_buffers_.back();
      auto &tensors = raw_tensors_[slot];
      buffers.reserve(outputs_.size());
      tensors.reserve(outputs_.size());
      for (const auto &output : outputs_) {
        buffers.emplace_back(output.bytes);
        output_mappings_.push_back(takeExpected(
            "map output buffer",
            hailort::DmaMappedBuffer::create(
                *vdevice_, buffers.back().data(), output.bytes,
                HAILO_DMA_BUFFER_DIRECTION_D2H)));
        auto output_binding =
            takeExpected("Bindings::output", binding.output(output.name));
        requireStatus("output set_buffer",
                      output_binding.set_buffer(buffers.back().view()));
        tensors.push_back({buffers.back().data(), mirror_info(output.info)});
      }
      bindings_.push_back(std::move(binding));
    }
  }

  int batch_size_;
  FrameBatcher &batcher_;
  HailoResultHandler on_result_;
  HailoErrorHandler on_error_;
  std::unique_ptr<hailort::VDevice> vdevice_;
  std::shared_ptr<hailort::InferModel> infer_model_;
  std::string input_name_;
  size_t input_size_ = 0;
  std::vector<OutputSpec> outputs_;
  // Declaration order makes buffers outlive mappings and the configured model.
  std::vector<MappedBuffer> input_buffers_;
  std::vector<std::vector<MappedBuffer>> output_buffers_;
  std::vector<hailort::DmaMappedBuffer> input_mappings_;
  std::vector<hailort::DmaMappedBuffer> output_mappings_;
  hailort::ConfiguredInferModel configured_;
  std::vector<hailort::ConfiguredInferModel::Bindings> bindings_;
  std::vector<std::vector<RawTensor>> raw_tensors_;
  std::atomic<hailo_status> completion_status_{HAILO_UNINITIALIZED};
  std::atomic<bool> stopping_{false};
  std::atomic<bool> shutdown_called_{false};
  std::thread worker_;
};

HefContextInfo BatchedHailoRunner::inspectHef(const std::string &hef_path) {
  auto hef = takeExpected("Hef::create", hailort::Hef::create(hef_path));
  auto groups = takeExpected("Hef::get_network_groups_infos",
                             hef.get_network_groups_infos());
  return {static_cast<int>(groups.size()),
          groups.size() == 1 && groups.front().is_multi_context};
}

BatchedHailoRunner::BatchedHailoRunner(
    const std::string &hef_path, int batch_size, FrameBatcher &batcher,
    HailoResultHandler on_result, HailoErrorHandler on_error)
    : impl_(std::make_unique<Impl>(hef_path, batch_size, batcher,
                                  std::move(on_result), std::move(on_error))),
      batch_size_(batch_size) {}
BatchedHailoRunner::~BatchedHailoRunner() = default;
void BatchedHailoRunner::start() { impl_->start(); }
void BatchedHailoRunner::stop() { impl_->stop(); }
int BatchedHailoRunner::batch_size() const { return batch_size_; }

int BatchedHailoRunner::infer(const std::vector<BatchFrame> &frames, size_t n,
                              std::vector<std::vector<RawTensor>> &out,
                              std::string &err) {
  try {
    impl_->infer(frames, n, out);
    err.clear();
    return 0;
  } catch (const std::exception &e) {
    err = e.what();
    return -1;
  }
}

}  // namespace vb
