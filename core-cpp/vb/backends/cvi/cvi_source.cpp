#include "cvi_source.h"
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>
#include "vb/json.h"
#if defined(VB_CVI_SSCMA_VIDEO)
extern "C" {
#include "video.h"
}
#endif
namespace vb { namespace {
constexpr size_t kMaxFrameBytes = 64u * 1024u * 1024u;
struct Packet {
  std::vector<uint8_t> bytes;
  int w = 0;
  int h = 0;
  int stride = 0;
  uint64_t seq = 0;
  double wall_ms = 0;
  double mono_s = 0;
};
#if defined(VB_CVI_SSCMA_VIDEO)
class Source;
std::mutex g_owner_mu;
Source* g_owner = nullptr;
#endif
class Source final : public FrameSource {
 public:
  explicit Source(const StreamSpec& s, std::string& err) : stream_(s) {
    try {
      Json j = json_parse(s.options_json.empty() ? "{}" : s.options_json);
      if (!j.contains("width") || !j.contains("height") ||
          !j.contains("fps") || !j.contains("timeout_ms")) {
        err = "cvi source requires width, height, fps, timeout_ms";
        return;
      }
      w_ = j.at("width").get<int>();
      h_ = j.at("height").get<int>();
      fps_ = j.at("fps").get<int>();
      timeout_ms_ = j.at("timeout_ms").get<int>();
      if (j.value("pixel_format", std::string("rgb888")) != "rgb888") {
        err = "sscma cvi source requires pixel_format=rgb888";
        return;
      }
      if (w_ <= 0 || h_ <= 0 || fps_ <= 0 || fps_ > 255 ||
          timeout_ms_ < 0 || timeout_ms_ > 60000 ||
          static_cast<size_t>(w_) > kMaxFrameBytes / 3u ||
          static_cast<size_t>(h_) > kMaxFrameBytes /
              (static_cast<size_t>(w_) * 3u)) {
        err = "cvi source camera bounds exceed safe limits";
        return;
      }
      configured_ = true;
    } catch (const std::exception& ex) {
      err = std::string("cvi source json: ") + ex.what();
    }
  }

  ~Source() override { close(); }

  bool open(std::string& err) override {
#if defined(VB_CVI_SSCMA_VIDEO)
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mu_);
    if (!configured_) {
      err = "cvi source not configured";
      return false;
    }
    {
      std::lock_guard<std::mutex> owner_lock(g_owner_mu);
      if (g_owner != nullptr && g_owner != this) {
        err = "cvi source camera already owned";
        return false;
      }
      g_owner = this;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (state_ == State::Streaming || state_ == State::Opening) return true;
      state_ = State::Opening;
      first_ = false;
      queue_.clear();
      seq_ = 0;
    }
    if (initVideo() != 0) return rollback_open("initVideo failed", err);
    initialized_ = true;
    video_ch_param_t param{};
    param.format = VIDEO_FORMAT_RGB888;
    param.width = static_cast<uint32_t>(w_);
    param.height = static_cast<uint32_t>(h_);
    param.fps = static_cast<uint8_t>(fps_);
    if (setupVideo(VIDEO_CH0, &param) != 0 ||
        registerVideoFrameHandler(VIDEO_CH0, 0, &Source::callback, this) != 0 ||
        startVideo() != 0) {
      return rollback_open("SSCMA video setup/start failed", err);
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      started_ = true;
    }
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms_), [this] {
            return state_ == State::Streaming || state_ == State::Closing;
          })) {
        lock.unlock();
        return rollback_open("SSCMA video started but no RGB frame before timeout", err);
      }
      if (state_ != State::Streaming) {
        lock.unlock();
        return rollback_open("SSCMA video closed before first frame", err);
      }
    }
    return true;
#else
    err = "cvi source requires SSCMA_VIDEO_ROOT and VB_CVI_SSCMA_VIDEO";
    return false;
#endif
  }

  int read(FrameBuf& out, int timeout_ms) override {
    std::unique_lock<std::mutex> lock(mu_);
    if (state_ != State::Streaming && state_ != State::Opening) return -1;
    const int wait_ms = timeout_ms >= 0 ? timeout_ms : timeout_ms_;
    if (!cv_.wait_for(lock, std::chrono::milliseconds(wait_ms), [this] {
          return !queue_.empty() || state_ == State::Closing ||
                 state_ == State::Closed;
        })) return 0;
    if (queue_.empty()) return state_ == State::Closing ? -1 : 0;
    std::shared_ptr<Packet> packet = queue_.back();
    queue_.clear();
    lock.unlock();
    out = FrameBuf{};
    out.stream_index = stream_.index;
    out.seq = packet->seq;
    out.wall_ms = packet->wall_ms;
    out.t_mono_s = packet->mono_s;
    out.w = packet->w;
    out.h = packet->h;
    out.stride = packet->stride;
    out.hstride = packet->h;
    out.fmt = PixFmt::RGB888;
    out.mem = Mem::Host;
    out.host = packet->bytes.data();
    out.full_host = out.host;
    out.full_stride = out.stride;
    out.full_w = out.w;
    out.full_h = out.h;
    out.hold = std::move(packet);
    return 1;
  }

  void close() override {
#if defined(VB_CVI_SSCMA_VIDEO)
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
    close_impl();
#endif
  }

 private:
#if defined(VB_CVI_SSCMA_VIDEO)
  void close_impl() {
    bool stop = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (state_ == State::Closed && !started_ && !initialized_) return;
      state_ = State::Closing;
      stop = started_;
      started_ = false;
    }
    cv_.notify_all();
    if (stop || initialized_) deinitVideo();
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.clear();
      first_ = false;
      state_ = State::Closed;
    }
    {
      std::lock_guard<std::mutex> owner_lock(g_owner_mu);
      if (g_owner == this) g_owner = nullptr;
    }
    initialized_ = false;
  }
#endif

  const char* decode_path() const override { return "sscma_video_rgb888"; }

  enum class State { Closed, Opening, Streaming, Closing };

#if defined(VB_CVI_SSCMA_VIDEO)
  static int callback(void* data, void*, void* user) {
    return static_cast<Source*>(user)->on_frame(
        static_cast<const VIDEO_FRAME_INFO_S*>(data));
  }

  int on_frame(const VIDEO_FRAME_INFO_S* info) {
    if (info == nullptr) return -1;
    const VIDEO_FRAME_S& frame = info->stVFrame;
    if (frame.enPixelFormat != PIXEL_FORMAT_RGB_888 ||
        frame.u32Width != static_cast<CVI_U32>(w_) ||
        frame.u32Height != static_cast<CVI_U32>(h_) ||
        frame.u64PhyAddr[0] == 0 ||
        frame.u32Stride[0] < static_cast<CVI_U32>(w_ * 3)) return -1;
    const size_t stride = frame.u32Stride[0];
    if (stride > kMaxFrameBytes / static_cast<size_t>(h_)) return -1;
    const size_t bytes = stride * static_cast<size_t>(h_);
    auto packet = std::make_shared<Packet>();
    packet->bytes.resize(bytes);
    packet->w = w_;
    packet->h = h_;
    packet->stride = static_cast<int>(stride);
    packet->wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    packet->mono_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    void* mapped = CVI_SYS_Mmap(frame.u64PhyAddr[0], bytes);
    if (mapped == nullptr) return -1;
    std::memcpy(packet->bytes.data(), mapped, bytes);
    CVI_SYS_Munmap(mapped, bytes);
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (state_ == State::Closing || state_ == State::Closed) return -1;
      packet->seq = ++seq_;
      queue_.clear();
      queue_.push_back(packet);
      first_ = true;
      state_ = State::Streaming;
    }
    cv_.notify_all();
    return 0;
  }

  bool rollback_open(const char* reason, std::string& err) {
    err = reason;
    close_impl();
    return false;
  }
#endif

  StreamSpec stream_;
  int w_ = 0;
  int h_ = 0;
  int fps_ = 0;
  int timeout_ms_ = 1000;
  bool configured_ = false;
  bool started_ = false;
  bool initialized_ = false;
  bool first_ = false;
  uint64_t seq_ = 0;
  State state_ = State::Closed;
  std::mutex lifecycle_mu_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Packet>> queue_;
};
}  // namespace
std::unique_ptr<FrameSource> make_cvi_source(const StreamSpec& s, std::string& e) {
  auto p = std::make_unique<Source>(s, e);
  return e.empty() ? std::move(p) : nullptr;
}
}  // namespace vb
