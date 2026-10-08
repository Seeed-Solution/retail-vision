#include "cvi_stage2.h"
#include "cvi_resize.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>

#include "cvi_sha256.h"
#include "cviruntime.h"
#include "vb/decoder.h"
#include "vb/rate_crop.h"

namespace vb {
namespace {

struct M {
  std::vector<uint8_t> bytes;
  CVI_MODEL_HANDLE h = nullptr;
  CVI_TENSOR* i = nullptr;
  CVI_TENSOR* o = nullptr;
  int ni = 0;
  int no = 0;
  ~M() {
    if (h) CVI_NN_CleanupModel(h);
  }
};

bool shape(const CVI_TENSOR& t, std::initializer_list<int> want) {
  if (t.shape.dim_size != want.size()) return false;
  size_t i = 0;
  for (int d : want) {
    if (t.shape.dim[i++] != d) return false;
  }
  return true;
}

bool finite_spec(const Stage2Spec& s) {
  return std::isfinite(s.scale) && s.scale == 1.0f / 255.0f &&
         std::isfinite(s.mean[0]) && std::isfinite(s.mean[1]) &&
         std::isfinite(s.mean[2]) && s.mean[0] == 0 && s.mean[1] == 0 &&
         s.mean[2] == 0;
}

class S final : public Stage2Context {
 public:
  S(const Stage2Spec& sp, std::mutex& device_mu, std::string& e)
      : sp_(sp), mu_(device_mu) {
    if (sp_.in_h != 32 || sp_.in_w != 96 || sp_.bgr || !finite_spec(sp_)) {
      e = "cvi stage2 requires RGB 32x96, mean=0, scale=1/255";
      return;
    }
    m_ = std::make_shared<M>();
    std::ifstream file(sp_.model_path, std::ios::binary | std::ios::ate);
    if (!file) {
      e = "cvi stage2 model open failed";
      return;
    }
    const auto size = file.tellg();
    if (size <= 0 || static_cast<uint64_t>(size) > UINT32_MAX) {
      e = "cvi stage2 model size invalid";
      return;
    }
    m_->bytes.resize(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(m_->bytes.data()), size)) {
      e = "cvi stage2 model read failed";
      return;
    }
    if (cvi_detail::sha256_hex(m_->bytes).empty()) {
      e = "cvi stage2 model hash failed";
      return;
    }
    if (CVI_NN_RegisterModelFromBuffer(
            reinterpret_cast<const int8_t*>(m_->bytes.data()),
            static_cast<uint32_t>(m_->bytes.size()), &m_->h) != 0 ||
        !m_->h) {
      e = "CVI_NN_RegisterModelFromBuffer(stage2) failed";
      return;
    }
    if (CVI_NN_GetInputOutputTensors(m_->h, &m_->i, &m_->ni, &m_->o,
                                     &m_->no) != 0 ||
        m_->ni != 1 || m_->no < 1 || !m_->i || !m_->o) {
      e = "cvi stage2 tensor enumeration failed";
      return;
    }
    if (m_->i->fmt != CVI_FMT_FP32 || !shape(*m_->i, {1, 3, 32, 96}) ||
        m_->i->count != 3u * 32u * 96u ||
        m_->i->mem_size < m_->i->count * sizeof(float) ||
        m_->o->fmt != CVI_FMT_FP32 || !shape(*m_->o, {1, 66, 1, 24}) ||
        m_->o->count != 66u * 24u ||
        m_->o->mem_size < m_->o->count * sizeof(float)) {
      e = "cvi stage2 model contract mismatch";
      return;
    }
    if (!m_->o->name || std::string(m_->o->name) != "output_Add_f32") {
      e = "cvi stage2 output name must be output_Add_f32";
      return;
    }
    ready_ = true;
  }

  int infer_crops(const FrameBuf& f, const CropReq* c, size_t n,
                  TensorView* outs, std::string& e) override {
    size_t frame_bytes = 0;
    if (!ready_) {
      e = "cvi stage2 not ready";
      return -1;
    }
    if (!c || !outs || n == 0 || n > 16 || f.mem != Mem::Host || !f.host ||
        (f.fmt != PixFmt::RGB888 && f.fmt != PixFmt::BGR888) ||
        !host_rgb_layout_bytes(f.w, f.h, f.stride, frame_bytes)) {
      e = "cvi stage2 invalid frame/crops";
      return -1;
    }
    const int fw = f.full_w > 0 ? f.full_w : f.w;
    const int fh = f.full_h > 0 ? f.full_h : f.h;
    const int full_stride = f.full_stride > 0 ? f.full_stride : f.stride;
    const uint8_t* src = f.full_host ? f.full_host : f.host;
    if (!src || fw <= 0 || fh <= 0 ||
        (!f.full_host && (fw != f.w || fh != f.h || f.crop_x0 != 0 ||
                          f.crop_y0 != 0)) ||
        !host_rgb_layout_bytes(fw, fh, full_stride, frame_bytes)) {
      e = "cvi stage2 full-frame layout invalid";
      return -1;
    }

    std::lock_guard<std::mutex> lock(mu_);
    stored_.assign(n, std::vector<float>(66u * 24u));
    for (size_t k = 0; k < n; ++k) {
      if (!valid(c[k])) {
        e = "cvi stage2 crop bounds invalid";
        return -1;
      }
      const float ax = c[k].x0 * fw;
      const float bx = c[k].x1 * fw;
      const float ay = c[k].y0 * fh;
      const float by = c[k].y1 * fh;
      std::vector<uint8_t> resized;
      if (!cvi_resize_u8_inter_linear(src, fw, fh, full_stride, ax, ay, bx,
                                      by, 96, 32, resized)) {
        e = "cvi stage2 resize failed";
        return -1;
      }
      constexpr size_t plane = 32u * 96u;
      buf_.assign(3u * plane, 0);
      for (size_t i = 0; i < plane; ++i) {
        const int r = f.fmt == PixFmt::RGB888 ? 0 : 2;
        const int b = f.fmt == PixFmt::RGB888 ? 2 : 0;
        buf_[i] = resized[3 * i + r] * sp_.scale - sp_.mean[0];
        buf_[plane + i] = resized[3 * i + 1] * sp_.scale - sp_.mean[1];
        buf_[2 * plane + i] = resized[3 * i + b] * sp_.scale - sp_.mean[2];
      }
      if (CVI_NN_SetTensorPtr(m_->i, buf_.data()) != 0 ||
          CVI_NN_Forward(m_->h, m_->i, m_->ni, m_->o, m_->no) != 0) {
        e = "CVI stage2 input bind/forward failed";
        return -1;
      }
      float* p = static_cast<float*>(CVI_NN_TensorPtr(m_->o));
      if (!p || !std::all_of(p, p + 66u * 24u,
                             [](float v) { return std::isfinite(v); })) {
        e = "CVI stage2 output pointer/nonfinite output";
        return -1;
      }
      std::copy(p, p + 66u * 24u, stored_[k].begin());
      outs[k].data = stored_[k].data();
      outs[k].raw_data = stored_[k].data();
      outs[k].count = 66u * 24u;
      outs[k].dims = {66, 24};
      outs[k].dtype = 0;
      outs[k].name = m_->o->name;
    }
    return 0;
  }

  int infer_rgb(const uint8_t* p, int w, int h, int stride, TensorView* o,
                std::string& e) override {
    FrameBuf f;
    f.host = p;
    f.w = w;
    f.h = h;
    f.stride = stride;
    f.mem = Mem::Host;
    f.fmt = PixFmt::RGB888;
    const CropReq c{0, 0, 1, 1};
    return infer_crops(f, &c, 1, o, e);
  }

 private:
  static bool valid(const CropReq& c) {
    return std::isfinite(c.x0) && std::isfinite(c.y0) &&
           std::isfinite(c.x1) && std::isfinite(c.y1) && c.x0 >= 0 &&
           c.y0 >= 0 && c.x1 <= 1 && c.y1 <= 1 && c.x0 < c.x1 &&
           c.y0 < c.y1;
  }

  Stage2Spec sp_;
  std::shared_ptr<M> m_;
  std::mutex& mu_;
  std::vector<float> buf_;
  std::vector<std::vector<float>> stored_;
  bool ready_ = false;
};

}  // namespace

std::unique_ptr<Stage2Context> make_cvi_stage2(const Stage2Spec& s,
                                               std::string& e,
                                               std::mutex* mu) {
  static std::mutex fallback;
  auto p = std::make_unique<S>(s, mu ? *mu : fallback, e);
  return e.empty() ? std::move(p) : nullptr;
}

}  // namespace vb
