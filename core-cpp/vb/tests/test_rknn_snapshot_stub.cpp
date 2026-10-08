// Host-only contract test for the RKNN source snapshot path.  It links the
// production Hybrid implementation against small RKNN/RGA fakes, so no NPU or
// RGA device is required.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <cstdarg>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "hybrid_rga_rknn.h"
#include "rga/im2d.h"
#include "rknn_api.h"
#include "rknn_stage2.h"
#include "vb/backend.h"
#include "vb/decoder.h"

namespace {
struct FakeMem { rknn_tensor_mem mem{}; std::vector<uint8_t> bytes; };
std::map<int, FakeMem*> by_fd;
int next_fd = 100;
int create_calls = 0, destroy_calls = 0;
std::vector<uint32_t> create_sizes;
std::vector<int> destroyed_fds;
std::vector<int> alloc_fds;
std::vector<int> close_fds;
bool alloc_fail = false, rga_fail = false, sync_fail = false;
int sync_from_fail_on_call = 0, sync_from_calls = 0;
int rga_fail_on_call = 0;
rga_buffer_t last_src{}, last_dst{};
std::vector<rga_buffer_t> rga_calls;
std::vector<int> stage2_events;
struct RgaRectCall { im_rect src{}, dst{}; };
std::vector<RgaRectCall> rga_rect_calls;

void reset_faults() {
    alloc_fail = rga_fail = sync_fail = false;
    sync_from_fail_on_call = sync_from_calls = 0;
}
}  // namespace

extern "C" int __real_open(const char*, int, ...);
extern "C" int __real_ioctl(int, unsigned long, ...);
extern "C" int __real_close(int);
extern "C" int __wrap_open(const char* path, int flags, ...) {
    if (std::strcmp(path, "/dev/dma_heap/system") == 0) return 700;
    return __real_open(path, flags);
}
extern "C" int __wrap_close(int fd) {
    close_fds.push_back(fd);
    return fd == 700 ? 0 : __real_close(fd);
}
extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
    va_list ap; va_start(ap, request); void* arg = va_arg(ap, void*); va_end(ap);
    if (fd == 700 && request == DMA_HEAP_IOCTL_ALLOC) {
        auto* a = static_cast<dma_heap_allocation_data*>(arg);
        auto* f = new FakeMem; f->bytes.assign(a->len, 0xaa);
        f->mem.virt_addr = f->bytes.data(); f->mem.fd = next_fd++; f->mem.size = a->len;
        by_fd[f->mem.fd] = f; alloc_fds.push_back(f->mem.fd); a->fd = f->mem.fd; return 0;
    }
    return __real_ioctl(fd, request, arg);
}

extern "C" {
int rknn_init(rknn_context* c, void*, uint32_t, uint32_t, rknn_init_extend*) {
    *c = 1; return RKNN_SUCC;
}
int rknn_destroy(rknn_context) { return RKNN_SUCC; }
int rknn_dup_context(rknn_context*, rknn_context* out) { *out = 2; return RKNN_SUCC; }
int rknn_set_core_mask(rknn_context, rknn_core_mask) { return RKNN_SUCC; }
int rknn_query(rknn_context, rknn_query_cmd cmd, void* out, uint32_t) {
    if (cmd == RKNN_QUERY_SDK_VERSION) {
        auto* v = static_cast<rknn_sdk_version*>(out);
        std::strcpy(v->api_version, "fake"); std::strcpy(v->drv_version, "fake");
    } else if (cmd == RKNN_QUERY_IN_OUT_NUM) {
        auto* n = static_cast<rknn_input_output_num*>(out); n->n_input = 1; n->n_output = 1;
    } else if (cmd == RKNN_QUERY_OUTPUT_ATTR || cmd == RKNN_QUERY_NATIVE_OUTPUT_ATTR) {
        auto* a = static_cast<rknn_tensor_attr*>(out); a->index = 0; a->n_dims = 2;
        a->dims[0] = 1; a->dims[1] = 1; a->n_elems = 1; a->size = 4;
        a->size_with_stride = 4; a->type = RKNN_TENSOR_FLOAT32;
        a->fmt = RKNN_TENSOR_NHWC; a->w_stride = 1; a->h_stride = 1;
    } else {
        auto* a = static_cast<rknn_tensor_attr*>(out);
        a->index = 0; a->n_dims = 4; a->dims[0] = 1; a->dims[1] = 32;
        a->dims[2] = 32; a->dims[3] = 3; a->fmt = RKNN_TENSOR_NHWC;
        a->n_elems = 3072; a->size = 3072; a->size_with_stride = 3072;
        a->w_stride = 32; a->h_stride = 32;
    }
    return RKNN_SUCC;
}
int rknn_set_io_mem(rknn_context, rknn_tensor_mem*, rknn_tensor_attr*) { return RKNN_SUCC; }
rknn_tensor_mem* rknn_create_mem(rknn_context, uint32_t size) {
    ++create_calls;
    create_sizes.push_back(size);
    if (alloc_fail) return nullptr;
    auto* f = new FakeMem;
    f->bytes.assign(size, 0xaa);
    f->mem.virt_addr = f->bytes.data(); f->mem.fd = next_fd++; f->mem.size = size;
    by_fd[f->mem.fd] = f;
    return &f->mem;
}
int rknn_destroy_mem(rknn_context, rknn_tensor_mem* m) {
    if (!m) return RKNN_SUCC;
    ++destroy_calls; destroyed_fds.push_back(m->fd);
    auto it = by_fd.find(m->fd);
    if (it != by_fd.end()) { delete it->second; by_fd.erase(it); }
    return RKNN_SUCC;
}
int rknn_mem_sync(rknn_context, rknn_tensor_mem*, rknn_mem_sync_mode mode) {
    if (mode == RKNN_MEMORY_SYNC_FROM_DEVICE) {
        stage2_events.push_back(2);
        ++sync_from_calls;
        if (sync_fail || (sync_from_fail_on_call > 0 && sync_from_calls == sync_from_fail_on_call)) return -77;
    }
    return RKNN_SUCC;
}
int rknn_run(rknn_context, rknn_run_extend*) { stage2_events.push_back(3); return RKNN_SUCC; }
int rknn_outputs_get(rknn_context, uint32_t, rknn_output[], rknn_output_extend*) { return RKNN_SUCC; }
int rknn_outputs_release(rknn_context, uint32_t, rknn_output[]) { return RKNN_SUCC; }
}

rga_buffer_t wrapbuffer_fd_t(int fd, int width, int height, int wstride,
                             int hstride, int format) {
    rga_buffer_t b{}; b.fd = fd; b.width = width; b.height = height;
    b.wstride = wstride; b.hstride = hstride; b.format = format; return b;
}
const char* imStrError_t(IM_STATUS) { return "fake RGA"; }
IM_STATUS improcess(rga_buffer_t src, rga_buffer_t dst, rga_buffer_t,
                    im_rect src_rect, im_rect dst_rect, im_rect, int, int*, im_opt_t*, int) {
    stage2_events.push_back(1);
    last_src = src; last_dst = dst; rga_calls.push_back(dst);
    rga_rect_calls.push_back({src_rect, dst_rect});
    const int call_no = static_cast<int>(rga_calls.size());
    if (rga_fail || (rga_fail_on_call > 0 && call_no == rga_fail_on_call))
        return IM_STATUS_INVALID_PARAM;
    auto it = by_fd.find(dst.fd); if (it == by_fd.end()) return IM_STATUS_INVALID_PARAM;
    auto& b = it->second->bytes;
    for (int y = 0; y < dst.height; ++y)
        for (int x = 0; x < dst.width; ++x) {
            const size_t p = static_cast<size_t>(y) * dst.wstride * 3u + static_cast<size_t>(x) * 3u;
            b[p] = 10; b[p + 1] = 20; b[p + 2] = 30;
        }
    return IM_STATUS_SUCCESS;
}

namespace vb::rknn_detail {
std::string sha256_hex(const void*, size_t) { return std::string(64, '0'); }
}

int main() {
    const char* model = "/tmp/rknn_snapshot_fake_model.bin";
    { std::FILE* f = std::fopen(model, "wb"); std::fputs("x", f); std::fclose(f); }
    std::string err;
    auto h = vb::RknnHybrid::create(model, 1, err);
    assert(h && err.empty());
    vb::FrameBuf f; f.mem = vb::Mem::DmaBuf; f.fmt = vb::PixFmt::NV12;
    f.dmabuf_fd = 42; f.w = f.full_w = 1920; f.h = f.full_h = 1080;
    f.stride = 1984; f.hstride = 1088;
    std::vector<uint8_t> rgb; int w = 0, height = 0;
    assert(h->copy_snapshot_rgb(f, rgb, w, height, err));
    assert(w == 1920 && height == 1080 && rgb.size() == size_t(w) * height * 3);
    assert(last_src.width == 1920 && last_src.height == 1080 && last_src.wstride == 1984 && last_src.hstride == 1088);
    assert(last_dst.width == 1920 && last_dst.height == 1080 && last_dst.wstride == 1920);
    for (size_t i = 0; i < rgb.size(); i += 3) assert(rgb[i] == 10 && rgb[i + 1] == 20 && rgb[i + 2] == 30);
    const int alloc_after_success = create_calls;

    // Same geometry reuses scratch; a larger geometry grows it once and the
    // grown allocation is reused when returning to the original size.
    const int snapshot_creates_before_reuse = create_calls;
    reset_faults();
    assert(h->copy_snapshot_rgb(f, rgb, w, height, err));
    assert(create_calls == snapshot_creates_before_reuse);
    f.w = f.full_w = 1936; f.stride = 1984;
    assert(h->copy_snapshot_rgb(f, rgb, w, height, err));
    assert(create_calls == snapshot_creates_before_reuse + 1);
    const int grown_snapshot_fd = last_dst.fd;
    // Returning to a smaller width changes the required pixel pitch, so the
    // implementation allocates a matching scratch descriptor; that new
    // descriptor is then reused on the next same-size snapshot.
    f.w = f.full_w = 1920; f.stride = 1984;
    assert(h->copy_snapshot_rgb(f, rgb, w, height, err));
    assert(create_calls == snapshot_creates_before_reuse + 2);
    assert(last_dst.fd != grown_snapshot_fd);
    const int returned_snapshot_fd = last_dst.fd;
    assert(h->copy_snapshot_rgb(f, rgb, w, height, err));
    assert(create_calls == snapshot_creates_before_reuse + 2);
    assert(last_dst.fd == returned_snapshot_fd);

    f.full_w = 1918; rgb.assign(3, 9); w = height = 9;
    assert(!h->copy_snapshot_rgb(f, rgb, w, height, err) && rgb.empty() && w == 0 && height == 0);
    f.w = f.full_w = 1936; f.stride = 1984;
    alloc_fail = true; assert(!h->copy_snapshot_rgb(f, rgb, w, height, err) && rgb.empty());
    const int alloc_count_after_failure = create_calls;
    f.w = f.full_w = 1920;
    reset_faults(); rga_fail = true; assert(!h->copy_snapshot_rgb(f, rgb, w, height, err) && rgb.empty());
    reset_faults(); sync_fail = true; assert(!h->copy_snapshot_rgb(f, rgb, w, height, err) && rgb.empty());
    reset_faults();
    f.w = f.full_w = 8192; f.h = f.full_h = 8192; f.stride = 8192; f.hstride = 8192;
    assert(!h->copy_snapshot_rgb(f, rgb, w, height, err) && create_calls == alloc_count_after_failure);
    assert(create_calls == alloc_after_success + 3);
    // The input source DMA-BUF is borrowed and must never be destroyed by the
    // hybrid context.
    assert(std::find(destroyed_fds.begin(), destroyed_fds.end(), 42) == destroyed_fds.end());

    // Stage2's two-pass path must describe the intermediate in pixel units:
    // crop width 1862 => pass_w=117, aligned pixel pitch=128.
    vb::Stage2Spec spec; spec.model_path = model; spec.in_w = 32; spec.in_h = 32;
    auto model_obj = vb::load_rknn_stage2_model(spec, 1, err);
    if (!model_obj) { std::fprintf(stderr, "stage2 model fake failed: %s\n", err.c_str()); return 2; }
    err.clear();
    auto stage2 = vb::make_rknn_stage2_context(model_obj, spec, err);
    if (!stage2) { std::fprintf(stderr, "stage2 context fake failed: %s\n", err.c_str()); return 3; }
    f.w = f.full_w = 1920; f.h = f.full_h = 1080; f.stride = 1984; f.hstride = 1088;
    vb::CropReq crop{0.0f, 0.0f, 0.97f, 0.9f, 1}; vb::TensorView out;
    rga_calls.clear(); rga_rect_calls.clear(); stage2_events.clear(); reset_faults();
    assert(stage2->infer_crops(f, &crop, 1, &out, err) == 0);
    assert(rga_calls.size() == 2 && rga_calls[0].width == 117 &&
           rga_calls[0].height == 61 && rga_calls[0].wstride == 128);
    assert(last_src.width == 117 && last_src.height == 61);
    assert(rga_rect_calls.size() == 2);
    assert(rga_rect_calls[0].src.x == 0 && rga_rect_calls[0].src.y == 0 &&
           rga_rect_calls[0].src.width == 1864 && rga_rect_calls[0].src.height == 972);
    assert(rga_rect_calls[0].dst.width == 117 && rga_rect_calls[0].dst.height == 61);
    assert(rga_rect_calls[1].src.width == 117 && rga_rect_calls[1].src.height == 61);
    assert(rga_rect_calls[1].dst.width == 32 && rga_rect_calls[1].dst.height == 32);
    assert((stage2_events == std::vector<int>{1, 1, 2, 3}));

    // Two valid crops each require their own device-to-CPU barrier before NN.
    vb::CropReq crops[2] = {crop, vb::CropReq{0.02f, 0.02f, 0.95f, 0.85f, 2}};
    rga_calls.clear(); rga_rect_calls.clear(); stage2_events.clear(); reset_faults();
    vb::TensorView outs[2];
    assert(stage2->infer_crops(f, crops, 2, outs, err) == 0);
    assert((stage2_events == std::vector<int>{1, 1, 2, 3, 1, 1, 2, 3}));
    assert(rga_calls.size() == 4 && outs[0].data != nullptr && outs[1].data != nullptr);

    // A barrier failure on the second crop preserves the first RUN but makes
    // the full output array unusable and prevents the second RUN.
    rga_calls.clear(); rga_rect_calls.clear(); stage2_events.clear(); reset_faults();
    sync_from_fail_on_call = 2;
    outs[0] = vb::TensorView{}; outs[1] = vb::TensorView{};
    assert(stage2->infer_crops(f, crops, 2, outs, err) < 0);
    assert((stage2_events == std::vector<int>{1, 1, 2, 3, 1, 1, 2}));
    assert(outs[0].data == nullptr && outs[0].count == 0);
    assert(outs[1].data == nullptr && outs[1].count == 0);
    reset_faults();

    // A temporary DMA-BUF must be closed when either RGA pass fails.
    auto closed_temp_count = [&]() {
        size_t count = 0;
        for (int fd : close_fds)
            if (std::find(alloc_fds.begin(), alloc_fds.end(), fd) != alloc_fds.end()) ++count;
        return count;
    };
    const size_t closed_before = closed_temp_count();
    rga_calls.clear(); rga_rect_calls.clear(); reset_faults(); rga_fail_on_call = 1;
    assert(stage2->infer_crops(f, &crop, 1, &out, err) < 0);
    assert(rga_calls.size() == 1);
    assert(closed_temp_count() == closed_before + 1);
    assert(std::find(alloc_fds.begin(), alloc_fds.end(), close_fds.back()) != alloc_fds.end());
    rga_fail_on_call = 2;
    rga_calls.clear(); rga_rect_calls.clear();
    assert(stage2->infer_crops(f, &crop, 1, &out, err) < 0);
    assert(rga_calls.size() == 2);
    assert(closed_temp_count() == closed_before + 2);
    assert(std::find(alloc_fds.begin(), alloc_fds.end(), close_fds.back()) != alloc_fds.end());
    rga_fail_on_call = 0;
    std::remove(model);
    std::puts("rknn snapshot fake contract: PASS");
}
