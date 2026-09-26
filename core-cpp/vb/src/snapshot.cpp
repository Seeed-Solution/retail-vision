// Snapshot encoding (spec BASE-1 §5.4 snapshot, §8 M1.8).
//
// Built with VB_HAVE_JPEG (libjpeg-turbo, behind CMake option VB_WITH_JPEG,
// default OFF) snapshots encode real JPEG; otherwise every snapshot request
// replies ok:false,"snapshot unsupported".
#include <algorithm>
#include <cstring>

#include "snapshot.h"

#if defined(VB_HAVE_JPEG)
#include <turbojpeg.h>
#endif

namespace vb {

#if defined(VB_HAVE_JPEG)

namespace {

void clamp_box(SnapBox& b, int w, int h) {
    b.x0 = std::max(0.0f, std::min(static_cast<float>(w), std::floor(b.x0)));
    b.y0 = std::max(0.0f, std::min(static_cast<float>(h), std::floor(b.y0)));
    b.x1 = std::max(1.0f, std::min(static_cast<float>(w), std::ceil(b.x1)));
    b.y1 = std::max(1.0f, std::min(static_cast<float>(h), std::ceil(b.y1)));
}

}  // namespace

bool snapshot_encode_jpeg(const SnapshotRingEntry& e, uint32_t track_id, bool crop,
                          int max_side, std::vector<uint8_t>& jpeg, int& out_w,
                          int& out_h, std::string& err) {
    const uint8_t* src = e.pixels.data();
    int w = e.w, h = e.h, stride = e.w * 3;
    std::vector<uint8_t> cropped;
    if (crop && track_id != 0) {
        bool found = false;
        SnapBox b{};
        for (const auto& box : e.boxes)
            if (box.track_id == track_id) {
                b = box;
                found = true;
                break;
            }
        if (found) {
            clamp_box(b, e.w, e.h);
            int cw = std::max(1, static_cast<int>(b.x1 - b.x0));
            int ch = std::max(1, static_cast<int>(b.y1 - b.y0));
            cropped.resize(static_cast<size_t>(cw) * ch * 3);
            for (int y = 0; y < ch; ++y) {
                std::memcpy(cropped.data() + static_cast<size_t>(y) * cw * 3,
                            e.pixels.data() +
                                (static_cast<size_t>(y) + static_cast<size_t>(b.y0)) *
                                    stride +
                                static_cast<size_t>(b.x0) * 3,
                            static_cast<size_t>(cw) * 3);
            }
            src = cropped.data();
            w = cw;
            h = ch;
            stride = cw * 3;
        }
        // Track not found in this frame: full frame (caller sees seq in meta).
    }
    // Downscale to max_side (nearest-neighbor; snapshot quality is not on the
    // frame path). The turbojpeg scaling API is not used so this stays
    // dependency-free beyond the encoder itself.
    if (max_side > 0 && std::max(w, h) > max_side) {
        float s = static_cast<float>(max_side) / static_cast<float>(std::max(w, h));
        int nw = std::max(1, static_cast<int>(w * s));
        int nh = std::max(1, static_cast<int>(h * s));
        std::vector<uint8_t> scaled(static_cast<size_t>(nw) * nh * 3);
        for (int y = 0; y < nh; ++y) {
            int sy = std::min(h - 1, static_cast<int>(y / s));
            for (int x = 0; x < nw; ++x) {
                int sx = std::min(w - 1, static_cast<int>(x / s));
                const uint8_t* p = src + static_cast<size_t>(sy) * stride +
                                   static_cast<size_t>(sx) * 3;
                uint8_t* q = scaled.data() + (static_cast<size_t>(y) * nw + x) * 3;
                q[0] = p[0];
                q[1] = p[1];
                q[2] = p[2];
            }
        }
        cropped = std::move(scaled);
        src = cropped.data();
        w = nw;
        h = nh;
        stride = nw * 3;
    }

    tjhandle handle = tjInitCompress();
    if (!handle) {
        err = "tjInitCompress failed";
        return false;
    }
    unsigned long size = 0;
    unsigned char* buf = nullptr;
    if (tjCompress2(handle, const_cast<unsigned char*>(src), w, stride, h, TJPF_RGB,
                    &buf, &size, TJSAMP_420, 85, TJFLAG_FASTDCT) != 0) {
        err = std::string("jpeg encode: ") + tjGetErrorStr2(handle);
        tjFree(buf);
        tjDestroy(handle);
        return false;
    }
    jpeg.assign(buf, buf + size);
    tjFree(buf);
    tjDestroy(handle);
    out_w = w;
    out_h = h;
    return true;
}

#else  // !VB_HAVE_JPEG

bool snapshot_encode_jpeg(const SnapshotRingEntry& e, uint32_t track_id, bool crop,
                          int max_side, std::vector<uint8_t>& jpeg, int& out_w,
                          int& out_h, std::string& err) {
    (void)e;
    (void)track_id;
    (void)crop;
    (void)max_side;
    (void)jpeg;
    (void)out_w;
    (void)out_h;
    err = "snapshot unsupported";
    return false;
}

#endif  // VB_HAVE_JPEG

}  // namespace vb
