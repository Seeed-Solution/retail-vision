// Snapshot encoding (spec BASE-1 §5.4 snapshot, §8 M1.8).
//
// Built with VB_HAVE_JPEG (libjpeg-turbo, behind CMake option VB_WITH_JPEG,
// default OFF) snapshots encode real JPEG; otherwise every snapshot request
// replies ok:false,"snapshot unsupported".
#include <algorithm>
#include <cmath>
#include <cstring>

#include "snapshot.h"

#if defined(VB_HAVE_JPEG)
#include <turbojpeg.h>
#endif

namespace vb {

bool snapshot_clip_box(float x0, float y0, float x1, float y1, int w, int h,
                       int& cx0, int& cy0, int& cw, int& ch) {
    if (w <= 0 || h <= 0) return false;
    const float fx0 = std::floor(x0), fy0 = std::floor(y0);
    const float fx1 = std::ceil(x1), fy1 = std::ceil(y1);
    // Rejects empty (x1 <= x0) and NaN boxes.
    if (!(fx1 > fx0) || !(fy1 > fy0)) return false;
    const float fw = static_cast<float>(w), fh = static_cast<float>(h);
    const int ix0 = static_cast<int>(std::max(0.0f, std::min(fw, fx0)));
    const int iy0 = static_cast<int>(std::max(0.0f, std::min(fh, fy0)));
    const int ix1 = static_cast<int>(std::max(0.0f, std::min(fw, fx1)));
    const int iy1 = static_cast<int>(std::max(0.0f, std::min(fh, fy1)));
    // The clipped rect must still have area; y0 == h (or x0 == w) used to pass
    // through and read one row/column past the end of the pixel buffer.
    if (ix1 <= ix0 || iy1 <= iy0) return false;
    cx0 = ix0;
    cy0 = iy0;
    cw = ix1 - ix0;
    ch = iy1 - iy0;
    return true;
}

#if defined(VB_HAVE_JPEG)

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
            int cx0 = 0, cy0 = 0, cw = 0, ch = 0;
            if (!snapshot_clip_box(b.x0, b.y0, b.x1, b.y1, e.w, e.h, cx0, cy0, cw,
                                   ch)) {
                err = "snapshot crop box does not intersect the frame";
                return false;
            }
            cropped.resize(static_cast<size_t>(cw) * ch * 3);
            for (int y = 0; y < ch; ++y) {
                std::memcpy(cropped.data() + static_cast<size_t>(y) * cw * 3,
                            e.pixels.data() +
                                (static_cast<size_t>(y) + static_cast<size_t>(cy0)) *
                                    stride +
                                static_cast<size_t>(cx0) * 3,
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
