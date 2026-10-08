#include "vb/backend.h"

#include <iostream>
#include <cstdlib>
#include <limits>

int main() {
    int checks = 0;
    auto check = [&](bool ok, const char* name) {
        ++checks;
        std::cout << (ok ? "PASS " : "FAIL ") << name << '\n';
        if (!ok) std::exit(1);
    };
    vb::FrameBuf frame;
    size_t bytes = 0;
    check(frame.hstride == 0, "FrameBuf defaults hstride to zero");
    check(vb::nv12_storage_bytes(1920, 1080, 1984, frame.hstride, bytes) &&
          bytes == 3214080, "default-zero legacy visible-height storage");
    check(vb::nv12_storage_bytes(1920, 1080, 1984, 1088, bytes) &&
          bytes == 3237888, "padded height includes full NV12 storage");
    int storage_h = -1;
    auto layout = [&](int w, int h, int ys, int uvs, int planes, size_t yo,
                      size_t uvo, size_t mo, size_t ms) {
        return vb::nv12_dmabuf_layout(w, h, ys, uvs, planes, yo, uvo, mo, ms,
                                      storage_h, bytes);
    };
    check(layout(1920, 1080, 1984, 1984, 2, 0, 2158592, 0, 3237888) &&
          storage_h == 1088 && bytes == 3237888, "observed 1984/1080/1088 layout");
    check(layout(416, 416, 448, 448, 2, 0, 448*416, 0, 448*416*3/2) &&
          storage_h == 416, "existing horizontal-only padding");
    check(!layout(1920,1080,1984,1984,1,0,2158592,0,3237888), "missing UV plane");
    check(!layout(1920,1080,1984,1984,3,0,2158592,0,3237888), "extra plane");
    check(!layout(1920,1080,1984,1920,2,0,2158592,0,3237888), "unequal UV stride");
    check(!layout(1920,1080,1984,1984,2,1,2158592,0,3237888), "nonzero Y offset");
    check(!layout(1920,1080,1984,1984,2,0,2158592,1,3237888), "nonzero backing offset");
    check(!layout(1920,1080,1984,1984,2,0,2158593,0,3237888), "nonintegral UV row offset");
    check(!layout(1920,1080,1984,1984,2,0,1984*1078,0,3237888), "storage below visible height");
    check(!layout(1920,1080,1984,1984,2,0,1984*1087,0,3237888), "odd storage height");
    check(!layout(1920,1080,1984,1984,2,0,1984*8194,0,50000000), "storage exceeds 8192");
    check(!layout(1920,1080,1984,1984,2,0,2158592,0,3237887), "accessible extent one byte short");
    check(!layout(1920,1080,0,0,2,0,2158592,0,3237888), "zero stride");
    check(!layout(1920,1080,-1984,-1984,2,0,2158592,0,3237888), "negative stride");
    check(!layout(1920,1080,1918,1918,2,0,1918*1088,0,3237888), "stride below width");
    check(!layout(1920,1080,1985,1985,2,0,1985*1088,0,3239520), "odd NV12 stride");
    check(!layout(1920,1080,1984,1984,2,0,0,0,3237888), "zero UV offset");
    check(!layout(1920,1080,2,2,2,0,std::numeric_limits<size_t>::max()-1,
                  0,std::numeric_limits<size_t>::max()), "oversize UV offset cannot narrow to int");
    check(!vb::nv12_storage_bytes(1920,1080,1984,-1,bytes), "negative explicit hstride");
    check(!vb::nv12_storage_bytes(1919,1080,1984,1088,bytes), "odd visible width");
    check(!vb::nv12_storage_bytes(1920,1079,1984,1088,bytes), "odd visible height");
    check(vb::nv12_storage_bytes(1920,1080,10000,1088,bytes) && bytes==16320000,
          "stride above 8192 remains permitted");
    check(storage_h == 0, "rejected layout clears storage height");
    std::cout << "checks=" << checks << " result=PASS\n";
}
