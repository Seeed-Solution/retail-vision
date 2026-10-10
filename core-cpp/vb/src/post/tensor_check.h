// Shared decoder input validation (review fix D2: buffer length / null
// pointer). Header-only so the decoders stay in their existing translation
// units (no build-file change).
//
// A TensorView carries both a shape and the number of floats actually
// readable at `data`. Decoders used to trust the shape alone, so a view whose
// count was smaller than the shape implied (or whose data was null) read out
// of bounds. Every decoder now validates its inputs through check_view()
// before touching any element.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "vb/decoder.h"

namespace vb {
namespace post_detail {

// Product of the dims of a batch-stripped view. False when a dim is
// non-positive (or the dim list is empty) or the product overflows size_t.
inline bool shape_elements(const TensorView& t, size_t& elem) {
    if (t.dims.empty()) return false;
    elem = 1;
    for (int64_t d : t.dims) {
        if (d <= 0) return false;
        const size_t u = static_cast<size_t>(d);
        if (u > std::numeric_limits<size_t>::max() / elem) return false;
        elem *= u;
    }
    return true;
}

// True when the view is safe to read: non-null data, positive non-overflowing
// dims, and `count` exactly equal to the dims product. On failure err is set
// with the context tag `ctx` (e.g. "yolov8 decoder").
//
// Exact equality is deliberate: the adapters pass batch-1 tensors with the
// batch dimension already stripped from `dims`, so a larger `count` means the
// decoder and the adapter disagree about the layout, and a smaller one means
// the shape cannot be read in full.
inline bool check_view(const TensorView& t, const char* ctx, std::string& err) {
    if (t.data == nullptr) {
        err = std::string(ctx) + ": tensor data is null";
        return false;
    }
    size_t elem = 0;
    if (!shape_elements(t, elem)) {
        err = std::string(ctx) +
              ": tensor dims are empty, non-positive or overflow";
        return false;
    }
    if (t.count != elem) {
        err = std::string(ctx) + ": tensor holds " + std::to_string(t.count) +
              " elements but its dims describe " + std::to_string(elem);
        return false;
    }
    return true;
}

// Model canvas must be positive, otherwise the normalized outputs below
// divide by zero and non-finite values reach the detections.
inline bool check_canvas(int model_w, int model_h, const char* ctx,
                         std::string& err) {
    if (model_w <= 0 || model_h <= 0) {
        err = std::string(ctx) + ": model canvas must be positive";
        return false;
    }
    return true;
}

// Non-finite handling (review fix D4). A NaN score escapes the
// `score <= threshold` comparison and is emitted, where it becomes JSON null;
// so the affected candidate is dropped instead. The count is reported through
// err, the only error channel a Decoder has (callers ignore err on success
// today; the note keeps the event observable).
inline void note_dropped(std::string& err, const char* ctx, size_t dropped) {
    if (dropped == 0) return;
    err = std::string(ctx) + ": dropped " + std::to_string(dropped) +
          " non-finite candidate(s)";
}

}  // namespace post_detail
}  // namespace vb
