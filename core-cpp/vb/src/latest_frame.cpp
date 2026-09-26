// LatestFrame: single-slot drop-old mailbox (spec BASE-1 §5.4, M1.7).
#include <mutex>
#include "vb/runtime.h"

namespace vb {

void LatestFrame::put(FrameBuf f) {
    std::lock_guard<std::mutex> lk(mu_);
    if (slot_.has_value()) ++dropped_;  // old frame's hold releases on destroy
    slot_ = std::move(f);
}

bool LatestFrame::take(FrameBuf& out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!slot_.has_value()) return false;
    out = std::move(*slot_);
    slot_.reset();
    return true;
}

uint64_t LatestFrame::dropped() const {
    std::lock_guard<std::mutex> lk(mu_);
    return dropped_;
}

}  // namespace vb
