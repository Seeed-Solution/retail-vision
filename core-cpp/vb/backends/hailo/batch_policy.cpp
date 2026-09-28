// Hailo batch-size policy (spec BASE-1 §M2.3).
//
// Copied from fall-detection platforms/rpi-hailo/src/batch_policy.cpp
// (origin/main@eb72e1e, Apache-2.0; see NOTICE). Namespaced to vb::; logic is
// unchanged.
#include "batch_policy.h"
#include <charconv>
#include <stdexcept>

namespace vb {
namespace {
int integer(const char *name, const std::string &s, int lo, int hi) {
  int n = 0; auto r = std::from_chars(s.data(), s.data() + s.size(), n);
  if (s.empty() || r.ec != std::errc{} || r.ptr != s.data() + s.size() || n < lo || n > hi)
    throw std::invalid_argument(std::string(name) + " out of range");
  return n;
}
}
BatchConfig parseBatchMode(const std::string &v) {
  if (v == "auto") return {BatchMode::Auto, 1};
  if (v == "off") return {BatchMode::Off, 1};
  if (v == "1" || v == "4" || v == "8") return {BatchMode::Fixed, integer("HAILO_BATCH_MODE", v, 1, 8)};
  throw std::invalid_argument("HAILO_BATCH_MODE must be auto, off, 1, 4, or 8");
}
int parseBatchWaitMs(const std::string &v) { return integer("HAILO_BATCH_WAIT_MS", v, 0, 1000); }
BatchDecision chooseBatch(const BatchConfig &c, int groups, bool mc, int streams) {
  if (groups < 1 || streams < 1) throw std::invalid_argument("invalid network group or stream count");
  if (c.mode == BatchMode::Off) return {false, false, 1};
  if (c.mode == BatchMode::Fixed) return {mc, true, c.fixed_size};
  if (groups != 1 || !mc) return {mc, false, 1};
  return {true, true, streams <= 3 ? 1 : streams == 4 ? 4 : 8};
}
const char *batchModeName(const BatchConfig &c) { return c.mode == BatchMode::Auto ? "auto" : c.mode == BatchMode::Off ? "off" : "fixed"; }
}
