// SHA-256 hex digest for Backend::model_sha256() (spec BASE-1 §M2.1).
//
// Same FIPS 180-4 implementation as backends/cvi/sha256.cpp, kept under its own
// symbol on purpose: `-DVB_BACKENDS="cpu;rknn"` links both backend OBJECT
// libraries into one vb-runtime, and two definitions of vb::sha256_hex would be
// a duplicate-symbol link error. The digest produced is identical.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vb {
namespace cvi_detail {

// Lowercase hex digest of `len` bytes at `data`.
std::string sha256_hex(const void* data, size_t len);

inline std::string sha256_hex(const std::vector<uint8_t>& bytes) {
    return sha256_hex(bytes.data(), bytes.size());
}

}  // namespace cvi_detail
}  // namespace vb
