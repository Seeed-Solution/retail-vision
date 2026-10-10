// SHA-256 hex digest for Backend::model_sha256() (spec BASE-1 §M2.3).
//
// Same FIPS 180-4 implementation as backends/cpu/sha256.cpp and
// backends/rknn/model_sha256.cpp, kept under its own symbol on purpose:
// `-DVB_BACKENDS="cpu;rknn;hailo"` links all backend OBJECT libraries into one
// vb-runtime, and two definitions of vb::sha256_hex would be a
// duplicate-symbol link error. The digest produced is identical.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vb {
namespace hailo_detail {

// Lowercase hex digest of `len` bytes at `data`.
std::string sha256_hex(const void* data, size_t len);

inline std::string sha256_hex(const std::vector<uint8_t>& bytes) {
    return sha256_hex(bytes.data(), bytes.size());
}

}  // namespace hailo_detail
}  // namespace vb
