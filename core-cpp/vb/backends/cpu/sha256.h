// Minimal SHA-256 (FIPS 180-4), used by the CPU backend for
// Backend::model_sha256(). Self-contained, no external dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vb {

// Returns the lowercase hex digest of the input bytes.
std::string sha256_hex(const void* data, size_t len);

inline std::string sha256_hex(const std::vector<uint8_t>& bytes) {
    return sha256_hex(bytes.data(), bytes.size());
}

}  // namespace vb
