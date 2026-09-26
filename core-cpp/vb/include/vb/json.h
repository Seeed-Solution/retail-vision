// Thin JSON wrapper over the vendored nlohmann/json single header (spec BASE-1 §5.3).
#pragma once

#include <string>

#include "nlohmann/json.hpp"

namespace vb {

using Json = nlohmann::json;

// Parse a JSON document; throws std::invalid_argument with context on failure.
Json json_parse(const std::string& text);

// Serialize (compact, like json.dumps with default separators; UTF-8 preserved).
std::string json_dump(const Json& j);

}  // namespace vb
