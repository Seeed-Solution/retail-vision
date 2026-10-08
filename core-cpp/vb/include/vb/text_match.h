#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "vb/json.h"

namespace vb {
struct TextPatternSegment { std::vector<uint32_t> set; int min = 0; int max = 0; };
using TextPattern = std::vector<TextPatternSegment>;
bool decode_utf8(const std::string& s, std::vector<uint32_t>& out);
bool parse_text_patterns(const Json& j, std::vector<TextPattern>& out, std::string& err);
bool text_matches(const std::string& text, const std::vector<TextPattern>& patterns);
}
