#include "vb/json.h"

namespace vb {

Json json_parse(const std::string& text) {
    try {
        return Json::parse(text);
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("json parse error: ") + e.what());
    }
}

std::string json_dump(const Json& j) {
    return j.dump();
}

}  // namespace vb
