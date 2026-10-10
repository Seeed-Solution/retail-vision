// RuntimeConfig::from_json rejects values this runtime would ignore
// (backend.align, backend.nms_threshold, native.jpeg_quality) with the same
// messages as vision_base/config.py, and accepts the defaults or absent keys.
#include <string>

#include "check.h"
#include "vb/json.h"
#include "vb/runtime.h"

using namespace vb;

static std::string load_err(const char* text) {
    std::string err;
    RuntimeConfig::from_json(json_parse(text), err, false);
    return err;
}

int main() {
    // absent keys and defaults are accepted
    CHECK_STREQ(load_err(R"({"backend": {"name": "synthetic"}})"), "");
    CHECK_STREQ(load_err(R"({"backend": {"name": "synthetic", "align": "center",
                                         "nms_threshold": 0.45},
                             "native": {"jpeg_quality": 85}})"), "");

    CHECK_STREQ(load_err(R"({"backend": {"name": "synthetic", "align": "top_left"}})"),
                "backend.align: 'top_left' is not supported by this vb-runtime "
                "(only 'center')");
    CHECK_STREQ(load_err(R"({"backend": {"name": "synthetic", "nms_threshold": 0.5}})"),
                "backend.nms_threshold: 0.5 is not supported by this vb-runtime "
                "(only 0.45)");
    CHECK_STREQ(load_err(R"({"backend": {"name": "synthetic"},
                             "native": {"jpeg_quality": 90}})"),
                "native.jpeg_quality: 90 is not supported by this vb-runtime "
                "(only 85)");
    return 0;
}
