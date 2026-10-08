#pragma once
#include <memory>
#include <string>
#include "vb/backend.h"
namespace vb { std::unique_ptr<FrameSource> make_trt_source(const StreamSpec&, std::string&); }
