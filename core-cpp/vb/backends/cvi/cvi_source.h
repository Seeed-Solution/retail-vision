#pragma once
#include "vb/backend.h"
namespace vb { std::unique_ptr<FrameSource> make_cvi_source(const StreamSpec&, std::string&); }
