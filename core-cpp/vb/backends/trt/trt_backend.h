#pragma once
#include <memory>
#include <string>
#include "vb/backend.h"
namespace vb { std::unique_ptr<Backend> make_trt_backend(const std::string&, std::string&); }
