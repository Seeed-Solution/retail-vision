#pragma once
#include <mutex>
#include "vb/backend.h"
namespace vb { std::unique_ptr<Stage2Context> make_cvi_stage2(const Stage2Spec&, std::string&, std::mutex*); }
