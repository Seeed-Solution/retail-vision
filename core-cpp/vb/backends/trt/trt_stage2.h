#pragma once

#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {
struct TrtStage2Model;
std::shared_ptr<TrtStage2Model> load_trt_stage2_model(const Stage2Spec&, std::string&);
std::unique_ptr<Stage2Context> make_trt_stage2_context(
    const std::shared_ptr<TrtStage2Model>&, const Stage2Spec&, std::string&);
}
