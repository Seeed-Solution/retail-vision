#pragma once

#include <memory>
#include <string>

#include "vb/backend.h"

namespace vb {

// The model object owns one RKNN context used only as the source for
// rknn_dup_context(). Each returned Stage2Context owns its duplicate context,
// input/output memory and output storage.
class RknnStage2Model;

std::shared_ptr<RknnStage2Model> load_rknn_stage2_model(
    const Stage2Spec& spec, uint32_t core_mask, std::string& err);
std::unique_ptr<Stage2Context> make_rknn_stage2_context(
    const std::shared_ptr<RknnStage2Model>& model, const Stage2Spec& spec,
    std::string& err);

}  // namespace vb
