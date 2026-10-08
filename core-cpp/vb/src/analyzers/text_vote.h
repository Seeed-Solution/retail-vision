#pragma once
#include <memory>
#include "vb/analyzer.h"
namespace vb { std::unique_ptr<Analyzer> make_text_vote_analyzer(); }
