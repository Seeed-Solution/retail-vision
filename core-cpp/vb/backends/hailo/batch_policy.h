// Hailo batch-size policy (spec BASE-1 §M2.3).
//
// Copied from fall-detection platforms/rpi-hailo/src/batch_policy.{h,cpp}
// (origin/main@eb72e1e, Apache-2.0; see NOTICE). Namespaced to vb::; logic is
// unchanged. decide/caps wiring: caps.max_batch comes from chooseBatch() with
// the stream count the deployment configures (backend.batch.streams), so the
// 16-stream setup can select the shared batch-8 behaviour while a single
// stream stays at batch 1 (spec §11 D1: the Pi 5 is CPU-bound; per-stream
// batch 1 must not wait for a full batch).
#pragma once

#include <string>

namespace vb {

enum class BatchMode { Auto, Off, Fixed };
struct BatchConfig { BatchMode mode; int fixed_size = 1; };
struct BatchDecision { bool multi_context = false; bool shared = false; int batch_size = 1; };

BatchConfig parseBatchMode(const std::string &value);
int parseBatchWaitMs(const std::string &value);
BatchDecision chooseBatch(const BatchConfig &config, int network_group_count,
                          bool network_group_multi_context, int streams);
const char *batchModeName(const BatchConfig &config);
}
