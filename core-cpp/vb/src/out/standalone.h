// vb-runtime --standalone (spec BASE-1 §6.10, M1.18/M1.19).
#pragma once

#include <cstdint>
#include <string>

namespace vb {

struct StandaloneOpts {
    std::string config_path;
    std::string output = "jsonl";  // "jsonl" | "mqtt"
    int frame_every = 0;           // 0: no vb.frame/1 records
    int status_every = -1;         // -1: mqtt.status_interval_s (default 10)
};

int run_standalone(const StandaloneOpts& opts);

}  // namespace vb
