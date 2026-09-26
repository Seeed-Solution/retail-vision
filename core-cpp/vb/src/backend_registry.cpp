// Name-keyed backend registry (spec BASE-1 §6.1, M1.7). Core code registers
// no platform adapter; adapters self-register via register_backend().
// The synthetic backend (deterministic test frames + detections,
// src/sources/synthetic.cpp) is a built-in dev/test adapter, not a platform.
#include <map>
#include <memory>
#include <mutex>

#include "vb/backend.h"
#include "sources/synthetic.h"

namespace vb {

namespace {

std::mutex& reg_mu() {
    static std::mutex m;
    return m;
}

std::map<std::string, BackendFactory>& registry() {
    static std::map<std::string, BackendFactory> r;
    return r;
}

struct RegisterSynthetic {
    RegisterSynthetic() {
        register_backend("synthetic", &make_synthetic_backend);
    }
};
const RegisterSynthetic kRegisterSynthetic;

}  // namespace

bool register_backend(const char* name, BackendFactory f) {
    std::lock_guard<std::mutex> lk(reg_mu());
    return registry().emplace(std::string(name), f).second;
}

std::unique_ptr<Backend> create_backend(const std::string& name,
                                        const std::string& backend_json,
                                        std::string& err) {
    BackendFactory f = nullptr;
    {
        std::lock_guard<std::mutex> lk(reg_mu());
        auto it = registry().find(name);
        if (it != registry().end()) f = it->second;
    }
    if (!f) {
        err = "unknown backend: " + name;
        return nullptr;
    }
    return f(backend_json, err);
}

}  // namespace vb
