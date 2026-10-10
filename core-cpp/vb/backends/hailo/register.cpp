// Hailo backend self-registration (spec BASE-1 §6.1): a static object calls
// register_backend() at load time; core code never branches on names.
#include "hailo_backend.h"
#include "vb/backend.h"

namespace vb {

namespace {
struct HailoBackendRegister {
    HailoBackendRegister() { register_backend("hailo", &make_hailo_backend); }
};
static HailoBackendRegister g_register;

}  // namespace

}  // namespace vb
