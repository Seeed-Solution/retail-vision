// CPU backend self-registration (spec BASE-1 §6.1): a static object calls
// register_backend() at load time; core code never branches on names.
#include "cpu_backend.h"
#include "vb/backend.h"

namespace vb {

namespace {
struct CpuBackendRegister {
    CpuBackendRegister() { register_backend("cpu", &make_cpu_backend); }
};
static CpuBackendRegister g_register;

}  // namespace

}  // namespace vb
