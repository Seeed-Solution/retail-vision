// RK3576 / RK3588 backend self-registration (spec BASE-1 §6.1): a static object
// calls register_backend() at load time; core code never branches on names.
#include "rknn_backend.h"
#include "vb/backend.h"

namespace vb {

namespace {
struct RknnBackendRegister {
    RknnBackendRegister() { register_backend("rknn", &make_rknn_backend); }
};
static RknnBackendRegister g_register;

}  // namespace

}  // namespace vb
