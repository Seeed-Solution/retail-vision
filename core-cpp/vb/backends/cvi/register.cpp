#include "cvi_backend.h"
#include "vb/backend.h"
namespace vb { namespace { struct Reg { Reg(){ register_backend("cvi", &make_cvi_backend); } } reg; } }
