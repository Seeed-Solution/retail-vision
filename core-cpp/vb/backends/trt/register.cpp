// TensorRT backend self-registration (BASE-1 §6.1). Kept in OBJECT library.
#include "trt_backend.h"
#include "vb/backend.h"
namespace vb { namespace { struct Register { Register(){ register_backend("trt", &make_trt_backend); } }; static Register reg; } }
