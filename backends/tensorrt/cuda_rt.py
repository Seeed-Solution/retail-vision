"""Minimal CUDA runtime binding via ctypes.

The Jetson compose file already mounts the host CUDA tree into the container
(`/usr/local/cuda`), so `libcudart.so` is present at runtime. Everything the
TensorRT execution path needs from CUDA is four calls -- allocate device
memory, copy in, copy out, synchronise -- so they are bound directly with
ctypes rather than pulling in a wheel. pycuda would need a compiler on the
board and torch is exactly the dependency this backend exists to drop.
"""
from __future__ import annotations

import ctypes

_MEMCPY_HOST_TO_DEVICE = 1
_MEMCPY_DEVICE_TO_HOST = 2

# cudaStreamCreate hands back a *blocking* stream that implicitly synchronises
# with the legacy default stream. With one detector per RTSP source, every
# runner then serialises against every other one the moment anything touches
# the default stream — no error, just lost throughput that only shows up
# multi-camera. (Lesson taken from fall-detection's
# platforms/jetson/main/trt_runner.cpp, which hit exactly this.)
_STREAM_NON_BLOCKING = 0x01
_HOST_ALLOC_DEFAULT = 0x00

_CANDIDATES = (
    "libcudart.so",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "/usr/local/cuda/lib64/libcudart.so",
)


def _load():
    errors = []
    for name in _CANDIDATES:
        try:
            return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:  # try the next soname
            errors.append(f"{name}: {exc}")
    raise RuntimeError("libcudart not found; tried:\n  " + "\n  ".join(errors))


_LIB = _load()

_LIB.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_LIB.cudaMalloc.restype = ctypes.c_int
_LIB.cudaFree.argtypes = [ctypes.c_void_p]
_LIB.cudaFree.restype = ctypes.c_int
_LIB.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.c_int, ctypes.c_void_p]
_LIB.cudaMemcpyAsync.restype = ctypes.c_int
_LIB.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_uint]
_LIB.cudaStreamCreateWithFlags.restype = ctypes.c_int
# Pinned (page-locked) host staging. A pageable copy destination forces the
# driver through its own staging buffer, so the "async" copy cannot actually
# overlap with compute.
_LIB.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                               ctypes.c_uint]
_LIB.cudaHostAlloc.restype = ctypes.c_int
_LIB.cudaFreeHost.argtypes = [ctypes.c_void_p]
_LIB.cudaFreeHost.restype = ctypes.c_int
_LIB.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_LIB.cudaStreamSynchronize.restype = ctypes.c_int
_LIB.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
_LIB.cudaStreamDestroy.restype = ctypes.c_int
_LIB.cudaGetErrorString.argtypes = [ctypes.c_int]
_LIB.cudaGetErrorString.restype = ctypes.c_char_p


def _check(code, what):
    if code != 0:
        message = _LIB.cudaGetErrorString(code)
        text = message.decode() if message else "unknown error"
        raise RuntimeError(f"{what} failed: cudaError {code} ({text})")


def malloc(nbytes: int) -> ctypes.c_void_p:
    pointer = ctypes.c_void_p()
    _check(_LIB.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(nbytes)), "cudaMalloc")
    return pointer


def free(pointer) -> None:
    if pointer:
        _check(_LIB.cudaFree(pointer), "cudaFree")


def stream_create() -> ctypes.c_void_p:
    stream = ctypes.c_void_p()
    _check(_LIB.cudaStreamCreateWithFlags(ctypes.byref(stream),
                                          _STREAM_NON_BLOCKING),
           "cudaStreamCreateWithFlags")
    return stream


def host_alloc(nbytes: int) -> ctypes.c_void_p:
    """Page-locked host memory, so H2D/D2H copies are genuinely asynchronous."""
    pointer = ctypes.c_void_p()
    _check(_LIB.cudaHostAlloc(ctypes.byref(pointer), nbytes,
                              _HOST_ALLOC_DEFAULT), "cudaHostAlloc")
    return pointer


def host_free(pointer: ctypes.c_void_p) -> None:
    if pointer and pointer.value:
        _check(_LIB.cudaFreeHost(pointer), "cudaFreeHost")


def stream_destroy(stream) -> None:
    if stream:
        _check(_LIB.cudaStreamDestroy(stream), "cudaStreamDestroy")


def stream_synchronize(stream) -> None:
    _check(_LIB.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


def memcpy_host_to_device(device_ptr, host_array, stream) -> None:
    _check(_LIB.cudaMemcpyAsync(device_ptr,
                                ctypes.c_void_p(host_array.ctypes.data),
                                ctypes.c_size_t(host_array.nbytes),
                                _MEMCPY_HOST_TO_DEVICE, stream),
           "cudaMemcpyAsync(H2D)")


def memcpy_device_to_host(host_array, device_ptr, stream) -> None:
    _check(_LIB.cudaMemcpyAsync(ctypes.c_void_p(host_array.ctypes.data),
                                device_ptr,
                                ctypes.c_size_t(host_array.nbytes),
                                _MEMCPY_DEVICE_TO_HOST, stream),
           "cudaMemcpyAsync(D2H)")


def pinned_array(shape, dtype):
    """(numpy view, owning pointer) over page-locked host memory.

    The array is a view onto CUDA-owned memory, so the pointer has to outlive
    it and be released with host_free(); numpy will not do that for you.
    """
    import numpy as np

    nbytes = int(np.prod(shape)) * int(np.dtype(dtype).itemsize)
    pointer = host_alloc(nbytes)
    buffer = (ctypes.c_byte * nbytes).from_address(pointer.value)
    array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    return array, pointer
