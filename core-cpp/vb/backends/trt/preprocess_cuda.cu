#include "preprocess_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace vb {
namespace {

template <typename T>
__device__ inline void storeNormalized(T* destination, std::size_t index, float value);

template <>
__device__ inline void storeNormalized<float>(float* destination, std::size_t index, float value) {
    destination[index] = value;
}

template <>
__device__ inline void storeNormalized<__half>(__half* destination, std::size_t index, float value) {
    destination[index] = __float2half(value);
}

__device__ inline int round_to_even(float value) {
    const float base = floorf(value);
    const float fraction = value - base;
    const int integral = static_cast<int>(base);
    if (fraction > 0.5f) return integral + 1;
    if (fraction < 0.5f) return integral;
    return (integral % 2 == 0) ? integral : integral + 1;
}

__device__ inline float sourcePixel(const unsigned char* source, int width, int height,
                                    int stride, int x, int y, int channel,
                                    float pad_value, float divide, bool model_rgb, bool source_rgb) {
    (void)pad_value;
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    return static_cast<float>(source[static_cast<std::size_t>(y) * stride +
                                     static_cast<std::size_t>(x) * 3 + channel]);
}

template <typename T>
__global__ void preprocessKernel(const unsigned char* source,
                                 int source_width, int source_height,
                                 int source_stride_bytes, T* destination,
                                 int destination_width, int destination_height,
                                 float scale, float pad_x, float pad_y,
                                 float pad_value, float divide, bool model_rgb, bool source_rgb) {
    const std::size_t pixel_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pixel_count = static_cast<std::size_t>(destination_width) * destination_height;
    if (pixel_index >= pixel_count) return;
    const int x = static_cast<int>(pixel_index % static_cast<std::size_t>(destination_width));
    const int y = static_cast<int>(pixel_index / static_cast<std::size_t>(destination_width));
    // Match LetterboxGeom::fit()/CPU preprocess: resized dimensions use
    // Python round (ties to even), then padding is applied to that integer
    // rectangle. Using the raw float dimensions admits one extra row/column
    // for cases such as 7x3 -> 5x5 (scaled height 2.142857).
    const int resized_width = round_to_even(static_cast<float>(source_width) * scale);
    const int resized_height = round_to_even(static_cast<float>(source_height) * scale);
    const float letterbox_pad = pad_value / divide;
    const bool inside = x >= pad_x && y >= pad_y &&
                        x < static_cast<int>(pad_x) + resized_width &&
                        y < static_cast<int>(pad_y) + resized_height;

    const std::size_t plane = static_cast<std::size_t>(destination_width) * destination_height;
    for (int output_channel = 0; output_channel < 3; ++output_channel) {
        float normalized = letterbox_pad;
        if (inside) {
            // Pixel-center mapping matches cv::resize's half-pixel convention
            // closely while keeping the entire preprocessing pass on CUDA.
            const float source_x = ((static_cast<float>(x) - pad_x + 0.5f) / scale) - 0.5f;
            const float source_y = ((static_cast<float>(y) - pad_y + 0.5f) / scale) - 0.5f;
            const int x0 = static_cast<int>(floorf(source_x));
            const int y0 = static_cast<int>(floorf(source_y));
            const float x_weight = source_x - static_cast<float>(x0);
            const float y_weight = source_y - static_cast<float>(y0);
            // TensorRT models conventionally consume RGB.  The capture buffer
            // is BGR, so read the opposite source channel here.
            const int source_channel = source_rgb ? (model_rgb ? output_channel : 2 - output_channel)
                                                   : (model_rgb ? 2 - output_channel : output_channel);
            const float top_left = sourcePixel(source, source_width, source_height,
                                               source_stride_bytes, x0, y0,
                                               source_channel, pad_value, divide, model_rgb, source_rgb);
            const float top_right = sourcePixel(source, source_width, source_height,
                                                source_stride_bytes, x0 + 1, y0,
                                                source_channel, pad_value, divide, model_rgb, source_rgb);
            const float bottom_left = sourcePixel(source, source_width, source_height,
                                                  source_stride_bytes, x0, y0 + 1,
                                                  source_channel, pad_value, divide, model_rgb, source_rgb);
            const float bottom_right = sourcePixel(source, source_width, source_height,
                                                   source_stride_bytes, x0 + 1, y0 + 1,
                                                   source_channel, pad_value, divide, model_rgb, source_rgb);
            const float top = top_left + (top_right - top_left) * x_weight;
            const float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
            normalized = (top + (bottom - top) * y_weight) / divide;
        }
        storeNormalized(destination, static_cast<std::size_t>(output_channel) * plane + pixel_index,
                        normalized);
    }
}

template <typename T>
__global__ void cropPreprocessKernel(const unsigned char* source,
                                     int source_width, int source_height,
                                     int source_stride_bytes, T* destination,
                                     int destination_width, int destination_height,
                                     float x0, float y0, float x1, float y1,
                                     bool model_bgr, bool source_bgr,
                                     float mean0, float mean1, float mean2,
                                     float scale) {
    const std::size_t pixel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(destination_width) * destination_height;
    if (pixel >= count) return;
    const int ox = static_cast<int>(pixel % static_cast<std::size_t>(destination_width));
    const int oy = static_cast<int>(pixel / static_cast<std::size_t>(destination_width));
    const float sx = x0 + (static_cast<float>(ox) + 0.5f) * (x1 - x0) /
                                  static_cast<float>(destination_width) - 0.5f;
    const float sy = y0 + (static_cast<float>(oy) + 0.5f) * (y1 - y0) /
                                  static_cast<float>(destination_height) - 0.5f;
    const int ax = static_cast<int>(floorf(sx));
    const int ay = static_cast<int>(floorf(sy));
    const float fx = sx - static_cast<float>(ax);
    const float fy = sy - static_cast<float>(ay);
    const std::size_t plane = count;
    const float mean[3] = {mean0, mean1, mean2};
    for (int ch = 0; ch < 3; ++ch) {
        const int model_ch = model_bgr ? 2 - ch : ch;
        const int source_ch = source_bgr ? 2 - model_ch : model_ch;
        const float a = static_cast<float>(sourcePixel(source, source_width, source_height,
                                                        source_stride_bytes, ax, ay, source_ch,
                                                        0.0f, 1.0f, false, false));
        const float b = static_cast<float>(sourcePixel(source, source_width, source_height,
                                                        source_stride_bytes, ax + 1, ay, source_ch,
                                                        0.0f, 1.0f, false, false));
        const float c = static_cast<float>(sourcePixel(source, source_width, source_height,
                                                        source_stride_bytes, ax, ay + 1, source_ch,
                                                        0.0f, 1.0f, false, false));
        const float d = static_cast<float>(sourcePixel(source, source_width, source_height,
                                                        source_stride_bytes, ax + 1, ay + 1, source_ch,
                                                        0.0f, 1.0f, false, false));
        const float value = ((a + (b - a) * fx) +
                             ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy) * scale - mean[ch];
        storeNormalized(destination, static_cast<std::size_t>(ch) * plane + pixel, value);
    }
}

}  // namespace

bool launchCudaPreprocess(const unsigned char* source_device,
                          int source_width, int source_height,
                          int source_stride_bytes, void* destination_device,
                          int destination_width, int destination_height,
                          float scale, float pad_x, float pad_y,
                          float pad_value, float divide, bool model_rgb, bool source_rgb,
                          PreprocessOutputType output_type,
                          cudaStream_t stream) {
    if (source_device == nullptr || destination_device == nullptr ||
        source_width <= 0 || source_height <= 0 || destination_width <= 0 ||
        destination_height <= 0 || source_stride_bytes < source_width * 3 ||
        !(scale > 0.0f) || !std::isfinite(scale) || !std::isfinite(divide) || !(divide > 0.0f) || stream == nullptr) {
        return false;
    }
    constexpr int kThreads = 256;
    const std::size_t pixels = static_cast<std::size_t>(destination_width) * destination_height;
    const int blocks = static_cast<int>((pixels + kThreads - 1) / kThreads);
    if (output_type == PreprocessOutputType::Float32) {
        preprocessKernel<<<blocks, kThreads, 0, stream>>>(
            source_device, source_width, source_height, source_stride_bytes,
            static_cast<float*>(destination_device), destination_width, destination_height,
            scale, pad_x, pad_y, pad_value, divide, model_rgb, source_rgb);
    } else {
        preprocessKernel<<<blocks, kThreads, 0, stream>>>(
            source_device, source_width, source_height, source_stride_bytes,
            static_cast<__half*>(destination_device), destination_width, destination_height,
            scale, pad_x, pad_y, pad_value, divide, model_rgb, source_rgb);
    }
    // Peek so the caller can report/clear the launch error with its normal
    // CUDA diagnostic helper.
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launchCudaCropPreprocess(const unsigned char* source_device,
                              int source_width, int source_height,
                              int source_stride_bytes, void* destination_device,
                              int destination_width, int destination_height,
                              float x0, float y0, float x1, float y1,
                              bool model_bgr, bool source_bgr,
                              const float mean[3], float scale,
                              PreprocessOutputType output_type,
                              cudaStream_t stream) {
    if (!source_device || !destination_device || !mean || source_width <= 0 || source_height <= 0 ||
        destination_width <= 0 || destination_height <= 0 || source_stride_bytes < source_width * 3 ||
        !std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1) ||
        !(x0 < x1) || !(y0 < y1) || !std::isfinite(scale) || !(scale > 0.0f) || !stream) return false;
    const std::size_t pixels = static_cast<std::size_t>(destination_width) * destination_height;
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((pixels + kThreads - 1) / kThreads);
    if (output_type == PreprocessOutputType::Float32) {
        cropPreprocessKernel<<<blocks, kThreads, 0, stream>>>(source_device, source_width, source_height,
            source_stride_bytes, static_cast<float*>(destination_device), destination_width, destination_height,
            x0, y0, x1, y1, model_bgr, source_bgr, mean[0], mean[1], mean[2], scale);
    } else {
        cropPreprocessKernel<<<blocks, kThreads, 0, stream>>>(source_device, source_width, source_height,
            source_stride_bytes, static_cast<__half*>(destination_device), destination_width, destination_height,
            x0, y0, x1, y1, model_bgr, source_bgr, mean[0], mean[1], mean[2], scale);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace vb
