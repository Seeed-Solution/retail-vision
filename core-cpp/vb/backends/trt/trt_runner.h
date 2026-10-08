// TensorRT adapter derived from fall-detection@5af128d (Apache-2.0).
// Modified for FrameBuf/TensorView, no OpenCV or business/C API code.
#pragma once
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>
#include "vb/backend.h"
#include "preprocess_cuda.h"

namespace vb {
struct TrtDeleter { template <typename T> void operator()(T* p) const { delete p; } };

struct TrtShared {
    class Logger final : public nvinfer1::ILogger {
    public: void log(Severity s, const char* m) noexcept override;
    } logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine;
    int model_w = 0, model_h = 0;
};

class TrtRunner {
public:
    explicit TrtRunner(std::shared_ptr<TrtShared> shared);
    ~TrtRunner();
    TrtRunner(const TrtRunner&) = delete;
    bool infer(const FrameBuf& frame, const InputSpec& input, std::vector<TensorView>& views,
               std::vector<std::vector<float>>& storage, DetectionResult& result,
               std::string& err);
private:
    std::shared_ptr<TrtShared> shared_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;
    cudaStream_t stream_ = nullptr;
    void* input_device_ = nullptr;
    unsigned char* source_device_ = nullptr;
    size_t source_capacity_ = 0;
    std::vector<void*> output_device_;
    std::vector<std::vector<int64_t>> output_dims_;
    std::vector<nvinfer1::DataType> output_types_;
    unsigned char* source_host_ = nullptr;
    std::vector<unsigned char*> output_host_;
    std::vector<size_t> output_bytes_;
    cudaEvent_t h2d_begin_ = nullptr, h2d_end_ = nullptr;
    cudaEvent_t infer_begin_ = nullptr, infer_end_ = nullptr;
    cudaEvent_t d2h_begin_ = nullptr, d2h_end_ = nullptr;
    std::string input_name_;
    nvinfer1::DataType input_type_ = nvinfer1::DataType::kFLOAT;
    size_t input_bytes_ = 0;
    bool ready_ = false;
    std::string init_error_;
};

std::shared_ptr<TrtShared> load_trt_engine(const std::string& path, std::string& err);
}
