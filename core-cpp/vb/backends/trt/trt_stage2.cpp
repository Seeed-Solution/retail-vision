#include "trt_stage2.h"
#include "preprocess_cuda.h"
#include "trt_runner.h"
#include "../rknn/model_sha256.h"
#include "vb/decoder.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vb { namespace {
constexpr size_t kMaxFrameBytes = 8192u * 8192u * 3u;
constexpr size_t kMaxCrops = 16;
size_t type_bytes(nvinfer1::DataType t) { return t == nvinfer1::DataType::kFLOAT ? 4 : t == nvinfer1::DataType::kHALF ? 2 : 0; }
bool volume(const nvinfer1::Dims& d, size_t& n) { if (d.nbDims <= 0 || d.nbDims > 8) return false; n = 1; for (int i=0;i<d.nbDims;++i) { if (d.d[i] <= 0 || n > SIZE_MAX/static_cast<size_t>(d.d[i])) return false; n *= static_cast<size_t>(d.d[i]); } return true; }
bool ck(cudaError_t x, const char* w, std::string& e) { if (x == cudaSuccess) return true; e = std::string("trt stage2 ") + w + ": " + cudaGetErrorString(x); return false; }
bool valid(const CropReq& c) { return std::isfinite(c.x0)&&std::isfinite(c.y0)&&std::isfinite(c.x1)&&std::isfinite(c.y1)&&c.x0>=0&&c.y0>=0&&c.x1<=1&&c.y1<=1&&c.x0<c.x1&&c.y0<c.y1; }
struct Rect { float x0,y0,x1,y1; };
bool rect(const FrameBuf& f, const CropReq& c, Rect& r) {
    float x0,y0,x1,y1; int bw=f.w,bh=f.h,ox=0,oy=0;
    if (f.letterboxed) { f.geom.to_model_norm(c.x0,c.y0,x0,y0); f.geom.to_model_norm(c.x1,c.y1,x1,y1); }
    else { x0=c.x0; y0=c.y0; x1=c.x1; y1=c.y1; bw=f.full_w>0?f.full_w:f.w; bh=f.full_h>0?f.full_h:f.h; ox=f.crop_x0; oy=f.crop_y0; }
    r.x0=std::clamp(x0*bw-ox,0.0f,static_cast<float>(f.w)); r.y0=std::clamp(y0*bh-oy,0.0f,static_cast<float>(f.h));
    r.x1=std::clamp(x1*bw-ox,0.0f,static_cast<float>(f.w)); r.y1=std::clamp(y1*bh-oy,0.0f,static_cast<float>(f.h)); return r.x0<r.x1&&r.y0<r.y1;
}
bool read(const std::string& p, std::vector<uint8_t>& b, std::string& e) { std::ifstream f(p,std::ios::binary|std::ios::ate); if(!f){e="trt stage2: cannot open engine: "+p;return false;} auto n=f.tellg(); if(n<=0||static_cast<uint64_t>(n)>(512ull<<20)){e="trt stage2: engine is empty or too large";return false;} b.resize(static_cast<size_t>(n));f.seekg(0);if(!f.read(reinterpret_cast<char*>(b.data()),n)){e="trt stage2: engine read failed";return false;}return true; }
std::mutex cache_mu;
std::mutex cache_load_mu;
std::unordered_map<std::string, std::weak_ptr<TrtStage2Model>> cache;
}

struct TrtStage2Model { std::shared_ptr<TrtShared> shared; std::string sha; std::string input_name,output_name; std::vector<int64_t> dims; nvinfer1::DataType in_type{},out_type{}; size_t in_bytes=0,out_bytes=0; };

class TrtStage2 final : public Stage2Context {
public:
 TrtStage2(std::shared_ptr<TrtStage2Model> m, Stage2Spec s, std::string& e):model_(std::move(m)),spec_(s) {
    if(!model_){e="trt stage2 model unavailable";return;} ctx_.reset(model_->shared->engine->createExecutionContext()); if(!ctx_){e="trt stage2 createExecutionContext failed";return;}
    if(!ck(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"stream",e)||!ck(cudaMalloc(&input_,model_->in_bytes),"input allocation",e)||!ck(cudaMalloc(&output_,model_->out_bytes),"output allocation",e)||!ck(cudaHostAlloc(&out_host_,model_->out_bytes,cudaHostAllocPortable),"output staging",e)||!ctx_->setTensorAddress(model_->input_name.c_str(),input_)||!ctx_->setTensorAddress(model_->output_name.c_str(),output_)){if(e.empty())e="trt stage2 tensor address setup failed";return;}ready_=true;
 }
 ~TrtStage2() override { if(stream_)cudaStreamSynchronize(stream_); if(input_)cudaFree(input_);if(output_)cudaFree(output_);if(out_host_)cudaFreeHost(out_host_);if(src_dev_)cudaFree(src_dev_);if(src_host_)cudaFreeHost(src_host_);if(stream_)cudaStreamDestroy(stream_); }
 int infer_crops(const FrameBuf& f,const CropReq* c,size_t n,TensorView* o,std::string& e) override {
    // n is caller-owned; never walk an unbounded output array on malformed input.
    clear(o, n <= kMaxCrops ? n : 0);
    if(!ready_){e="trt stage2 context is not ready";return -1;} if(!c||!o||n==0||n>kMaxCrops){e="trt stage2 invalid crop list";return -1;}
    if(f.mem!=Mem::Host||!f.host||(f.fmt!=PixFmt::RGB888&&f.fmt!=PixFmt::BGR888)||f.w<=0||f.h<=0||f.w>8192||f.h>8192){e="trt stage2 requires valid host RGB/BGR frame";return -1;}
    const size_t row_bytes=static_cast<size_t>(f.w)*3;
    if(f.stride<0||static_cast<size_t>(f.stride)<row_bytes||static_cast<size_t>(f.h)>kMaxFrameBytes/row_bytes){e="trt stage2 requires valid host RGB/BGR frame";return -1;}
    std::lock_guard<std::mutex> lk(mu_); if(!stage_source(f,e)){clear(o,n);return -1;} const auto typ=model_->in_type==nvinfer1::DataType::kFLOAT?PreprocessOutputType::Float32:PreprocessOutputType::Float16;
    outputs_.assign(n, {});
    for(size_t i=0;i<n;++i){if(!valid(c[i])){e="trt stage2 crop bbox is invalid";clear(o,n);return -1;}Rect r;if(!rect(f,c[i],r)){e="trt stage2 crop is empty after clamp";clear(o,n);return -1;}if(!launchCudaCropPreprocess(src_dev_,f.w,f.h,f.w*3,input_,spec_.in_w,spec_.in_h,r.x0,r.y0,r.x1,r.y1,spec_.bgr,f.fmt==PixFmt::BGR888,spec_.mean,spec_.scale,typ,stream_)){e="trt stage2 CUDA crop preprocess failed";sync();clear(o,n);return -1;}if(!ctx_->enqueueV3(stream_)){e="trt stage2 enqueueV3 failed";sync();clear(o,n);return -1;}if(!ck(cudaMemcpyAsync(out_host_,output_,model_->out_bytes,cudaMemcpyDeviceToHost,stream_),"output copy",e)||!ck(cudaStreamSynchronize(stream_),"stream synchronize",e)){sync();clear(o,n);return -1;}if(!copy(outputs_[i],e)){clear(o,n);return -1;}o[i].data=outputs_[i].data();o[i].count=outputs_[i].size();o[i].dims=model_->dims;o[i].dtype=0;o[i].raw_data=nullptr;}
    return 0;
 }
 int infer_rgb(const uint8_t* p,int w,int h,int stride,TensorView* o,std::string& e) override {
    clear(o,1);
    if(!p||!o||w<=0||h<=0||w>8192||h>8192){e="trt stage2 infer_rgb frame invalid";return -1;}
    const size_t row_bytes=static_cast<size_t>(w)*3;
    if(stride<0||static_cast<size_t>(stride)<row_bytes||static_cast<size_t>(h)>kMaxFrameBytes/row_bytes){e="trt stage2 infer_rgb frame invalid";return -1;}
    FrameBuf f;f.w=w;f.h=h;f.stride=stride;f.host=p;f.fmt=PixFmt::RGB888;f.mem=Mem::Host;CropReq c{0,0,1,1};return infer_crops(f,&c,1,o,e);}
private:
 static void clear(TensorView* o,size_t n){if(o)for(size_t i=0;i<n;++i)o[i]=TensorView{};}
 bool stage_source(const FrameBuf& f,std::string& e){size_t bytes=static_cast<size_t>(f.w)*static_cast<size_t>(f.h)*3;if(bytes>src_cap_){unsigned char*nd=nullptr,*nh=nullptr;if(!ck(cudaMalloc(&nd,bytes),"source allocation",e)||!ck(cudaHostAlloc(&nh,bytes,cudaHostAllocPortable),"source staging",e)){if(nd)cudaFree(nd);if(nh)cudaFreeHost(nh);return false;}if(src_dev_)cudaFree(src_dev_);if(src_host_)cudaFreeHost(src_host_);src_dev_=nd;src_host_=nh;src_cap_=bytes;}for(int y=0;y<f.h;++y)std::memcpy(src_host_+static_cast<size_t>(y)*f.w*3,f.host+static_cast<size_t>(y)*f.stride,static_cast<size_t>(f.w)*3);return ck(cudaMemcpyAsync(src_dev_,src_host_,bytes,cudaMemcpyHostToDevice,stream_),"source copy",e);}
 void sync(){if(stream_)cudaStreamSynchronize(stream_);}
 bool copy(std::vector<float>& dst,std::string& e){dst.resize(model_->out_bytes/type_bytes(model_->out_type));if(model_->out_type==nvinfer1::DataType::kFLOAT)std::memcpy(dst.data(),out_host_,model_->out_bytes);else{auto*p=static_cast<const __half*>(out_host_);for(size_t i=0;i<dst.size();++i)dst[i]=__half2float(p[i]);}for(float v:dst)if(!std::isfinite(v)){e="trt stage2 output contains NaN or Inf";return false;}return true;}
 std::shared_ptr<TrtStage2Model> model_;Stage2Spec spec_;std::unique_ptr<nvinfer1::IExecutionContext,TrtDeleter> ctx_;cudaStream_t stream_=nullptr;void*input_=nullptr;void*output_=nullptr;void*out_host_=nullptr;unsigned char*src_dev_=nullptr,*src_host_=nullptr;size_t src_cap_=0;std::mutex mu_;std::vector<std::vector<float>> outputs_;bool ready_=false;
};

std::shared_ptr<TrtStage2Model> load_trt_stage2_model(const Stage2Spec& s,std::string& e){
 if(s.in_h<=0||s.in_w<=0||s.in_h>4096||s.in_w>4096||!std::isfinite(s.scale)||s.scale<=0||!std::isfinite(s.mean[0])||!std::isfinite(s.mean[1])||!std::isfinite(s.mean[2])){e="trt stage2 invalid input contract";return{};}
 std::vector<uint8_t>b;if(!read(s.model_path,b,e))return{};std::string sha=rknn_detail::sha256_hex(b),key=sha+":"+std::to_string(s.in_h)+"x"+std::to_string(s.in_w)+":"+(s.bgr?"bgr":"rgb")+":"+std::to_string(s.scale);for(float v:s.mean)key+=":"+std::to_string(v);{std::lock_guard<std::mutex>l(cache_mu);if(auto i=cache.find(key);i!=cache.end())if(auto p=i->second.lock())return p;}
 std::lock_guard<std::mutex> load_lk(cache_load_mu);
 { std::lock_guard<std::mutex> l(cache_mu); if(auto i=cache.find(key);i!=cache.end()) if(auto p=i->second.lock()) return p; }
 // load_trt_engine performs the shared runner's exactly-one-input check.
 auto sh=load_trt_engine(s.model_path,e);if(!sh)return{};if(sh->model_h!=s.in_h||sh->model_w!=s.in_w){e="trt stage2 engine input shape does not match input_hw";return{};}auto m=std::make_shared<TrtStage2Model>();m->shared=std::move(sh);m->sha=sha;
 int input_count=0;
 for(int i=0;i<m->shared->engine->getNbIOTensors();++i){const char*n=m->shared->engine->getIOTensorName(i);auto mode=m->shared->engine->getTensorIOMode(n);auto t=m->shared->engine->getTensorDataType(n);auto d=m->shared->engine->getTensorShape(n);size_t cnt;const size_t elem_bytes=type_bytes(t);if(!volume(d,cnt)||!elem_bytes||cnt>SIZE_MAX/elem_bytes){e="trt stage2 tensor byte size overflows";return{};}for(int j=0;j<d.nbDims;++j)if(d.d[j]>INT32_MAX){e="trt stage2 tensor dimension exceeds int32";return{};}if(mode==nvinfer1::TensorIOMode::kINPUT){if(++input_count>1){e="trt stage2 expects exactly one input";return{};}m->input_name=n;m->in_type=t;m->in_bytes=cnt*elem_bytes;if(d.nbDims!=4||d.d[0]!=1||d.d[1]!=3||d.d[2]!=s.in_h||d.d[3]!=s.in_w||(t!=nvinfer1::DataType::kFLOAT&&t!=nvinfer1::DataType::kHALF)){e="trt stage2 input must be float/half NCHW [1,3,H,W]";return{};}}else{if(m->out_bytes){e="trt stage2 expects exactly one output";return{};}m->output_name=n;m->out_type=t;m->out_bytes=cnt*elem_bytes;for(int j=0;j<d.nbDims;++j)m->dims.push_back(d.d[j]);if(m->dims.size()==3&&m->dims[0]==1)m->dims.erase(m->dims.begin());else if(m->dims.size()==4&&m->dims[0]==1&&m->dims[2]==1){m->dims.erase(m->dims.begin());m->dims.erase(m->dims.begin()+1);}if(m->dims.size()!=2||m->dims[0]<=0||m->dims[1]<=0||(t!=nvinfer1::DataType::kFLOAT&&t!=nvinfer1::DataType::kHALF)){e="trt stage2 output must be float/half rank [C,T]";return{};}}}
 if(input_count!=1||!m->in_bytes||!m->out_bytes){e="trt stage2 requires exactly one input and one output";return{};}std::lock_guard<std::mutex>l(cache_mu);cache[key]=m;return m;
}
std::unique_ptr<Stage2Context> make_trt_stage2_context(const std::shared_ptr<TrtStage2Model>&m,const Stage2Spec&s,std::string&e){auto p=std::make_unique<TrtStage2>(m,s,e);return e.empty()?std::move(p):nullptr;}
}
