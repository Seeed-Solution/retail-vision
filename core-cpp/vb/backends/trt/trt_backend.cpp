// Jetson TensorRT backend (BASE-1 M2.2). No OpenCV or parking business logic.
#include "trt_backend.h"
#include "trt_runner.h"
#include "trt_stage2.h"
#include "trt_source.h"
#include "../rknn/model_sha256.h"
#include "vb/decoder.h"
#include "vb/json.h"
#include "vb/post.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <limits>

namespace vb { namespace {
bool sha_ok(const std::string& s){if(s.empty())return true;if(s.size()!=64)return false;for(char c:s)if(!std::isxdigit(static_cast<unsigned char>(c)))return false;return true;}
std::string lower(std::string s){for(char&c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s;}
class TrtContext final : public InferenceContext {
public: TrtContext(std::shared_ptr<TrtShared> shared,std::unique_ptr<Decoder> decoder,InputSpec input):shared_(std::move(shared)),runner_(shared_),decoder_(std::move(decoder)),input_(input){}
 int infer(const FrameBuf* const* frames,size_t n,float score,float nms_th,DetectionResult* out,std::string& err) override {if(n!=1){err="trt backend max_batch is 1";return -1;} views_.clear();storage_.clear();if(!runner_.infer(*frames[0],input_,views_,storage_,out[0],err))return -1;out[0].dets.clear();out[0].kpts.clear();if(!decoder_){if(views_.size()!=1){err="trt backend requires backend.decoder for multiple outputs";return -1;}const auto&v=views_[0];if(v.dims.size()!=2){err="trt backend: YOLOX output must be rank 2 after batch";return -1;}int a=0,c=0;bool t=false;if(v.dims[1]>=6&&v.dims[1]<=200){a=v.dims[0];c=static_cast<int>(v.dims[1])-5;}else if(v.dims[0]>=6&&v.dims[0]<=200){a=v.dims[1];c=static_cast<int>(v.dims[0])-5;t=true;}if(a<=0||c<=0){err="trt backend: unexpected YOLOX output shape";return -1;}if(t){tmp_.resize(static_cast<size_t>(a)*(c+5));for(int i=0;i<a;++i)for(int j=0;j<c+5;++j)tmp_[static_cast<size_t>(i)*(c+5)+j]=v.data[static_cast<size_t>(j)*a+i];yolox_decode(tmp_.data(),a,c,shared_->model_w,shared_->model_h,score,out[0].dets);}else yolox_decode(v.data,a,c,shared_->model_w,shared_->model_h,score,out[0].dets);nms(out[0].dets,out[0].kpts,nms_th,true);}else if(!decoder_->decode(views_.data(),views_.size(),shared_->model_w,shared_->model_h,score,nms_th,out[0],err))return -1;return 0;}
private: std::shared_ptr<TrtShared> shared_; TrtRunner runner_; std::unique_ptr<Decoder> decoder_; InputSpec input_{}; std::vector<TensorView> views_; std::vector<std::vector<float>> storage_; std::vector<float> tmp_;
};
class TrtBackend final : public Backend {
public: explicit TrtBackend(const std::string& text,std::string& err){Json j;try{j=text.empty()?Json::object():json_parse(text);}catch(const std::exception&e){err=std::string("trt backend json: ")+e.what();return;}if(!j.contains("model_path")){err="trt backend requires backend.model_path";return;}path_=j.at("model_path").get<std::string>();std::string want=j.value("model_sha256",std::string());if(!sha_ok(want)){err="backend.model_sha256 must be 64 hex characters";return;}std::ifstream f(path_,std::ios::binary);std::vector<uint8_t>b((std::istreambuf_iterator<char>(f)),{});if(b.empty()){err="trt backend: model file is empty or unreadable";return;}sha_=rknn_detail::sha256_hex(b);if(!want.empty()&&lower(sha_)!=lower(want)){err="model sha256 mismatch: expected "+want+", got "+sha_;return;}shared_=load_trt_engine(path_,err);if(!shared_)return;auto dj=j.find("decoder");if(dj!=j.end()&&dj->is_object()&&!dj->empty()){decoder_json_=dj->dump();}std::string typ="yolox";if(dj!=j.end()&&dj->is_object())typ=dj->value("type",std::string("yolox"));input_=InputSpec::default_for_decoder(typ);auto aj=j.find("align");if(aj!=j.end()){if(!aj->is_string()){err="backend.align must be center or top_left";return;}try{input_.align=align_from_string(aj->get<std::string>());}catch(const std::invalid_argument&){err="backend.align must be center or top_left";return;}}auto ij=j.find("input");if(ij!=j.end()&&ij->is_object()){std::string co=ij->value("color_order",std::string(input_.color_order==ColorOrder::RGB?"rgb":"bgr"));if(co=="rgb")input_.color_order=ColorOrder::RGB;else if(co=="bgr")input_.color_order=ColorOrder::BGR;else{err="backend.input.color_order must be bgr or rgb";return;}if(ij->contains("divide")){double d=ij->at("divide").get<double>();if(!(d>0)){err="backend.input.divide must be > 0";return;}input_.divide=static_cast<float>(d);}}max_contexts_=j.value("max_contexts",1);if(max_contexts_<1||max_contexts_>2){err="trt backend max_contexts must be 1..2";return;}ok_=true;}
 bool ok()const{return ok_;}const char*name()const override{return "trt";}Caps caps()const override{Caps c;c.max_contexts=max_contexts_;c.max_batch=1;c.exclusive_device=false;return c;}std::pair<int,int>model_hw()const override{return{shared_->model_w,shared_->model_h};}std::string model_sha256()const override{return sha_;}
 std::unique_ptr<FrameSource>create_source(const StreamSpec&s,std::string&err)override{return make_trt_source(s,err);}std::unique_ptr<InferenceContext>create_context(int i,std::string&err)override{if(i<0||i>=max_contexts_){err="trt backend: context index outside configured range";return nullptr;}std::unique_ptr<Decoder>d;if(!decoder_json_.empty()){d=make_decoder(decoder_json_,err);if(!d)return nullptr;}return std::make_unique<TrtContext>(shared_,std::move(d),input_);}std::unique_ptr<Stage2Context>create_stage2(const Stage2Spec&s,std::string&err)override{auto m=load_trt_stage2_model(s,err);return m?make_trt_stage2_context(m,s,err):nullptr;}
private:std::string path_,sha_,decoder_json_;std::shared_ptr<TrtShared>shared_;InputSpec input_{};int max_contexts_=1;bool ok_=false;
};
} 
std::unique_ptr<Backend>make_trt_backend(const std::string&j,std::string&e){e.clear();auto b=std::make_unique<TrtBackend>(j,e);if(!e.empty()||!b->ok())return nullptr;return b;}
}
