#include "cvi_backend.h"
#include "cvi_source.h"
#include "cvi_stage2.h"
#include "cvi_resize.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <vector>
#include "cviruntime.h"
#include "cvi_sha256.h"
#include "vb/decoder.h"
#include "vb/json.h"
#include "vb/letterbox.h"
#include "vb/rate_crop.h"
#include "vb/post.h"
namespace vb { namespace {
struct Model { std::vector<uint8_t> bytes; CVI_MODEL_HANDLE h=nullptr; CVI_TENSOR* in=nullptr; CVI_TENSOR* out=nullptr; int ni=0,no=0; Model()=default; ~Model(){if(h) CVI_NN_CleanupModel(h);} Model(const Model&)=delete; };
bool load(const std::string& p,std::vector<uint8_t>& b,std::string&e){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f){e="cvi model open failed: "+p;return false;}auto n=f.tellg();if(n<=0||static_cast<uint64_t>(n)>512u*1024*1024){e="cvi model size invalid";return false;}b.resize(static_cast<size_t>(n));f.seekg(0);if(!f.read(reinterpret_cast<char*>(b.data()),n)){e="cvi model read failed";return false;}return true;}
bool open_model(const std::vector<uint8_t>& b,std::shared_ptr<Model>&m,std::string&e){m=std::make_shared<Model>();if(b.empty()||b.size()>UINT32_MAX){e="CVI model bytes invalid";return false;}m->bytes=b;if(CVI_NN_RegisterModelFromBuffer(reinterpret_cast<const int8_t*>(m->bytes.data()),static_cast<uint32_t>(m->bytes.size()),&m->h)!=0||!m->h){e="CVI_NN_RegisterModelFromBuffer failed";return false;}if(CVI_NN_GetInputOutputTensors(m->h,&m->in,&m->ni,&m->out,&m->no)!=0||m->ni!=1||m->no<1||!m->in||!m->out){e="cvi model requires one input and at least one output";return false;}return true;}
bool shape(const CVI_TENSOR&t,std::initializer_list<int> want){if(t.shape.dim_size!=want.size())return false;size_t i=0;for(int x:want)if(t.shape.dim[i++]!=x)return false;return true;}
void prep(const FrameBuf& f,const LetterboxGeom& g,std::vector<float>& dst) {
  const int mw=g.model_w, mh=g.model_h;
  dst.assign(static_cast<size_t>(3)*mw*mh, 114.0f);
  const int rw=std::max(1,static_cast<int>(f.w*g.scale));
  const int rh=std::max(1,static_cast<int>(f.h*g.scale));
  std::vector<uint8_t> resized;
  if (!cvi_resize_u8_inter_linear(f.host, f.w, f.h, f.stride, 0.0f, 0.0f,
                                  static_cast<float>(f.w), static_cast<float>(f.h),
                                  rw, rh, resized)) return;
  const size_t plane=static_cast<size_t>(mw)*mh;
  for (int y=0; y<rh && y+static_cast<int>(g.pad_y)<mh; ++y)
    for (int x=0; x<rw && x+static_cast<int>(g.pad_x)<mw; ++x) {
      const size_t si=(static_cast<size_t>(y)*rw+x)*3;
      const size_t o=static_cast<size_t>(y+static_cast<int>(g.pad_y))*mw+
                     x+static_cast<int>(g.pad_x);
      const int r=f.fmt==PixFmt::BGR888?2:0;
      const int b=f.fmt==PixFmt::BGR888?0:2;
      dst[o]=resized[si+r]; dst[plane+o]=resized[si+1]; dst[2*plane+o]=resized[si+b];
    }
}
class Ctx final:public InferenceContext{public:Ctx(std::shared_ptr<Model>m,Decoder*d,int w,int h,std::mutex&mu):m_(std::move(m)),d_(d),w_(w),h_(h),mu_(mu){}int infer(const FrameBuf*const*f,size_t n,float score,float nms_th,DetectionResult*out,std::string&e)override{if(n!=1||!f||!f[0]||!out){e="cvi max_batch is 1";return -1;}const FrameBuf&fr=*f[0];size_t frame_bytes=0;if(fr.mem!=Mem::Host||!fr.host||(fr.fmt!=PixFmt::BGR888&&fr.fmt!=PixFmt::RGB888)||!host_rgb_layout_bytes(fr.w,fr.h,fr.stride,frame_bytes)){e="cvi requires bounded host RGB/BGR frame";return -1;}std::lock_guard<std::mutex>lk(mu_);auto&in=*m_->in;auto&ot=*m_->out;if(in.fmt!=CVI_FMT_FP32||!shape(in,{1,3,416,416})||in.count!=3u*416u*416u||in.mem_size<in.count*sizeof(float)){e="cvi detector input contract mismatch";return -1;}if(ot.fmt!=CVI_FMT_FP32||!shape(ot,{1,3549,6})||ot.count!=3549u*6u||ot.mem_size<ot.count*sizeof(float)||!ot.name||std::string(ot.name)!="output_Transpose_f32"){e="cvi detector output contract mismatch";return -1;}prep(fr,LetterboxGeom::fit(fr.w,fr.h,w_,h_,Align::TopLeft),buf_);if(CVI_NN_SetTensorPtr(&in,buf_.data())!=0){e="CVI_NN_SetTensorPtr(input) failed";return -1;}int rc=CVI_NN_Forward(m_->h,m_->in,m_->ni,m_->out,m_->no);if(rc!=0){e="CVI_NN_Forward rc="+std::to_string(rc);return -1;}float*p=static_cast<float*>(CVI_NN_TensorPtr(&ot));if(!p || !std::all_of(p,p+ot.count,[](float x){return std::isfinite(x);})){e="CVI output pointer/nonfinite output";return -1;}TensorView v;v.data=p;v.raw_data=p;v.count=ot.count;v.dtype=0;v.name=ot.name?ot.name:"";v.scale=ot.qscale;v.zero_point=ot.zero_point;v.dims={3549,6};out->geom=LetterboxGeom::fit(fr.w,fr.h,w_,h_,Align::TopLeft);if(d_){if(!d_->decode(&v,1,w_,h_,score,nms_th,*out,e))return -1;}else{yolox_decode(p,3549,1,w_,h_,score,out->dets);vb::nms(out->dets,out->kpts,nms_th,true);}return 0;}private:std::shared_ptr<Model>m_;Decoder*d_;int w_,h_;std::mutex&mu_;std::vector<float>buf_;};
class Backend final:public vb::Backend{public:explicit Backend(const std::string&js,std::string&e){try{Json j=json_parse(js.empty()?"{}":js);if(!j.contains("model_path")){e="cvi backend requires model_path";return;}path_=j.at("model_path").get<std::string>();std::vector<uint8_t>b;if(!load(path_,b,e))return;sha_=cvi_detail::sha256_hex(b);if(j.contains("model_sha256")&&j.at("model_sha256").get<std::string>()!=sha_){e="cvi model_sha256 mismatch";return;}if(!open_model(b,m_,e))return;if(!shape(*m_->in,{1,3,416,416})||!shape(*m_->out,{1,3549,6})||m_->in->fmt!=CVI_FMT_FP32||m_->out->fmt!=CVI_FMT_FP32){e="cvi detector model shape/dtype mismatch";return;}auto d=j.find("decoder");if(d!=j.end())dec_=make_decoder(d->is_object()?d->dump():d->get<std::string>(),e);ok_=e.empty();}catch(const std::exception&x){e=std::string("cvi backend json: ")+x.what();}}const char*name()const override{return "cvi";}Caps caps()const override{Caps c;c.max_contexts=1;c.max_batch=1;c.exclusive_device=true;c.keypoints=dec_?dec_->keypoints():0;return c;}std::pair<int,int>model_hw()const override{return{416,416};}std::string model_sha256()const override{return sha_;}std::unique_ptr<FrameSource>create_source(const StreamSpec&s,std::string&e)override{return make_cvi_source(s,e);}std::unique_ptr<InferenceContext>create_context(int,std::string&e)override{if(!ok_){e="cvi backend is not ready";return{};}return std::make_unique<Ctx>(m_,dec_.get(),416,416,mu_);}std::unique_ptr<Stage2Context>create_stage2(const Stage2Spec&s,std::string&e)override{return make_cvi_stage2(s,e,&mu_);}private:std::string path_,sha_;std::shared_ptr<Model>m_;std::unique_ptr<Decoder>dec_;std::mutex mu_;bool ok_=false;}; std::unique_ptr<vb::Backend> make_impl(const std::string&j,std::string&e){auto p=std::make_unique<Backend>(j,e);return e.empty()?std::move(p):nullptr;} }
std::unique_ptr<vb::Backend>make_cvi_backend(const std::string&j,std::string&e){return make_impl(j,e);}
}
