#include "trt_stage2.h"
#include "vb/backend.h"
#include "vb/decoder.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
int main(int argc,char**argv){
 if(argc!=2){std::fprintf(stderr,"usage: trt_stage2_test ENGINE\n");return 2;}
 vb::Stage2Spec s; s.model_path=argv[1]; s.in_h=32; s.in_w=96; s.scale=1.0f/255.0f; s.mean[0]=1.0f; s.mean[1]=2.0f; s.mean[2]=3.0f;
 std::string e; auto m=vb::load_trt_stage2_model(s,e); if(!m){std::fprintf(stderr,"load: %s\n",e.c_str());return 3;}
 auto ctx=vb::make_trt_stage2_context(m,s,e); if(!ctx){std::fprintf(stderr,"context: %s\n",e.c_str());return 4;}
 std::vector<unsigned char> px(16*8*3); for(int y=0;y<8;++y)for(int x=0;x<16;++x){auto*p=&px[(y*16+x)*3];p[0]=static_cast<unsigned char>(x*7);p[1]=static_cast<unsigned char>(y*17);p[2]=static_cast<unsigned char>(x*7+y*3);}
 vb::FrameBuf f; f.w=16;f.h=8;f.stride=48;f.host=px.data();f.fmt=vb::PixFmt::BGR888;f.mem=vb::Mem::Host;
 vb::CropReq c[2]={{0,0,0.5f,1},{0.5f,0,1,1}}; vb::TensorView o[2];
 if(ctx->infer_crops(f,c,2,o,e)!=0){std::fprintf(stderr,"infer: %s\n",e.c_str());return 5;}
 // Public bound: reject n>16 without walking an unbounded output array.
 vb::CropReq too_many[17]{}; vb::TensorView one{};
 if(ctx->infer_crops(f,too_many,17,&one,e)==0){std::fprintf(stderr,"accepted >16 crops\n");return 5;}
 // Reject dimensions before any signed stride multiplication (INT_MAX boundary).
 if(ctx->infer_rgb(px.data(),INT32_MAX,1,INT32_MAX,&one,e)==0){std::fprintf(stderr,"accepted overflowing infer_rgb dimensions\n");return 5;}
 if(!o[0].data||!o[1].data||o[0].data==o[1].data||o[0].count!=o[1].count){std::fprintf(stderr,"scratch/output ownership failure\n");return 6;}
 bool different=false; for(size_t i=0;i<o[0].count;++i){if(!std::isfinite(o[0].data[i])||!std::isfinite(o[1].data[i]))return 7;if(o[0].data[i]!=o[1].data[i])different=true;} if(!different){std::fprintf(stderr,"different crops produced identical outputs\n");return 8;}
 std::puts("trt stage2 n=2 independent scratch: PASS"); return 0;
}
