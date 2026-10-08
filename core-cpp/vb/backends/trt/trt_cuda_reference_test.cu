// Hand-calculated production CUDA sampler checks, active in Release.
#include "preprocess_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
void check(bool ok,const char* what) {
 if(!ok){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(2);}
}
void equal(float got,float expected,float tol,const char* what) {
 if(!std::isfinite(got)||std::fabs(got-expected)>tol){
  std::fprintf(stderr,"FAIL: %s got=%g expected=%g\n",what,got,expected);std::exit(2);
 }
}
}
int main(){
 // BGR rows, two pixels followed by six unused stride bytes.
 const unsigned char host[24]={0,10,20,30,40,50,0,0,0,0,0,0,60,70,80,90,100,110,0,0,0,0,0,0};
 unsigned char* src=nullptr;float* out=nullptr;__half* hout=nullptr;cudaStream_t st=nullptr;
 check(cudaMalloc((void**)&src,sizeof(host))==cudaSuccess,"source alloc");
 check(cudaMalloc((void**)&out,48*sizeof(float))==cudaSuccess,"float alloc");
 check(cudaMalloc((void**)&hout,48*sizeof(__half))==cudaSuccess,"half alloc");
 check(cudaMemcpy(src,host,sizeof(host),cudaMemcpyHostToDevice)==cudaSuccess,"H2D");
 check(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking)==cudaSuccess,"stream");
 // Production-sized tensor oracle: 1920x1080 -> 416. The source is a
 // nontrivial per-channel gradient; the checked pixel is away from edges so
 // its bilinear value is hand-computable, while padding rows are checked in
 // full for both top-left and centered alignment.
 const int lw=1920, lh=1080, mw=416, mh=416, lstride=lw*3;
 std::vector<unsigned char> large_host(static_cast<size_t>(lh)*lstride);
 for (int y=0; y<lh; ++y) for (int x=0; x<lw; ++x) {
  auto* q=&large_host[static_cast<size_t>(y)*lstride+x*3];
  q[0]=static_cast<unsigned char>((x+y+10)%256);
  q[1]=static_cast<unsigned char>((2*x+y+20)%256);
  q[2]=static_cast<unsigned char>((3*x+2*y+30)%256);
 }
 unsigned char* large_src=nullptr; float* large_out=nullptr; __half* large_half=nullptr;
 check(cudaMalloc((void**)&large_src,large_host.size())==cudaSuccess,"large source alloc");
 check(cudaMalloc((void**)&large_out,static_cast<size_t>(mw)*mh*3*sizeof(float))==cudaSuccess,"large output alloc");
 check(cudaMalloc((void**)&large_half,static_cast<size_t>(mw)*mh*3*sizeof(__half))==cudaSuccess,"large half alloc");
 check(cudaMemcpy(large_src,large_host.data(),large_host.size(),cudaMemcpyHostToDevice)==cudaSuccess,"large H2D");
 const float lscale=416.0f/1920.0f, ldivide=2.0f;
 const int rounded_h=234, center_py=91;
 const int sample_x=10;
 const float sx=(sample_x+0.5f)/lscale-0.5f;
 const float top_left_sy=(0.5f)/lscale-0.5f;
 const float expect[3]={(sx+top_left_sy+10.0f)/ldivide,
                        (2.0f*sx+top_left_sy+20.0f)/ldivide,
                        (3.0f*sx+2.0f*top_left_sy+30.0f)/ldivide};
 for (int pass=0; pass<2; ++pass) {
  const float py=pass==0?0.0f:static_cast<float>(center_py);
  check(vb::launchCudaPreprocess(large_src,lw,lh,lstride,large_out,mw,mh,lscale,0,py,114,ldivide,false,false,vb::PreprocessOutputType::Float32,st),"large FP32 launch");
  float large_got[3];
  const size_t sample_index=static_cast<size_t>(pass==0?0:center_py)*mw+sample_x;
  for (int c=0;c<3;++c) check(cudaMemcpyAsync(large_got+c,large_out+static_cast<size_t>(c)*mw*mh+sample_index,sizeof(float),cudaMemcpyDeviceToHost,st)==cudaSuccess,"large sample D2H");
  check(cudaStreamSynchronize(st)==cudaSuccess,"large FP32 sync");
  for (int c=0;c<3;++c) equal(large_got[c],expect[c],2e-4f,"large hand bilinear");
  check(vb::launchCudaPreprocess(large_src,lw,lh,lstride,large_half,mw,mh,lscale,0,py,114,ldivide,false,false,vb::PreprocessOutputType::Float16,st),"large FP16 launch");
  __half large_hgot[3];
  for (int c=0;c<3;++c) check(cudaMemcpyAsync(large_hgot+c,large_half+static_cast<size_t>(c)*mw*mh+sample_index,sizeof(__half),cudaMemcpyDeviceToHost,st)==cudaSuccess,"large half D2H");
  check(cudaStreamSynchronize(st)==cudaSuccess,"large FP16 sync");
  for (int c=0;c<3;++c) equal(__half2float(large_hgot[c]),expect[c],2e-2f,"large half hand bilinear");
  for (int y=0;y<mh;++y) {
   const bool padding=pass==0?(y>=rounded_h):(y<center_py||y>=center_py+rounded_h);
   if (!padding) continue;
   const size_t row=static_cast<size_t>(y)*mw;
   float pad_probe[3];
   check(cudaMemcpyAsync(pad_probe,large_out+row,3*sizeof(float),cudaMemcpyDeviceToHost,st)==cudaSuccess,"large pad D2H");
   check(cudaStreamSynchronize(st)==cudaSuccess,"large pad sync");
   for (int c=0;c<3;++c) equal(pad_probe[c],57.0f,1e-5f,"large full padding row");
  }
 }
 check(cudaFree(large_half)==cudaSuccess,"large half free");
 check(cudaFree(large_out)==cudaSuccess,"large output free");
 check(cudaFree(large_src)==cudaSuccess,"large source free");
 // Non-integer resize boundary: 7x3 -> 5x5 rounds to a 5x2 source
 // rectangle. Verify every padding row and at least one real source row for
 // both alignments; this catches float-boundary overrun in the CUDA kernel.
 const unsigned char small_host[63] = {
   10,20,30, 11,21,31, 12,22,32, 13,23,33, 14,24,34, 15,25,35, 16,26,36,
   20,30,40, 21,31,41, 22,32,42, 23,33,43, 24,34,44, 25,35,45, 26,36,46,
   30,40,50, 31,41,51, 32,42,52, 33,43,53, 34,44,54, 35,45,55, 36,46,56
 };
 unsigned char* small_src=nullptr; float* small_out=nullptr;
 check(cudaMalloc((void**)&small_src,sizeof(small_host))==cudaSuccess,"small source alloc");
 check(cudaMalloc((void**)&small_out,75*sizeof(float))==cudaSuccess,"small output alloc");
 check(cudaMemcpy(small_src,small_host,sizeof(small_host),cudaMemcpyHostToDevice)==cudaSuccess,"small H2D");
 const float small_scale=5.0f/7.0f;
 for (int pass=0; pass<2; ++pass) {
  const float py = pass == 0 ? 0.0f : 1.0f;
  check(vb::launchCudaPreprocess(small_src,7,3,21,small_out,5,5,small_scale,0,py,114,1,false,false,vb::PreprocessOutputType::Float32,st),"small boundary launch");
  float small_got[75];
  check(cudaMemcpyAsync(small_got,small_out,sizeof(small_got),cudaMemcpyDeviceToHost,st)==cudaSuccess,"small D2H");
  check(cudaStreamSynchronize(st)==cudaSuccess,"small sync");
  for (int y=0; y<5; ++y) {
   const bool padding = pass == 0 ? (y >= 2) : (y == 0 || y >= 3);
   for (int c=0; c<3; ++c) for (int x=0; x<5; ++x) {
    const float v=small_got[c*25+y*5+x];
    if (padding) equal(v,114.0f,1e-5f,"small integer padding");
    else check(std::fabs(v-114.0f)>1e-3f,"small source row is not padding");
   }
  }
 }
 check(cudaFree(small_out)==cudaSuccess,"small output free");
 check(cudaFree(small_src)==cudaSuccess,"small source free");
 // Half-pixel positions -0.25,0.25,0.75,1.25 clamp to the source edge.
 const float red[16]={20,27.5,42.5,50,35,42.5,57.5,65,65,72.5,87.5,95,80,87.5,102.5,110};
 float expected[48],got[48];
 for(int c=0;c<3;++c)for(int i=0;i<16;++i)expected[c*16+i]=(red[i]-10*c)/255.f;
 check(vb::launchCudaPreprocess(src,2,2,12,out,4,4,2,0,0,114,255,true,false,vb::PreprocessOutputType::Float32,st),"RGB resize");
 check(cudaMemcpyAsync(got,out,sizeof(got),cudaMemcpyDeviceToHost,st)==cudaSuccess,"RGB D2H");
 check(cudaStreamSynchronize(st)==cudaSuccess,"RGB sync");
 for(int i=0;i<48;++i)equal(got[i],expected[i],2e-4f,"RGB bilinear /255");
 check(vb::launchCudaPreprocess(src,2,2,12,hout,4,4,2,0,0,114,255,true,false,vb::PreprocessOutputType::Float16,st),"half launch");
 __half hgot[48];
 check(cudaMemcpyAsync(hgot,hout,sizeof(hgot),cudaMemcpyDeviceToHost,st)==cudaSuccess,"half D2H");
 check(cudaStreamSynchronize(st)==cudaSuccess,"half sync");
 for(int i=0;i<48;++i)equal(__half2float(hgot[i]),expected[i],3e-4f,"half bilinear /255");
 // Rectangle x=[1,3): one pad pixel on each side, not two on the left.
 const float pad[24]={114,20,50,114,114,80,110,114,114,10,40,114,114,70,100,114,114,0,30,114,114,60,90,114};
 check(vb::launchCudaPreprocess(src,2,2,12,out,4,2,1,1,0,114,1,true,false,vb::PreprocessOutputType::Float32,st),"pad launch");
 check(cudaMemcpyAsync(got,out,sizeof(pad),cudaMemcpyDeviceToHost,st)==cudaSuccess,"pad D2H");
 check(cudaStreamSynchronize(st)==cudaSuccess,"pad sync");
 for(int i=0;i<24;++i)equal(got[i],pad[i],1e-5f,"pad and source stride");
 const float bgr[12]={0,30,60,90,10,40,70,100,20,50,80,110};
 check(vb::launchCudaPreprocess(src,2,2,12,out,2,2,1,0,0,114,1,false,false,vb::PreprocessOutputType::Float32,st),"BGR launch");
 check(cudaMemcpyAsync(got,out,sizeof(bgr),cudaMemcpyDeviceToHost,st)==cudaSuccess,"BGR D2H");
 check(cudaStreamSynchronize(st)==cudaSuccess,"BGR sync");
 for(int i=0;i<12;++i)equal(got[i],bgr[i],1e-5f,"BGR /1");
 // Stage2 crop path: source BGR, RGB model order, nonzero mean uses value*scale-mean.
 const float mean[3]={1.0f,2.0f,3.0f};
 check(vb::launchCudaCropPreprocess(src,2,2,12,out,2,2,0,0,2,2,false,true,mean,0.5f,vb::PreprocessOutputType::Float32,st),"crop launch");
 check(cudaMemcpyAsync(got,out,sizeof(float)*12,cudaMemcpyDeviceToHost,st)==cudaSuccess,"crop D2H");
 check(cudaStreamSynchronize(st)==cudaSuccess,"crop sync");
 const float crop_expected[12]={9.0f,24.0f,39.0f,54.0f,3.0f,18.0f,33.0f,48.0f,-3.0f,12.0f,27.0f,42.0f};
 for(int i=0;i<12;++i) equal(got[i],crop_expected[i],1e-5f,"crop mean/channel contract");
 check(!vb::launchCudaCropPreprocess(src,2,2,12,out,2,2,1,1,1,2,false,true,mean,1.0f,vb::PreprocessOutputType::Float32,st),"reject empty crop");
 check(!vb::launchCudaPreprocess(src,2,2,5,out,2,2,1,0,0,114,1,true,false,vb::PreprocessOutputType::Float32,st),"reject short stride");
 check(!vb::launchCudaPreprocess(src,2,2,12,out,2,2,1,0,0,114,0,true,false,vb::PreprocessOutputType::Float32,st),"reject divide zero");
 check(cudaStreamDestroy(st)==cudaSuccess,"destroy stream");check(cudaFree(hout)==cudaSuccess,"free half");check(cudaFree(out)==cudaSuccess,"free float");check(cudaFree(src)==cudaSuccess,"free source");
 std::puts("trt CUDA production sampler: PASS (RGB/BGR, FP32/FP16, resize, pad, stride)");
 return 0;
}
