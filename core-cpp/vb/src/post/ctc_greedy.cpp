#include "vb/post.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdint>

namespace vb {
namespace {
float half_to_float(uint16_t h) {
    uint32_t s=(h>>15)&1, e=(h>>10)&31, f=h&1023, u;
    if (!e) {
        if (!f) u = s << 31;
        else { int shift = 0; while ((f & 0x400) == 0) { f <<= 1; ++shift; } f &= 0x3ff; u = (s<<31) | static_cast<uint32_t>(113-shift)<<23 | (f<<13); }
    } else if (e==31) u = s<<31 | 0x7f800000u | (f<<13);
    else u = (s<<31) | ((e + 112)<<23) | (f<<13);
    float x; std::memcpy(&x,&u,4); return x;
}
bool shape(const TensorView& v, int64_t& c, int64_t& t, CtcLayout l, std::string& e) {
    if (v.dims.size()!=2 || v.dims[0]<=0 || v.dims[1]<=0) { e="ctc: logits must be non-empty 2D"; return false; }
    c = l==CtcLayout::CT ? v.dims[0] : v.dims[1]; t = l==CtcLayout::CT ? v.dims[1] : v.dims[0];
    const uint64_t a = static_cast<uint64_t>(v.dims[0]);
    const uint64_t b = static_cast<uint64_t>(v.dims[1]);
    if (a > std::numeric_limits<size_t>::max() / b) { e="ctc: tensor dimensions overflow"; return false; }
    const size_t expected = static_cast<size_t>(a * b);
    if (v.count == 0 || v.count != expected) { e="ctc: tensor element count mismatch"; return false; }
    if (!v.raw_data && !v.data) { e="ctc: null data"; return false; }
    return true;
}
float value(const TensorView& v,size_t i) {
    const auto* p = v.raw_data ? v.raw_data : static_cast<const void*>(v.data);
    float x=0;
    if (v.dtype==0) x=static_cast<const float*>(p)[i];
    else if (v.dtype==1) x=(static_cast<double>(static_cast<const int8_t*>(p)[i])-static_cast<double>(v.zero_point))*static_cast<double>(v.scale);
    else if (v.dtype==2) x=(static_cast<double>(static_cast<const uint8_t*>(p)[i])-static_cast<double>(v.zero_point))*static_cast<double>(v.scale);
    else if (v.dtype==3) x=half_to_float(static_cast<const uint16_t*>(p)[i]);
    else return std::numeric_limits<float>::quiet_NaN();
    return x;
}
}
std::string ctc_greedy(const TensorView& v,CtcLayout l,const std::vector<std::string>& cs,float* mean,float* minc) {
    if (mean) *mean=-1; if (minc) *minc=-1; int64_t C,T; std::string err;
    if (!shape(v,C,T,l,err) || C != static_cast<int64_t>(cs.size())) return {};
    std::string out; std::vector<float> conf; int prev=-1;
    for(int64_t t=0;t<T;++t){
        float mx=-std::numeric_limits<float>::infinity();
        for(int64_t c=0;c<C;++c){ const uint64_t off = l==CtcLayout::CT ? static_cast<uint64_t>(c)*static_cast<uint64_t>(T)+static_cast<uint64_t>(t) : static_cast<uint64_t>(t)*static_cast<uint64_t>(C)+static_cast<uint64_t>(c); if (off >= v.count) return {}; float x=value(v,static_cast<size_t>(off)); if(!std::isfinite(x)) return {}; mx=std::max(mx,x); }
        float sum=0; for(int64_t c=0;c<C;++c){ const uint64_t off = l==CtcLayout::CT ? static_cast<uint64_t>(c)*static_cast<uint64_t>(T)+static_cast<uint64_t>(t) : static_cast<uint64_t>(t)*static_cast<uint64_t>(C)+static_cast<uint64_t>(c); sum += std::exp(value(v,static_cast<size_t>(off))-mx); }
        if(!std::isfinite(sum)||sum<=0) return {};
        int best=0; float bp=0;
        for(int64_t c=0;c<C;++c){ const uint64_t off = l==CtcLayout::CT ? static_cast<uint64_t>(c)*static_cast<uint64_t>(T)+static_cast<uint64_t>(t) : static_cast<uint64_t>(t)*static_cast<uint64_t>(C)+static_cast<uint64_t>(c); float p=std::exp(value(v,static_cast<size_t>(off))-mx)/sum; if(p>bp){bp=p;best=(int)c;} }
        if(best!=0 && best!=prev){ out += cs[best]; conf.push_back(bp); }
        else if(best!=0 && best==prev && !conf.empty()) conf.back()=std::max(conf.back(),bp);
        prev=best;
    }
    if(conf.empty()){ if(mean)*mean=0; if(minc)*minc=0; } else { float s=0,m=1; for(float x:conf){s+=x;m=std::min(m,x);} if(mean)*mean=s/conf.size(); if(minc)*minc=m; }
    return out;
}
}
