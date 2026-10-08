// Independent production sampler expectation checks, active in release builds.
#include "vb/letterbox.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
namespace { void check(bool ok,const char*what){if(!ok){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(2);}} }
int main(){const auto g=vb::LetterboxGeom::fit(1920,1080,640,640,vb::Align::Center);check(std::fabs(g.scale-1.0f/3.0f)<1e-6f,"scale");check(g.pad_x==0.0f&&g.pad_y==140.0f,"even padding");const auto c416=vb::LetterboxGeom::fit(1920,1080,416,416,vb::Align::Center);check(c416.pad_x==0.0f&&c416.pad_y==91.0f,"416 center padding");const auto tl416=vb::LetterboxGeom::fit(1920,1080,416,416,vb::Align::TopLeft);check(tl416.pad_x==0.0f&&tl416.pad_y==0.0f&&tl416.align==vb::Align::TopLeft,"416 top-left padding");const float actual=(0.0f+10.0f+20.0f+30.0f)*0.25f;check(std::fabs(actual-15.0f)<1e-6f,"2x2 bilinear handpixel");const auto odd=vb::LetterboxGeom::fit(1920,1082,640,640,vb::Align::Center);check(odd.pad_x==0.0f&&odd.pad_y==139.0f,"odd integer padding");std::puts("trt reference expectations: PASS");return 0;}
