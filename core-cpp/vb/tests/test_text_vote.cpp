#include "check.h"
#include "vb/analyzer.h"
#include "vb/post.h"
#include "vb/text_match.h"
#include <cmath>
#include <vector>
#include <cstdint>
#include <limits>
using namespace vb;
static Stage2Read read(uint32_t id, const char* text, float mean, float minc, uint64_t seq, float x=0.1f) {
    Stage2Read r; r.track_id=id; r.text=text; r.mean_conf=mean; r.min_char_conf=minc; r.seq=seq;
    r.bbox[0]=x; r.bbox[1]=.2f; r.bbox[2]=.8f; r.bbox[3]=.9f; return r;
}
int main() {
    std::string e; auto a=create_analyzer("text_vote",e); CHECK(a!=nullptr); CHECK(!a->configure(R"({"source":"stage2"})",e));
    a->set_stage2_available(true); CHECK(a->configure(R"({"source":"stage2","window":3,"min_agree":2})",e));
    std::vector<Stage2Read> rs; std::vector<AnalyzerEvent> out; FrameMeta m{}; m.reads=&rs;
    for (uint64_t i=1;i<=2;++i) { rs={read(1,"AB",.9f,.8f,i)};m.seq=i;m.t_mono_s=i;a->on_frame(m,{},nullptr,out); }
    CHECK(out.size()==1); CHECK(out[0].type=="text_read"); CHECK(Json::parse(out[0].fields_json).at("votes")==2);
    std::string old=e; CHECK(!a->configure(R"({"source":"stage2","window":"bad"})",e)); CHECK(e.find("window must be integer")!=std::string::npos);
    CHECK(!a->configure(R"({"source":"stage2","window":3.9})",e));
    CHECK(!a->configure(R"({"source":"stage2","window":4294967299,"min_agree":2})",e));
    CHECK(!a->configure(R"({"source":"stage2","window":3,"min_agree":4294967298})",e));
    CHECK(!a->configure(R"({"source":"stage2","window":3,"min_agree":2,"flush_min_agree":4294967297})",e));
    CHECK(!a->configure(R"({"source":"stage2","window":3,"min_agree":2,"min_conf":1.0000000001})",e));
    CHECK(!a->configure(R"({"source":"stage2","window":3,"min_agree":2,"accept":[[{"set":"A","min":4294967297,"max":4294967297}]]})",e));
    CHECK(!a->configure(R"({"source":"stage2","window":18446744073709551615,"min_agree":2})",e));
    auto fresh = [&]() { auto x=create_analyzer("text_vote",e); CHECK(x!=nullptr); x->set_stage2_available(true); return x; };
    auto pending=fresh(); CHECK(pending->configure(R"({"source":"stage2","window":3,"min_agree":2})",e)); out.clear(); rs={read(8,"A",.9f,.8f,1)};m.seq=1;m.t_mono_s=1;pending->on_frame(m,{},nullptr,out);CHECK(!pending->configure(R"({"source":"stage2","window":4294967299,"min_agree":2})",e));rs={read(8,"A",.9f,.8f,2)};m.seq=2;m.t_mono_s=2;pending->on_frame(m,{},nullptr,out);CHECK(out.size()==1);
    auto b=fresh(); CHECK(b->configure(R"({"source":"stage2","window":4,"min_agree":4,"flush_min_agree":2})",e)); out.clear();
    for(uint64_t i=1;i<=6;++i){rs={read(2,(i==1||i==2)?"A":(i==3||i==4)?"B":(i==5?"A":"B"),.9f,.8f,i)};m.seq=i;m.t_mono_s=i;b->on_frame(m,{},nullptr,out);} b->on_track_removed(2,7,out); CHECK(out.size()==1); CHECK(Json::parse(out[0].fields_json).at("text")=="B"); CHECK(Json::parse(out[0].fields_json).at("votes")==3); CHECK(Json::parse(out[0].fields_json).at("frames")==4); CHECK(Json::parse(out[0].fields_json).at("flushed")==true);
    auto c=fresh(); CHECK(c->configure(R"({"source":"stage2","window":2,"min_agree":2,"flush_min_agree":0})",e)); out.clear(); rs={read(3,"A",.9f,.8f,1)};m.seq=1;m.t_mono_s=1;c->on_frame(m,{},nullptr,out);c->on_track_removed(3,2,out);CHECK(out.empty());
    auto d=fresh(); CHECK(d->configure(R"({"source":"stage2","window":2,"min_agree":1,"dedup_s":30})",e)); out.clear(); rs={read(4,"A",.9f,.8f,1)};m.seq=1;m.t_mono_s=1;d->on_frame(m,{},nullptr,out);rs={read(5,"A",.9f,.8f,2)};m.seq=2;m.t_mono_s=3;d->on_frame(m,{},nullptr,out);rs={read(6,"A",.9f,.8f,3)};m.seq=3;m.t_mono_s=31;d->on_frame(m,{},nullptr,out);CHECK(out.size()==2);CHECK(d->text_vote_dedup_count()==1);
    auto bad_an=fresh(); CHECK(bad_an->configure(R"({"source":"stage2","window":2,"min_agree":1})",e)); out.clear();
    auto invalid_utf8=read(7,"\xC3\x28",.9f,.8f,1);
    auto nan_mean=read(7,"A",.9f,.8f,2); nan_mean.mean_conf=std::numeric_limits<float>::quiet_NaN();
    auto inf_min=read(7,"A",.9f,.8f,3); inf_min.min_char_conf=std::numeric_limits<float>::infinity();
    auto zero_width=read(7,"A",.9f,.8f,4); zero_width.bbox[2]=zero_width.bbox[0];
    auto nan_bbox=read(7,"A",.9f,.8f,5); nan_bbox.bbox[0]=std::numeric_limits<float>::quiet_NaN();
    for(const auto& invalid : {invalid_utf8,nan_mean,inf_min,zero_width,nan_bbox}) {
        rs={invalid}; m.seq=invalid.seq; m.t_mono_s=invalid.seq;
        bad_an->on_frame(m,{},nullptr,out); CHECK(out.empty());
    }
    auto best=fresh(); CHECK(best->configure(R"({"source":"stage2","window":3,"min_agree":3})",e)); out.clear();
    for(const auto& candidate : {read(11,"A",.9f,.7f,20,.1f),read(11,"A",.7f,.8f,30,.2f),read(11,"A",.6f,.8f,10,.3f)}) {
        rs={candidate}; m.seq=100; m.t_mono_s+=1; best->on_frame(m,{},nullptr,out);
    }
    CHECK(out.size()==1); auto selected=Json::parse(out[0].fields_json);
    CHECK(selected.at("best_seq")==10); CHECK_NEAR(selected.at("min_char_conf").get<double>(),.8,1e-6);
    CHECK_NEAR(selected.at("bbox").at(0).get<double>(),.3,1e-6); CHECK_NEAR(selected.at("conf").get<double>(),2.2/3,1e-6);
    CHECK(a->configure(R"({"source":"stage2","window":3,"min_agree":2,"accept":[[{"set":"A","min":0,"max":16}]]})",e));
    CHECK(!a->configure(R"({"source":"stage2","window":3,"min_agree":2,"accept":[[{"set":"A","min":"0","max":1}]]})",e));
    CHECK(text_matches("京A12345",{})); std::vector<TextPattern> p; CHECK(parse_text_patterns(Json::parse(R"([[{"set":"京","min":1,"max":1}]])"),p,e)); CHECK(text_matches("京",p)); CHECK(!text_matches("A",p));
    float mean=0,minc=0; float x[6]={0, std::log(3.0f), 0, std::log(3.0f), 0, std::log(3.0f)}; TensorView tv{x,6,{2,3},""}; auto s=ctc_greedy(tv,CtcLayout::CT,{"","A"},&mean,&minc); CHECK(s=="AA"); CHECK_NEAR(mean,.75,1e-6); CHECK_NEAR(minc,.75,1e-6);
    float tc[6]={0,std::log(3.0f),std::log(3.0f),0,0,std::log(3.0f)}; TensorView tvtc{tc,6,{3,2},""}; CHECK(ctc_greedy(tvtc,CtcLayout::TC,{"","A"},&mean,&minc)=="AA");
    uint8_t uq[2]={127,255}; TensorView u{nullptr,2,{2,1},"",uq,2,1.0f,128}; CHECK(ctc_greedy(u,CtcLayout::CT,{"","A"},&mean,&minc)=="A");
    int8_t iq[2]={-128,127}; TensorView qi{nullptr,2,{2,1},"",iq,1,1.0f,std::numeric_limits<int32_t>::min()}; CHECK(ctc_greedy(qi,CtcLayout::CT,{"","A"},&mean,&minc)=="A");
    uint16_t hq[2]={0x0000,0x3c00}; TensorView hf{nullptr,2,{2,1},"",hq,3,1.0f,0}; CHECK(ctc_greedy(hf,CtcLayout::CT,{"","A"},&mean,&minc)=="A");
    // Consecutive A probabilities are .75 and .9: the collapsed character
    // takes the maximum confidence, rather than the first or their mean.
    float repeated_data[4]={0,std::log(3.0f),0,std::log(9.0f)};
    TensorView repeated{repeated_data,4,{2,2},""};
    CHECK(ctc_greedy(repeated,CtcLayout::TC,{"","A"},&mean,&minc)=="A");
    CHECK_NEAR(mean,.9,1e-6); CHECK_NEAR(minc,.9,1e-6);
    TensorView bad{x,0,{2,3},""}; CHECK(ctc_greedy(bad,CtcLayout::CT,{"","A"},&mean,&minc).empty());
    TensorView mismatch{x,6,{4,2},""}; CHECK(ctc_greedy(mismatch,CtcLayout::CT,{"","A"},&mean,&minc).empty());
    TextPattern worst_pattern; TextPatternSegment seg; seg.set={static_cast<uint32_t>('A')};seg.min=0;seg.max=16; for(int i=0;i<8;++i)worst_pattern.push_back(seg); CHECK(text_matches(std::string(128,'A'),{worst_pattern}));
    return 0;
}
